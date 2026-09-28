""" Trains the three models, fits temperature, calibrates conformal thresholds, saves artifacts.
    inputs: ImageFolder dataset (benign / malignant subfolders)
    outputs: deploy_artifacts/{model}_weights.pth, _calibration_scores.npz, _meta.json
"""
import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# project models
from model import HybridDualTopology, BaselineCNN, ViTOnlyBranch

# GPU if available; global seed (data splits use their own seeds below)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
torch.manual_seed(22)
np.random.seed(22)


# early stopping on val loss limits overfitting
def train_model(model, train_loader, val_loader, max_epochs=50, patience=7, lr=1e-4,
                 weight_decay=1e-4, class_weights=None, aux_loss_weight=0.3):
    """ Fine-tunes with AdamW, early stopping and LR scheduling.
        inputs: model, train/val loaders, training settings, optional class weights
        outputs: model restored to its lowest-val-loss weights
    """
    # class weights offset the benign/malignant imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler patience (5) < early-stop patience (7) so the LR decays first
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    has_aux = hasattr(model, "forward_with_aux")

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())  # best checkpoint so far
    epochs_without_improvement = 0

    print(f"\n--- Starting Model Training (max {max_epochs} epochs, early stopping patience={patience}"
          f"{', aux branch losses enabled (weight=' + str(aux_loss_weight) + ')' if has_aux else ''}) ---")

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if has_aux:
                # fused loss + weighted per-branch aux losses (0.3, untuned)
                fused_logits, local_aux_logits, global_aux_logits = model.forward_with_aux(images)
                loss = (
                    criterion(fused_logits, labels)
                    + aux_loss_weight * criterion(local_aux_logits, labels)
                    + aux_loss_weight * criterion(global_aux_logits, labels)
                )
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # validate on the fused output only, as deployed
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[*] Epoch [{epoch+1}/{max_epochs}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[*] Early stopping at epoch {epoch+1} "
                      f"(no val-loss improvement for {patience} epochs). "
                      f"Best val loss: {best_val_loss:.4f}")
                break

    # restore the best checkpoint, not the last epoch
    model.load_state_dict(best_state)
    return model


# rescales confidence without changing predictions
def fit_temperature(model, temp_loader, max_iter=50, lr=0.01):
    """ Fits a single temperature T on a holdout (Guo et al., 2017).
        inputs: model, holdout loader, LBFGS settings
        outputs: T (float)
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in temp_loader:
            images, labels = images.to(device), labels.to(device)
            logits_list.append(model(images))
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # start above 1, networks are usually overconfident
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    # LBFGS closure: NLL of the scaled logits
    def eval_loss():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    learned_t = temperature.item()
    print(f"Learned temperature: {learned_t:.4f}")
    return learned_t


# per-class thresholds keep coverage within each class
def calibrate_conformal_threshold(model, cal_loader, temperature=1.0, alpha_per_class=None):
    """ Class-conditional (Mondrian) conformal thresholds from the calibration set.
        inputs: model, calibration loader, temperature, {class: alpha} (default 0.05)
        outputs: q_hat, safety_floor, scores (dicts keyed by class index)
    """
    if alpha_per_class is None:
        alpha_per_class = {}

    model.eval()
    scores_per_class = {}

    print(f"Executing class-conditional conformal calibration (T={temperature:.4f})...")
    with torch.no_grad():
        for images, labels in cal_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) / temperature
            softmax_probs = torch.softmax(outputs, dim=1)
            true_class_probs = softmax_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            # score = 1 - true-class probability
            scores = (1.0 - true_class_probs).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for score, label in zip(scores, labels_np):
                scores_per_class.setdefault(int(label), []).append(float(score))

    q_hat_per_class = {}
    safety_floor_per_class = {}
    scores_arrays = {}
    for class_idx, scores_list in scores_per_class.items():
        scores_arr = np.array(scores_list)
        alpha = alpha_per_class.get(class_idx, 0.05)  # default 95% coverage
        n = len(scores_arr)
        # finite-sample quantile level, clipped to [0, 1]
        quantile_val = min(max(np.ceil((n + 1) * (1 - alpha)) / n, 0.0), 1.0)
        # 'higher' is the conservative choice
        q_hat = float(np.quantile(scores_arr, quantile_val, method='higher'))

        q_hat_per_class[class_idx] = q_hat
        safety_floor_per_class[class_idx] = 1.0 - q_hat  # min probability to enter the set
        scores_arrays[class_idx] = scores_arr

        print(f"  Class {class_idx}: n={n}, alpha={alpha}, q_hat={q_hat:.4f}, "
              f"safety_floor={1.0 - q_hat:.4f}")

    return q_hat_per_class, safety_floor_per_class, scores_arrays


# pipeline: split, train, calibrate, save
if __name__ == "__main__":
    print("Ingesting clinical dataset")

    # eval preprocessing: grayscale, 224x224, scale to [-1, 1]
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # train-only augmentation; elsewhere it would break exchangeability
    transform_pipeline_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # small angle keeps anatomy plausible
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # dataset root (one subfolder per class)
    data_path = r"c:/Users/Abrak/Documents/University of London/Fourth year/First Term/Final project/clean_training_set"
    master_dataset = datasets.ImageFolder(root=data_path, transform=transform_pipeline)
    # same files with augmentation, so indices match master_dataset
    master_dataset_train = datasets.ImageFolder(root=data_path, transform=transform_pipeline_train)

    # malignant index from folder names; must match triage_daemon.py
    malignant_index = None
    for class_name, class_idx in master_dataset.class_to_idx.items():
        if class_name.strip().lower() == "malignant":
            malignant_index = class_idx
            break
    print(f"Class mapping: {master_dataset.class_to_idx}")
    if malignant_index is None:
        print("WARNING: no class folder literally named 'Malignant' found - "
              "defaulting malignant_index to 1. Verify this against the "
              "class mapping printed above.")
        malignant_index = 1
    benign_index = 1 - malignant_index
    print(f"Malignant class index: {malignant_index} | Benign class index: {benign_index}")

    # missed malignancy costs more than an extra deferral
    ALPHA_MALIGNANT = 0.01  # 99% coverage
    ALPHA_BENIGN = 0.05     # 95% coverage
    alpha_per_class = {malignant_index: ALPHA_MALIGNANT, benign_index: ALPHA_BENIGN}
    print(f"Per-class alpha: {alpha_per_class}")

    # 80% train / 10% eval / 10% calibration
    total_samples = len(master_dataset)
    train_size = int(0.80 * total_samples)
    val_size = int(0.10 * total_samples)
    cal_size = total_samples - train_size - val_size

    # seed 42 must match triage_daemon.py
    train_data, val_data, cal_data = random_split(
        master_dataset, [train_size, val_size, cal_size],
        generator=torch.Generator().manual_seed(42)
    )
    # same seed and sizes give the same partition on the augmented copy
    train_data_aug, _, _ = random_split(
        master_dataset_train, [train_size, val_size, cal_size],
        generator=torch.Generator().manual_seed(42)
    )

    # holdout for early stopping and temperature, disjoint from calibration/eval
    temp_holdout_size = int(0.10 * len(train_data))
    fit_size = len(train_data) - temp_holdout_size
    fit_data, temp_data = random_split(
        train_data, [fit_size, temp_holdout_size],
        generator=torch.Generator().manual_seed(7)
    )
    # holdout stays unaugmented
    fit_data_aug, _ = random_split(
        train_data_aug, [fit_size, temp_holdout_size],
        generator=torch.Generator().manual_seed(7)
    )

    # num_workers=0 avoids dataloader deadlocks
    train_loader = DataLoader(fit_data_aug, batch_size=32, shuffle=True, num_workers=0)
    temp_loader = DataLoader(temp_data, batch_size=32, shuffle=False, num_workers=0)
    cal_loader = DataLoader(cal_data, batch_size=32, shuffle=False, num_workers=0)

    # inverse-frequency weights (mean 1); Subset indices mapped back to master_dataset
    fit_indices_in_master = [train_data.indices[i] for i in fit_data.indices]
    fit_labels = [master_dataset.targets[i] for i in fit_indices_in_master]
    class_counts = {c: fit_labels.count(c) for c in set(fit_labels)}
    num_classes = len(class_counts)
    raw_weights = {c: num_classes / count for c, count in class_counts.items()}
    weight_sum = sum(raw_weights.values())
    normalized_weights = {c: (w / weight_sum) * num_classes for c, w in raw_weights.items()}
    print(f"Training class counts: {class_counts} | class weights (mean=1): {normalized_weights}")
    class_weights_tensor = torch.tensor(
        [normalized_weights[c] for c in range(num_classes)], dtype=torch.float32
    ).to(device)

    print("Target Models to train")
    models_to_evaluate = {
        "baseline_cnn": BaselineCNN().to(device),
        "hybrid_topology": HybridDualTopology().to(device),
        # ablation: ViT alone, no CNN branch or fusion
        "vit_only": ViTOnlyBranch().to(device),
    }

    os.makedirs("deploy_artifacts", exist_ok=True)

    # identical settings for every model
    for model_name, target_model in models_to_evaluate.items():
        print(f"\n=======================================================")
        print(f"[*] Commencing Pipeline for: {model_name.upper()}")
        print(f"=======================================================")

        print(f"Starting training")
        # weight decay 1e-4: higher gave no benefit in comparable work (Wei et al., 2021)
        trained_model = train_model(
            target_model, train_loader, temp_loader,
            max_epochs=50, patience=7, lr=1e-4, weight_decay=1e-4,
            class_weights=class_weights_tensor,
        )

        print(f"Fitting temperature scaling")
        learned_temperature = fit_temperature(trained_model, temp_loader)

        print(f"Clibrating conformal (class-conditional)")
        q_hat_per_class, safety_floor_per_class, scores_per_class = calibrate_conformal_threshold(
            trained_model, cal_loader, temperature=learned_temperature, alpha_per_class=alpha_per_class
        )

        print(f"Finalizing {model_name} Artifacts")

        torch.save(trained_model.state_dict(), f"deploy_artifacts/{model_name}_weights.pth")

        # raw calibration scores, kept for the alpha sweep
        np.savez(
            f"deploy_artifacts/{model_name}_calibration_scores.npz",
            **{f"class_{k}": v for k, v in scores_per_class.items()}
        )

        # metadata read by triage_daemon.py (JSON keys must be strings)
        calibration_data = {
            "temperature": float(learned_temperature),
            "malignant_index": malignant_index,
            "benign_index": benign_index,
            "alpha_per_class": {str(k): v for k, v in alpha_per_class.items()},
            "q_hat_per_class": {str(k): v for k, v in q_hat_per_class.items()},
            "safety_floor_per_class": {str(k): v for k, v in safety_floor_per_class.items()},
        }
        with open(f"deploy_artifacts/{model_name}_meta.json", "w") as f:
            json.dump(calibration_data, f, indent=4)

        print(f"{model_name} saved successfully")

    print("\n FullTraining complete")
