""" Routes each scan: automated pass if the conformal set has one label, else deferral.
    Deferrals get a Grad-CAM heatmap and no predicted class.
    inputs: weights and calibration metadata from deploy_artifacts/
    outputs: {model}_triage_telemetry.csv, human_review_queue/, automated_pass_audit/
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import csv
import json
import numpy as np
import cv2
from model import BaselineCNN, HybridDualTopology, ViTOnlyBranch


# applies the same conformal rule to every scan
class ConformalTriageDaemon:
    """ Class-conditional conformal routing for single scans. """

    def __init__(self, trained_model: nn.Module, safety_floor_per_class: dict, temperature: float = 1.0):
        """ Stores the model and calibrated thresholds.
            inputs: model, {class: min probability to enter the set}, temperature
            outputs: none
        """
        self.model = trained_model
        self.model.eval()
        self.safety_floor_per_class = safety_floor_per_class
        self.temperature = temperature

    # single entry point for the routing decision
    def evaluate_tensor(self, input_tensor, scan_id, true_label=None):
        """ Builds the prediction set for one scan and routes it.
            inputs: input_tensor [1, 1, 224, 224], scan_id, true_label (telemetry only)
            outputs: dict with status, heatmap path, set size, coverage flag, probs
        """
        # same temperature as calibration
        outputs = self.model(input_tensor)
        scaled_outputs = outputs / self.temperature
        softmax_probs = torch.softmax(scaled_outputs, dim=1).squeeze(0)

        # each label is tested against its own threshold
        prediction_set = [
            idx for idx, prob in enumerate(softmax_probs)
            if prob >= self.safety_floor_per_class[idx]
        ]

        # telemetry only: was the true label in the set?
        true_label_in_set = (true_label in prediction_set) if true_label is not None else None

        # one label = automated pass, otherwise defer
        if len(prediction_set) != 1:
            result = self._trigger_deferral(input_tensor, outputs, scan_id)
        else:
            winning_class = prediction_set[0]
            # audit heatmap only, never shown to a clinician
            heatmap_path = self._save_automated_pass_heatmap(input_tensor, outputs, winning_class, scan_id)
            result = {"status": "AUTOMATED_PASS", "prediction": winning_class, "heatmap_path": heatmap_path}

        result["true_label_in_prediction_set"] = true_label_in_set
        result["prediction_set_size"] = len(prediction_set)
        # raw probs logged for later alpha sweeps
        result["probs"] = softmax_probs.detach().cpu().tolist()
        return result

    # shared by both routing paths
    def _compute_gradcam(self, outputs: torch.Tensor, target_class: int):
        """ Grad-CAM heatmap for one class from the CNN feature map.
            inputs: logits, target class index
            outputs: heatmap in [0, 1], or None for the ViT-only model
        """
        # ViT-only has no hooked feature map
        if self.model.activations is None:
            return None

        self.model.zero_grad()
        outputs[0, target_class].backward()

        gradients = self.model.gradients.detach()
        activations = self.model.activations.detach()

        # pooled gradients = per-channel importance
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for c in range(activations.shape[1]):
            activations[:, c, :, :] *= pooled_gradients[c]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU: keep supporting evidence only
        if np.max(heatmap) != 0:  # avoid divide-by-zero on an empty map
            heatmap /= np.max(heatmap)
        return heatmap

    # same overlay for both paths
    def _render_and_save(self, input_tensor: torch.Tensor, heatmap: np.ndarray, export_path: str) -> str:
        """ Overlays the heatmap on the scan and saves it.
            inputs: scan tensor, heatmap, output path
            outputs: output path
        """
        raw_img = input_tensor.squeeze().detach().cpu().numpy()
        raw_img = (raw_img * 0.5) + 0.5  # undo Normalize(0.5, 0.5)
        mock_bgr = cv2.cvtColor((raw_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        heatmap_resized = cv2.resize(heatmap, (mock_bgr.shape[1], mock_bgr.shape[0]))
        colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(colored_heatmap, 0.4, mock_bgr, 0.6, 0)  # 40% heatmap, 60% scan

        cv2.imwrite(export_path, superimposed_img)
        return export_path

    # deferral shows where the model looked, not what it guessed
    def _trigger_deferral(self, input_tensor: torch.Tensor, outputs: torch.Tensor, scan_id: str):
        """ Builds the deferral result and saves its heatmap if supported.
            inputs: scan tensor, logits, scan_id
            outputs: dict (DEFERRED, class "UNCERTAIN", heatmap path)
        """
        # heatmap for the top-scoring class
        winning_class = torch.argmax(outputs).item()
        heatmap = self._compute_gradcam(outputs, winning_class)

        export_path = None
        if heatmap is not None:
            export_path = os.path.join("human_review_queue", f"deferral_{scan_id}.png")
            self._render_and_save(input_tensor, heatmap, export_path)

        # class withheld so it can't anchor the radiologist
        return {
            "status": "DEFERRED",
            "predicted_class": "UNCERTAIN",
            "heatmap_path": export_path or ""
        }

    # lets automated passes be audited offline
    def _save_automated_pass_heatmap(self, input_tensor: torch.Tensor, outputs: torch.Tensor,
                                       winning_class: int, scan_id: str) -> str:
        """ Saves an audit heatmap for an automated scan.
            inputs: scan tensor, logits, class index, scan_id
            outputs: image path, or "" if unsupported
        """
        heatmap = self._compute_gradcam(outputs, winning_class)
        if heatmap is None:
            return ""
        export_path = os.path.join("automated_pass_audit", f"pass_{scan_id}_pred{winning_class}.png")
        return self._render_and_save(input_tensor, heatmap, export_path)


# evaluation run on the held-out split
if __name__ == "__main__":
    print("--- Triage Main ---")

    # model to evaluate
    model_name = "hybrid_topology"
    #model_name = "baseline_cnn"
    #model_name = "vit_only"

    # thresholds saved by train_pipeline.py
    meta_path = f"deploy_artifacts/{model_name}_meta.json"
    try:
        with open(meta_path, "r") as f:
            calibration_data = json.load(f)
            # JSON keys are strings, convert back to int
            safety_floor_per_class = {
                int(k): v for k, v in calibration_data["safety_floor_per_class"].items()
            }
            temperature = calibration_data.get("temperature", 1.0)
            malignant_index_from_meta = calibration_data.get("malignant_index")
            print(f"Safety floors loaded (per class): {safety_floor_per_class}")
            print(f"Temperature Loaded: {temperature:.4f}")
    except FileNotFoundError:
        print("CRITICAL: Calibration data not found. Run training pipeline first.")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Target: {device.type.upper()}")

    if model_name == "baseline_cnn":
        selected_model = BaselineCNN().to(device)
    elif model_name == "vit_only":
        selected_model = ViTOnlyBranch().to(device)
    else:
        selected_model = HybridDualTopology().to(device)

    # load trained weights
    weights_path = f"deploy_artifacts/{model_name}_weights.pth"
    try:
        selected_model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Hybrid Dual-Topology Weights Injected.")
    except FileNotFoundError:
        print(f"CRITICAL: Weights not found at {weights_path}.")
        exit(2)

    # inference mode
    selected_model.eval()

    daemon = ConformalTriageDaemon(trained_model=selected_model, safety_floor_per_class=safety_floor_per_class, temperature=temperature)

    print("--- Daemon Online and Ready ---")
    # same transform as the training pipeline's eval transform
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # dataset root (one subfolder per class)
    data_path = r"clean_training_set"
    master_dataset = datasets.ImageFolder(root=data_path, transform=transform_pipeline)

    # malignant index from folder names
    malignant_index = None
    for class_name, class_idx in master_dataset.class_to_idx.items():
        if class_name.strip().lower() == "malignant":
            malignant_index = class_idx
            break
    print(f"Class mapping: {master_dataset.class_to_idx}")
    if malignant_index is None:
        print("WARNING: no class folder literally named 'Malignant' found - "
              "defaulting malignant_index to 1. Verify this against the "
              "class mapping printed above before trusting Test 4's numbers.")
        malignant_index = 1
    # index saved at training time wins, thresholds were calibrated against it
    if malignant_index_from_meta is not None and malignant_index_from_meta != malignant_index:
        print(f"WARNING: malignant_index detected here ({malignant_index}) does not match "
              f"the one recorded in {model_name}_meta.json ({malignant_index_from_meta}) from "
              f"training. Using the value from meta.json, since that's what the per-class "
              f"alpha/safety floors were actually calibrated against.")
        malignant_index = malignant_index_from_meta
    benign_index = 1 - malignant_index

    # recreate the 80/10/10 split
    total_samples = len(master_dataset)
    train_size = int(0.80 * total_samples)
    val_size = int(0.10 * total_samples)
    cal_size = total_samples - train_size - val_size

    # seed 42 must match train_pipeline.py, or eval data could overlap training
    _, val_data, _ = random_split(
        master_dataset, [train_size, val_size, cal_size],
        generator=torch.Generator().manual_seed(42)
    )

    # batch size 1: one scan at a time
    test_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    print(f"[*] Commencing Triage Audit on {len(val_data)} unseen clinical scans...")

    # running totals
    telemetry_log = []
    deferred_scans = 0
    automated_passes = 0
    correct_automated = 0
    false_negatives = 0
    covered_scans = 0  # true label was in the prediction set
    os.makedirs("human_review_queue", exist_ok=True)
    os.makedirs("automated_pass_audit", exist_ok=True)

    for idx, (image, label) in enumerate(test_loader):
        image = image.to(device)
        image.requires_grad = True  # lets backward reach the Grad-CAM hook
        true_label = label.item()

        scan_id = f"SCAN_{idx:04d}"

        # true_label is telemetry only, not used for routing
        result = daemon.evaluate_tensor(image, scan_id, true_label=true_label)

        status = result["status"]
        predicted_class = result.get("prediction", "UNCERTAIN")
        is_false_negative = False

        if status == "DEFERRED":
            deferred_scans += 1
        elif status == "AUTOMATED_PASS":
            automated_passes += 1
            if predicted_class == true_label:
                correct_automated += 1
            # false negative: malignant scan automated as benign
            if true_label == malignant_index and predicted_class != malignant_index:
                is_false_negative = True
                false_negatives += 1

        if result.get("true_label_in_prediction_set"):
            covered_scans += 1

        # one telemetry row per scan
        probs = result.get("probs", [None, None])
        telemetry_log.append({
            "Scan_ID": scan_id,
            "True_Label": true_label,
            "Model_Status": status,
            "Predicted_Class": predicted_class,
            "Prediction_Set_Size": result.get("prediction_set_size"),
            "True_Label_In_Prediction_Set": result.get("true_label_in_prediction_set"),
            "Is_False_Negative": is_false_negative,
            "Heatmap_Path": result.get("heatmap_path", ""),
            "P_Benign": probs[benign_index] if len(probs) > benign_index else None,
            "P_Malignant": probs[malignant_index] if len(probs) > malignant_index else None,
        })

    # write telemetry CSV
    csv_file = f"{model_name}_triage_telemetry.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Scan_ID", "True_Label", "Model_Status", "Predicted_Class",
            "Prediction_Set_Size", "True_Label_In_Prediction_Set",
            "Is_False_Negative", "Heatmap_Path", "P_Benign", "P_Malignant",
        ])
        writer.writeheader()
        writer.writerows(telemetry_log)

    # console summary
    print("\n==================================================")
    print(f"=== TELEMETRY REPORT: {model_name.upper()} ===")
    print("==================================================")
    print(f"Total Scans Processed: {len(val_data)}")
    print(f"Deferred to Human:     {deferred_scans} ({(deferred_scans/len(val_data))*100:.1f}%)")
    print(f"Automated Passes:      {automated_passes} ({(automated_passes/len(val_data))*100:.1f}%)")
    if automated_passes > 0:
        print(f"Automated Accuracy:    {(correct_automated/automated_passes)*100:.1f}%")
    print(f"False Negatives:       {false_negatives} (malignant scans auto-passed as benign)")
    print(f"Empirical Coverage:    {(covered_scans/len(val_data))*100:.1f}% "
          f"({covered_scans}/{len(val_data)} scans had the true label inside C(x))")
    print(f"                       Per-class alpha (target 1-alpha coverage) from calibration: "
          f"{calibration_data.get('alpha_per_class', {})}")
    print("==================================================")
    print(f"CSV saved to {csv_file}")
    print(f"Automated-pass Grad-CAM audit images saved to automated_pass_audit/")
