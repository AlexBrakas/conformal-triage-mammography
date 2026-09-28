""" Models: baseline CNN, hybrid CNN + ViT, and ViT-only ablation.
    All take 1-channel 224x224 scans and return 2-class logits (benign / malignant).
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

# timm is only needed for the ViT models
try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# holds the tensors Grad-CAM needs
class BaseMedicalNetwork(nn.Module):
    """ Parent class storing the Grad-CAM feature map and its gradient. """
    def __init__(self):
        super(BaseMedicalNetwork, self).__init__()
        self.gradients = None
        self.activations = None

    # backward hook target
    def activations_hook(self, grad):
        """ Saves the feature-map gradient.
            inputs: grad (Tensor)
            outputs: none
        """
        self.gradients = grad


# pretrained convs expect RGB, scans are 1-channel
def adapt_conv_to_grayscale(conv: nn.Conv2d) -> nn.Conv2d:
    """ Builds a 1-channel copy of a pretrained RGB conv.
        inputs: conv (nn.Conv2d)
        outputs: nn.Conv2d with the RGB filters averaged into one channel
    """
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        # averaging keeps the pretrained edge/texture filters
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


# reference model for the comparison
class BaselineCNN(BaseMedicalNetwork):
    """ ImageNet ResNet-18 on grayscale input, last residual block trainable. """
    def __init__(self):
        super(BaselineCNN, self).__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = adapt_conv_to_grayscale(backbone.conv1)

        # train only layer4[-1] (~4.7M of 11.2M params) to limit overfitting
        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer4[-1].parameters():
            param.requires_grad = True

        # drop avgpool/fc so the 7x7 map is available for Grad-CAM
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 2)  # 512 = ResNet-18 output channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Runs the CNN and keeps the feature map for Grad-CAM.
            inputs: x [batch, 1, 224, 224]
            outputs: logits [batch, 2]
        """
        feature_map = self.feature_extractor(x)  # [batch, 512, 7, 7]
        # hook only when gradients are tracked
        if feature_map.requires_grad:
            feature_map.register_hook(self.activations_hook)
            self.activations = feature_map
        pooled = self.pool(feature_map).flatten(1)
        logits = self.classifier(pooled)
        return logits


# main model: CNN (local texture) + ViT (global context), fused late
class HybridDualTopology(BaseMedicalNetwork):
    """ ResNet-18 and DINOv2 ViT-S/14 branches, concatenated before the classifier. """
    def __init__(self):
        super(HybridDualTopology, self).__init__()
        if not _HAS_TIMM:
            raise ImportError(
                "timm is required for the ViT branch. Install with: pip install timm"
            )

        # local branch: same freeze scheme as BaselineCNN
        cnn_backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        cnn_backbone.conv1 = adapt_conv_to_grayscale(cnn_backbone.conv1)
        for param in cnn_backbone.parameters():
            param.requires_grad = False
        for param in cnn_backbone.layer4[-1].parameters():
            param.requires_grad = True
        self.cnn_block = nn.Sequential(*list(cnn_backbone.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        # global branch: self-supervised DINOv2 transfers well to small datasets
        vit_backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,          # return pooled features, no head
            dynamic_img_size=True,  # native 518px, interpolate position embeddings to 224
        )
        vit_backbone.patch_embed.proj = adapt_conv_to_grayscale(vit_backbone.patch_embed.proj)

        # train only the last block and final norm
        for param in vit_backbone.parameters():
            param.requires_grad = False
        for param in vit_backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in vit_backbone.norm.parameters():
            param.requires_grad = True

        self.vit_backbone = vit_backbone

        # per-branch norm puts both features on a similar scale before fusion
        self.local_norm = nn.LayerNorm(512)
        self.global_norm = nn.LayerNorm(384)

        self.classifier = nn.Linear(512 + 384, 2)  # ResNet 512 + DINOv2 384

        # aux heads give each branch its own training signal (modality competition)
        self.local_aux_classifier = nn.Linear(512, 2)
        self.global_aux_classifier = nn.Linear(384, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Inference pass (aux outputs dropped).
            inputs: x [batch, 1, 224, 224]
            outputs: fused logits [batch, 2]
        """
        fused_logits, _, _ = self.forward_with_aux(x)
        return fused_logits

    def forward_with_aux(self, x: torch.Tensor):
        """ Forward pass that also returns the aux logits (training only).
            inputs: x [batch, 1, 224, 224]
            outputs: (fused, local_aux, global_aux) logits, each [batch, 2]
        """
        local_feat_map = self.cnn_block(x)  # [batch, 512, 7, 7]
        # Grad-CAM uses the CNN branch only, the ViT has no spatial map
        if local_feat_map.requires_grad:
            local_feat_map.register_hook(self.activations_hook)
            self.activations = local_feat_map
        local_flat = self.cnn_pool(local_feat_map).flatten(1)  # [batch, 512]
        local_flat = self.local_norm(local_flat)

        global_flat = self.vit_backbone(x)  # [batch, 384]
        global_flat = self.global_norm(global_flat)

        combined = torch.cat((local_flat, global_flat), dim=1)
        fused_logits = self.classifier(combined)

        local_aux_logits = self.local_aux_classifier(local_flat)
        global_aux_logits = self.global_aux_classifier(global_flat)

        return fused_logits, local_aux_logits, global_aux_logits


# ablation: isolates the ViT branch from the fusion step
class ViTOnlyBranch(BaseMedicalNetwork):
    """ DINOv2 ViT-S/14 alone. No Grad-CAM (no CNN feature map). """
    def __init__(self):
        super(ViTOnlyBranch, self).__init__()
        if not _HAS_TIMM:
            raise ImportError(
                "timm is required for the ViT branch. Install with: pip install timm"
            )

        # same backbone and freeze scheme as the hybrid's ViT branch
        vit_backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        vit_backbone.patch_embed.proj = adapt_conv_to_grayscale(vit_backbone.patch_embed.proj)

        for param in vit_backbone.parameters():
            param.requires_grad = False
        for param in vit_backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in vit_backbone.norm.parameters():
            param.requires_grad = True

        self.vit_backbone = vit_backbone
        self.norm = nn.LayerNorm(384)  # same as the hybrid's ViT branch
        self.classifier = nn.Linear(384, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Runs the ViT branch and classifies the pooled features.
            inputs: x [batch, 1, 224, 224]
            outputs: logits [batch, 2]
        """
        feat = self.vit_backbone(x)  # [batch, 384]
        feat = self.norm(feat)
        logits = self.classifier(feat)
        return logits
