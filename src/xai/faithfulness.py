"""
Faithfulness validation for visual attribution maps:
- Pixel deletion test (AUC of deletion curve)
- Model-randomization sanity check (Adebayo et al.)
"""

import copy
import numpy as np
import torch


def deletion_faithfulness_curve(model, image_tensor, cam_map, fractions=[0.05, 0.10, 0.15, 0.20, 0.25], device="cpu"):
    """
    Measures drop in predicted probability as top salient pixels are masked with black/mean pixels.
    A faithful explanation will exhibit a rapid drop in predicted disease confidence.
    """
    model.eval()
    model.to(device)
    img = image_tensor.clone().to(device)

    with torch.no_grad():
        initial_logits = model(img)
        initial_probs = torch.softmax(initial_logits, dim=-1)
        pred_class = torch.argmax(initial_probs, dim=-1).item()
        base_prob = initial_probs[0, pred_class].item()

    flattened_cam = cam_map.flatten()
    sorted_indices = np.argsort(flattened_cam)[::-1]  # Most salient first
    h, w = cam_map.shape

    drops = {"base_prob": base_prob, "curve": []}

    for frac in fractions:
        num_pixels_to_mask = int(len(sorted_indices) * frac)
        masked_img = img.clone()

        mask_idx = sorted_indices[:num_pixels_to_mask]
        mask_y = mask_idx // w
        mask_x = mask_idx % w

        # Zero out the top pixels
        masked_img[0, :, mask_y, mask_x] = 0.0

        with torch.no_grad():
            new_logits = model(masked_img)
            new_prob = torch.softmax(new_logits, dim=-1)[0, pred_class].item()

        prob_drop = base_prob - new_prob
        drops["curve"].append({
            "fraction_deleted": frac,
            "new_prob": new_prob,
            "probability_drop": prob_drop
        })

    return drops


def model_randomization_sanity_check(trained_explainer, test_tensor, architecture="densenet121"):
    """
    Performs Adebayo sanity check by re-initializing network parameters.
    True explanation heatmaps MUST degrade and diverge from trained heatmaps.
    """
    # Generate trained heatmap
    orig_cam = trained_explainer.generate_heatmap(test_tensor)

    # Build randomized model copy
    from ..models.classifiers import build_model
    random_model = build_model(architecture=architecture, pretrained=False)
    from .gradcam import RetinalXAIExplainer
    random_explainer = RetinalXAIExplainer(random_model, architecture=architecture)
    rand_cam = random_explainer.generate_heatmap(test_tensor)

    # Compute correlation between original and randomized heatmap
    corr = np.corrcoef(orig_cam.flatten(), rand_cam.flatten())[0, 1]
    return {
        "correlation_with_random": float(corr),
        "sanity_check_passed": bool(corr < 0.6)  # True if heatmap is dependent on learned weights
    }
