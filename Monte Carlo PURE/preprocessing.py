"""
Zero-suppression preprocessing for CYGNO images.
Estimates pedestal per image, removes background, subtracts pedestal.
"""

import numpy as np
from scipy import ndimage


def estimate_pedestal(image, low_quantile=0.1, high_quantile=0.9):
    """
    Estimate pedestal (noise floor) from an image.

    Args:
        image: 2D array (H, W)
        low_quantile, high_quantile: percentile bounds to define "pedestal region"

    Returns:
        pedestal_value: scalar (the mode within the low-high region)
        pedestal_sigma: scalar (std of pixels in pedestal region)
    """
    # Define the "pedestal region" as the middle 80% of the distribution
    low_pct = np.percentile(image, low_quantile * 100)
    high_pct = np.percentile(image, high_quantile * 100)

    # Pixels in this region are assumed to be pure noise
    mask = (image >= low_pct) & (image <= high_pct)
    pedestal_pixels = image[mask]

    # Mode = most common value in pedestal region
    hist, edges = np.histogram(pedestal_pixels, bins=50)
    mode_bin = np.argmax(hist)
    pedestal_value = (edges[mode_bin] + edges[mode_bin + 1]) / 2

    # Sigma = standard deviation of pedestal pixels
    pedestal_sigma = np.std(pedestal_pixels)

    return pedestal_value, pedestal_sigma


def zero_suppress(image, k_sigma=3.0):
    """
    Apply zero-suppression to an image.

    Args:
        image: 2D array (H, W), normalized to [0, 1] or raw ADC counts
        k_sigma: threshold multiplier (default 3-5 is typical)

    Returns:
        suppressed: 2D array, same shape as input. Pixels <= threshold set to 0.
                   Surviving pixels have pedestal subtracted (so values = signal above noise).
        pedestal: estimated pedestal value
        mask: boolean mask of which pixels survived
    """
    pedestal, sigma = estimate_pedestal(image)
    threshold = pedestal + k_sigma * sigma

    # Create suppressed image
    suppressed = image.copy()
    mask = suppressed > threshold
    suppressed[~mask] = 0.0
    suppressed[mask] -= pedestal  # Subtract pedestal from surviving pixels
    suppressed = np.maximum(suppressed, 0.0)  # Clip negatives

    return suppressed, pedestal, mask


def connected_component_filter(image, min_size=2):
    """
    Remove isolated pixels (keep only connected clusters of size >= min_size).

    Args:
        image: 2D binary mask (True = signal, False = background)
        min_size: minimum cluster size to keep

    Returns:
        filtered: 2D boolean mask with small clusters removed
    """
    labeled, n_components = ndimage.label(image)

    filtered = np.zeros_like(image, dtype=bool)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if np.sum(comp_mask) >= min_size:
            filtered[comp_mask] = True

    return filtered


def preprocess_image(image, k_sigma=3.0, min_component_size=2):
    """
    Full zero-suppression pipeline for one image.

    Args:
        image: 2D array (H, W), in any scale (e.g. [0,1] or [0,255])
        k_sigma: threshold = pedestal + k_sigma * sigma
        min_component_size: minimum size of cluster to keep (removes noise spikes)

    Returns:
        suppressed: 2D array with background removed and pedestal subtracted
        pedestal: estimated pedestal value
        signal_mask: boolean mask of pixels that survived suppression + CC filter
    """
    # Step 1: Zero-suppression
    suppressed, pedestal, raw_mask = zero_suppress(image, k_sigma=k_sigma)

    # Step 2: Connected-component filter
    signal_mask = connected_component_filter(raw_mask, min_size=min_component_size)

    # Apply final mask to suppressed image
    suppressed[~signal_mask] = 0.0

    return suppressed, pedestal, signal_mask


def batch_preprocess(images, k_sigma=3.0, min_component_size=2):
    """
    Apply preprocessing to a batch of images.

    Args:
        images: array of shape (N, H, W) or (N, H, W, 1)
        k_sigma: threshold multiplier
        min_component_size: min cluster size

    Returns:
        suppressed_images: (N, H, W) preprocessed
        pedestals: (N,) pedestal values
        signal_masks: (N, H, W) boolean masks
    """
    if images.ndim == 4:
        images = images.squeeze(-1)  # Remove channel dim if present

    N = len(images)
    H, W = images.shape[1:]

    suppressed_images = np.zeros((N, H, W), dtype=np.float32)
    pedestals = np.zeros(N, dtype=np.float32)
    signal_masks = np.zeros((N, H, W), dtype=bool)

    for i in range(N):
        sup, ped, mask = preprocess_image(
            images[i], k_sigma=k_sigma, min_component_size=min_component_size
        )
        suppressed_images[i] = sup
        pedestals[i] = ped
        signal_masks[i] = mask

        if (i + 1) % max(1, N // 10) == 0:
            print(f"[preprocess] {i+1}/{N} images processed")

    return suppressed_images, pedestals, signal_masks


def estimate_energy_from_image(image):
    """
    Simple energy proxy: total signal above noise.
    Use this to bin results for energy-dependent evaluation.

    Args:
        image: 2D array (preprocessed, pedestal already subtracted)

    Returns:
        energy: scalar (sum of pixel values)
    """
    return np.sum(image)
