from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Dict, Any
from PIL import Image


def _ensure_gray(img: Image.Image) -> np.ndarray:
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def _clahe_enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    # Horizontal line removal
    h_kernel_w = max(15, w // 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    # Vertical line removal (light)
    v_kernel_h = max(15, h // 12)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(lines))
    return cleaned


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    # Find largest connected component (assume digit)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    # skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    mask = (labels == idx).astype(np.uint8) * 255
    return mask


def _center_and_resize(mask: np.ndarray, gray_ref: np.ndarray, size: Tuple[int, int], pixelate: bool = False) -> np.ndarray:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # fallback: just center original
        box = (0, 0, w, h)
    else:
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = int(0.1 * max(x1 - x0 + 1, y1 - y0 + 1))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
        box = (x0, y0, x1, y1)
    x0, y0, x1, y1 = box
    roi_mask = mask[y0:y1 + 1, x0:x1 + 1]
    roi_gray = gray_ref[y0:y1 + 1, x0:x1 + 1]
    # Resize with aspect ratio into square canvas
    target_w, target_h = size
    scale = min(target_w / roi_mask.shape[1], target_h / roi_mask.shape[0])
    new_w, new_h = max(1, int(roi_mask.shape[1] * scale)), max(1, int(roi_mask.shape[0] * scale))
    interp = cv2.INTER_NEAREST if pixelate else cv2.INTER_AREA
    roi_gray_res = cv2.resize(roi_gray, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    oy = (target_h - new_h) // 2
    ox = (target_w - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = roi_gray_res
    return canvas


def _normalize_stroke(gray: np.ndarray) -> np.ndarray:
    # Simple occupancy-based normalization
    thr = np.median(gray)
    binary = (gray > thr).astype(np.uint8) * 255
    occ = (binary > 0).mean()
    if occ < 0.05:
        # too thin/small -> dilate
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, k, iterations=1)
    elif occ > 0.4:
        # too thick -> erode
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.erode(binary, k, iterations=1)
    # Blend binary back to grayscale space to keep antialiasing mild
    gray_norm = np.where(binary > 0, gray, 0)
    return gray_norm


def preprocess_digit_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (28, 28),
    remove_lines: bool = True,
    normalize_stroke_thickness: bool = True,
    pixelate: bool = False,
    return_debug: bool = False,
) -> Dict[str, Any]:
    steps = {}
    gray = _ensure_gray(image)
    steps['gray'] = gray
    den = cv2.medianBlur(gray, 3)
    steps['denoise'] = den
    enh = _clahe_enhance(den)
    steps['enhanced'] = enh
    # Adaptive threshold (white foreground)
    bin_img = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    bin_inv = 255 - bin_img
    steps['binary_inv'] = bin_inv
    if remove_lines:
        no_lines = _remove_ruled_lines(bin_inv)
        steps['no_lines'] = no_lines
    else:
        no_lines = bin_inv
    # Largest component as mask
    mask = _largest_component_mask(no_lines)
    steps['mask'] = mask
    # Use enhanced gray as reference, masked
    gray_masked = cv2.bitwise_and(enh, mask)
    centered = _center_and_resize(mask, gray_masked, target_size, pixelate=pixelate)
    steps['centered'] = centered
    if normalize_stroke_thickness:
        norm = _normalize_stroke(centered)
    else:
        norm = centered
    steps['normalized'] = norm
    # Normalize to [0,1]
    arr = norm.astype(np.float32) / 255.0
    result = {
        'array': arr,
        'preview_uint8': (arr * 255).astype(np.uint8),
    }
    if return_debug:
        result['debug'] = steps
    return result
