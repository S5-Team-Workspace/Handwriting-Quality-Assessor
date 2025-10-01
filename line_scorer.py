from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image


def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def analyze_line(image: Image.Image) -> Dict[str, Any]:
    """
    Extract geometric features for line quality assessment.
    Returns:
      angle_deg: fitted line angle in degrees (-90..90, 0 is horizontal)
      angle_dev: absolute deviation from dominant angle (computed later vs expected)
      straightness: std of perpendicular distances normalized by line length (lower is better)
      length_px: estimated line length in pixels
      overlay: visualization image (RGB)
    """
    gray = _pil_to_gray_np(image)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # HoughLinesP to get a primary segment (fallback if none)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)

    h, w = gray.shape
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    angle_deg = 0.0
    length_px = 0.0

    if lines is not None and len(lines) > 0:
        # Choose the longest line
        best = None
        best_len = -1
        for l in lines:
            x1, y1, x2, y2 = l[0]
            L = np.hypot(x2 - x1, y2 - y1)
            if L > best_len:
                best_len = L
                best = (x1, y1, x2, y2)
        x1, y1, x2, y2 = best
        length_px = float(best_len)

        # Angle in degrees: horizontal 0°, positive counterclockwise
        angle_rad = np.arctan2(-(y2 - y1), (x2 - x1))  # invert y to make up positive
        angle_deg = float(np.degrees(angle_rad))

        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # Fall back to fit a line using all edge points
        ys, xs = np.nonzero(edges)
        if xs.size > 20:
            # Fit y = ax + b
            a, b = np.polyfit(xs, ys, 1)
            angle_rad = np.arctan(-a)  # slope to angle (invert for image y)
            angle_deg = float(np.degrees(angle_rad))
            # approximate length as diagonal extent
            length_px = float(np.hypot(w, h) / 2)
        else:
            # No line-like structure
            return {
                "angle_deg": 0.0,
                "angle_dev": None,
                "straightness": 1.0,
                "length_px": 0.0,
                "overlay": overlay[..., ::-1],  # BGR->RGB for PIL/Streamlit
            }

    # Straightness via distance to fitted line (using all edge points)
    ys, xs = np.nonzero(edges)
    straightness = 1.0
    if xs.size > 20:
        # Fit a robust line using cv2.fitLine (vx, vy, x0, y0)
        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        # Distance of each point to the line defined by (vx,vy) passing (x0,y0)
        # formula: |(vy)*(x-x0) - (vx)*(y-y0)|
        d = np.abs(vy * (xs - x0) - vx * (ys - y0))
        # Normalize by approximate length to make scale-free
        straightness = float(np.std(d) / max(1.0, length_px))

    return {
        "angle_deg": angle_deg,
        "angle_dev": None,  # computed in scorer vs desired angle
        "straightness": straightness,
        "length_px": length_px,
        "overlay": overlay[..., ::-1],  # BGR->RGB
    }


def score_line(image: Image.Image, line_type: str = "sleeping") -> Dict[str, Any]:
    """
    Rule-based line scoring with angle and straightness.
    line_type: "sleeping" (target 0°) or "slanting" (target 45°)
    Returns: {quality_score: 0..100, angle_deg, angle_dev, straightness, length_px, overlay}
    """
    feats = analyze_line(image)

    target = 0.0 if line_type.lower().startswith("sleep") else 45.0
    # Compute smallest absolute deviation from target modulo 180
    dev = abs(((feats["angle_deg"] - target + 90) % 180) - 90)
    feats["angle_dev"] = float(dev)

    # Angle contribution (tolerance ~12°)
    theta_tol = 12.0
    s_angle = max(0.0, 1.0 - dev / theta_tol)

    # Straightness contribution (lower is better; tolerance ~0.03)
    tau = 0.03
    s_straight = max(0.0, 1.0 - feats["straightness"] / tau)

    score = int(round(100 * (0.6 * s_angle + 0.4 * s_straight)))

    return {
        "quality_score": int(np.clip(score, 0, 100)),
        "angle_deg": feats["angle_deg"],
        "angle_dev": feats["angle_dev"],
        "straightness": feats["straightness"],
        "length_px": feats["length_px"],
        "overlay": feats["overlay"],
    }


if __name__ == "__main__":
    # Minimal smoke test: requires sample_line.png
    try:
        img = Image.open("sample_line.png")
        out = score_line(img, "sleeping")
        print({k: v for k, v in out.items() if k != "overlay"})
    except FileNotFoundError:
        print("Place a sample_line.png next to this file to test.")
