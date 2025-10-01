"""
Alphabet recognition via a simple Bayesian Network using pgmpy.
- Features are extracted from a grayscale image using OpenCV/numpy.
- Structure: Letter -> each feature (naive Bayes). Fit CPDs with MLE.
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple
from pathlib import Path
import pickle

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


FEATURES = [
    'aspect_ratio',
    'stroke_density',
    'edge_density',
    'symmetry_v',
    'symmetry_h',
    'hu1', 'hu2', 'hu3'
]


def extract_features_alphabet(pil_image) -> Dict[str, float]:
    # Convert to grayscale numpy
    img = np.array(pil_image.convert('L'))
    h, w = img.shape
    # Normalize to [0,1]
    f = img.astype(np.float32) / 255.0
    # Threshold
    _, th = cv2.threshold((f*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_inv = 255 - th
    # Aspect ratio of bounding box
    coords = cv2.findNonZero(th_inv)
    if coords is not None:
        x,y,ww,hh = cv2.boundingRect(coords)
        aspect_ratio = (ww / max(1, hh))
        roi = th_inv[y:y+hh, x:x+ww]
    else:
        aspect_ratio = 1.0
        roi = th_inv
    # Stroke density
    stroke_density = (th_inv > 0).mean()
    # Edge density
    edges = cv2.Canny(th_inv, 50, 150)
    edge_density = (edges > 0).mean()
    # Symmetry (vertical/horizontal) using ROI
    roi = cv2.resize(roi, (64,64), interpolation=cv2.INTER_NEAREST)
    v_left = roi[:, :32].astype(np.float32)
    v_right = np.fliplr(roi[:, 32:]).astype(np.float32)
    symmetry_v = 1.0 - (np.abs(v_left - v_right)/255.0).mean()
    h_top = roi[:32, :].astype(np.float32)
    h_bottom = np.flipud(roi[32:, :]).astype(np.float32)
    symmetry_h = 1.0 - (np.abs(h_top - h_bottom)/255.0).mean()
    # Hu moments on thresholded image
    m = cv2.moments(roi)
    hu = cv2.HuMoments(m).flatten()
    hu = np.sign(hu) * np.log1p(np.abs(hu) + 1e-8)
    feats = {
        'aspect_ratio': float(aspect_ratio),
        'stroke_density': float(stroke_density),
        'edge_density': float(edge_density),
        'symmetry_v': float(symmetry_v),
        'symmetry_h': float(symmetry_h),
        'hu1': float(hu[0]),
        'hu2': float(hu[1]),
        'hu3': float(hu[2]),
    }
    return feats


def _discretize(value: float, bins: List[float], labels: List[str]) -> str:
    idx = np.digitize([value], bins)[0]
    idx = min(idx, len(labels)-1)
    return labels[idx]


def discretize_features(feats: Dict[str, float]) -> Dict[str, str]:
    # Fixed bins per feature (simple, can be tuned)
    labels3 = ['L','M','H']
    out = {}
    out['aspect_ratio'] = _discretize(feats['aspect_ratio'], [0.8, 1.2], labels3)
    out['stroke_density'] = _discretize(feats['stroke_density'], [0.05, 0.20], labels3)
    out['edge_density'] = _discretize(feats['edge_density'], [0.02, 0.10], labels3)
    out['symmetry_v'] = _discretize(feats['symmetry_v'], [0.6, 0.85], labels3)
    out['symmetry_h'] = _discretize(feats['symmetry_h'], [0.6, 0.85], labels3)
    out['hu1'] = _discretize(feats['hu1'], [-5.0, -1.0], labels3)
    out['hu2'] = _discretize(feats['hu2'], [-7.0, -2.0], labels3)
    out['hu3'] = _discretize(feats['hu3'], [-7.0, -2.0], labels3)
    return out


def train_bn(dataset_dir: str, out_path: str = 'models/alphabet_bn.pkl') -> str:
    """
    Train a naive Bayes BN: Letter -> features using labeled folders A..Z inside dataset_dir.
    Each subfolder should contain images of that letter.
    """
    rows = []
    letters = []
    dataset_dir = Path(dataset_dir)
    for ch in [chr(c) for c in range(ord('A'), ord('Z')+1)]:
        sub = dataset_dir / ch
        if not sub.exists():
            continue
        # Chain multiple globs since the | operator isn't supported on Windows Path.glob
        paths = list(sub.glob('*.png')) + list(sub.glob('*.jpg')) + list(sub.glob('*.jpeg'))
        for p in paths:
            try:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                feats = extract_features_alphabet(Image.fromarray(img))
                d = discretize_features(feats)
                rows.append(d)
                letters.append(ch)
            except Exception:
                continue
    if not rows:
        raise ValueError("No training images found. Ensure dataset_dir has A..Z subfolders with images.")
    df = pd.DataFrame(rows)
    df['Letter'] = letters
    # Build model: Letter -> each feature
    edges = [('Letter', f) for f in FEATURES]
    model = BayesianModel(edges)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    return str(out_path)


def load_bn(path: str = 'models/alphabet_bn.pkl') -> BayesianModel:
    with open(path, 'rb') as f:
        return pickle.load(f)


def infer_letter(model: BayesianModel, pil_image, topk: int = 3) -> Dict[str, float]:
    feats = extract_features_alphabet(pil_image)
    d = discretize_features(feats)
    infer = VariableElimination(model)
    q = infer.query(variables=['Letter'], evidence=d, show_progress=False)
    probs = q.values
    # Derive the state names for 'Letter' robustly across pgmpy versions
    states = None
    try:
        cpd = model.get_cpds('Letter')
        if hasattr(cpd, 'state_names') and 'Letter' in cpd.state_names:
            states = cpd.state_names['Letter']
        elif hasattr(cpd, 'state_names_map') and 'Letter' in cpd.state_names_map:
            states = cpd.state_names_map['Letter']
    except Exception:
        states = None
    if states is None:
        # Fallback: use all uppercase letters by default
        states = [chr(c) for c in range(ord('A'), ord('Z')+1)]
        states = states[: len(probs)]
    mapping = {states[i]: float(probs[i]) for i in range(len(states))}
    # sort and take topk
    top = dict(sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:topk])
    return top


def suggestions_for_letter():
    return [
        "Increase contrast and write with steady, continuous strokes.",
        "Keep the letter centered and at a consistent size.",
        "Aim for symmetry when applicable (e.g., 'A', 'M'); avoid excessive slant.",
        "Ensure distinct features for similar letters (e.g., 'O' vs 'Q', 'I' vs 'L').",
    ]


def train_bn_from_hf(dataset_id: str = 'pittawat/letter_recognition', split: str = 'train', out_path: str = 'models/alphabet_bn.pkl') -> str:
    """Train BN directly from a Hugging Face dataset without writing images to disk.

    Requires: datasets, pyarrow. The dataset must have columns 'image' (PIL) and 'label' (0..25).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets package not installed. Install with: pip install datasets pyarrow") from e

    ds = load_dataset(dataset_id, split=split)
    rows = []
    letters = []
    # Use label feature mapping if present
    label_feature = ds.features.get('label')
    int2str = (label_feature.int2str if hasattr(label_feature, 'int2str') else (lambda i: chr(ord('A') + int(i))))

    for ex in ds:
        img = ex['image']  # PIL.Image
        if not isinstance(img, Image.Image):
            # Some datasets may store bytes or arrays; try to convert
            try:
                img = Image.fromarray(np.array(img))
            except Exception:
                continue
        feats = extract_features_alphabet(img)
        d = discretize_features(feats)
        rows.append(d)
        lbl = ex.get('label', None)
        if lbl is None:
            continue
        ch = int2str(int(lbl))
        # Ensure uppercase single-letter A..Z
        ch = ch[0].upper()
        letters.append(ch)

    if not rows or not letters or len(rows) != len(letters):
        raise ValueError("No usable samples found in the HF dataset. Ensure it has 'image' and 'label'.")

    df = pd.DataFrame(rows)
    df['Letter'] = letters
    edges = [('Letter', f) for f in FEATURES]
    model = BayesianModel(edges)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    return str(out_path)


def export_hf_to_folder(dataset_id: str, out_dir: str = 'data/alphabets', split: str = 'train', limit: int | None = None) -> str:
    """Export a HF dataset to A..Z folder structure with JPG images for offline training.

    This is optional; training can be done directly in-memory via train_bn_from_hf.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets package not installed. Install with: pip install datasets pyarrow") from e

    ds = load_dataset(dataset_id, split=split)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    label_feature = ds.features.get('label')
    int2str = (label_feature.int2str if hasattr(label_feature, 'int2str') else (lambda i: chr(ord('A') + int(i))))
    counts = {}
    saved = 0
    for ex in ds:
        img = ex['image']
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(np.array(img))
            except Exception:
                continue
        lbl = ex.get('label', None)
        if lbl is None:
            continue
        ch = int2str(int(lbl))[0].upper()
        sub = out_root / ch
        sub.mkdir(parents=True, exist_ok=True)
        counts[ch] = counts.get(ch, 0) + 1
        img.save(sub / f"{ch}_{counts[ch]:05d}.jpg")
        saved += 1
        if limit is not None and saved >= limit:
            break
    return str(out_root)
