import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from vae_scorer import load_vae, score_digit
from line_scorer import score_line
# Lightweight capability probe for BN dependencies; lazily import bn_alphabet later.
_bn_import_error = None
try:
    import pgmpy  # type: ignore
    import cv2  # type: ignore
    BN_AVAILABLE = True
except Exception as _e:
    BN_AVAILABLE = False
    _bn_import_error = str(_e)
    load_bn = infer_letter = suggestions_for_letter = None  # type: ignore
from game_engine import load_profile, save_profile, current_tasks, evaluate_task, on_task_completed
from preprocess import preprocess_digit_image

st.set_page_config(page_title="Handwriting Quality Assessor", layout="centered")

st.title("Handwriting Quality Assessor")
st.caption("Digit quality via VAE • Line quality via feature scoring (BN-ready)")

# --- Compatibility helper for Streamlit image API ---
def show_image(container, image, caption=None, clamp=False):
    """Display image in a way that works across Streamlit versions.

    Prefer new `width` API; fall back to legacy args when not available.
    """
    # New API (Streamlit >= 1.39): width='stretch'|'content'
    try:
        container.image(image, caption=caption, clamp=clamp, width='stretch')
        return
    except TypeError:
        pass
    # Legacy API fallback
    try:
        container.image(image, caption=caption, clamp=clamp, use_container_width=True)
    except TypeError:
        try:
            container.image(image, caption=caption, clamp=clamp, use_column_width=True)
        except TypeError:
            container.image(image, caption=caption)

# --- Compatibility alias for cache decorator ---
try:
    cache_resource = st.cache_resource
except AttributeError:  # older Streamlit
    cache_resource = st.cache

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Analyze", "Play"], index=0)
    options = ["Digit (MNIST)", "Line"] + (["Alphabet (BN)"] if BN_AVAILABLE else [])
    analysis_type = st.selectbox("Select Analysis Type", options, index=0)
    if not BN_AVAILABLE:
        st.caption("To enable Alphabet (BN), install pgmpy. See instructions below.")
        if _bn_import_error:
            st.caption(f"BN import error: {_bn_import_error}")
    # Profile for Play mode
    if mode == "Play":
        player_name = st.text_input("Player name", value="Player1")
        profile = load_profile(player_name)
        st.write(f"Level {profile.level} • XP {profile.xp}/{100*profile.level}")
        if profile.badges:
            st.caption("Badges: " + ", ".join(profile.badges))
    if analysis_type == "Digit (MNIST)":
        model_choice = st.selectbox(
            "VAE Model",
            [
                "Conv VAE (models/vae_mnist.pth)",
                "MLP VAE (VariationalAutoencoders/vae_logs_latent2_beta1.0/best_vae_model.pth)",
                "VAE14x14 (models/vae_14x14_best.pth)",
            ],
            index=0,
        )
        model_path = (
            "models/vae_mnist.pth"
            if model_choice.startswith("Conv")
            else ("VariationalAutoencoders/vae_logs_latent2_beta1.0/best_vae_model.pth" if model_choice.startswith("MLP") else "models/vae_14x14_best.pth")
        )
        st.markdown("Preprocessing")
        pp_enable = st.checkbox("Enable preprocessing", value=True)
        if pp_enable:
            pp_remove_lines = st.checkbox("Remove notebook lines", value=True)
            pp_norm_stroke = st.checkbox("Normalize stroke thickness", value=True)
            pp_pixelate = st.checkbox("Pixelate when resizing", value=False)
        else:
            pp_remove_lines = True
            pp_norm_stroke = True
            pp_pixelate = False
    if analysis_type == "Line":
        line_type = st.selectbox("Line Type", ["Sleeping (0°)", "Slanting (45°)"])
        line_type = "sleeping" if line_type.startswith("Sleeping") else "slanting"
    elif analysis_type == "Alphabet (BN)":
        st.markdown("Load a pre-trained BN or train from a dataset:")
        bn_choice = st.selectbox(
            "BN Model",
            [
                "Load (models/alphabet_bn.pkl)",
                "Train from dataset...",
                "Train from Hugging Face...",
            ],
        )
        ds_path = None
        hf_id = None
        hf_split = None
        if bn_choice == "Train from dataset...":
            ds_path = st.text_input("Dataset folder (A..Z subfolders)", value="data/alphabets")
        elif bn_choice == "Train from Hugging Face...":
            hf_id = st.text_input("HF dataset id", value="pittawat/letter_recognition")
            hf_split = st.selectbox("Split", ["train", "test"], index=0)
    else:
        line_type = None

uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"]) 

col1, col2 = st.columns(2)

if uploaded is not None:
    img = Image.open(uploaded)
    col1.subheader("Input Image")
    show_image(col1, img)

    if st.button("Analyze" if mode == "Analyze" else "Submit Task"):
        if analysis_type == "Digit (MNIST)":
            # Try to load VAE only once and cache it
            @cache_resource
            def _load_cached_vae(path: str):
                try:
                    return load_vae(model_path=path)
                except FileNotFoundError:
                    return None

            vae = _load_cached_vae(model_path)
            if vae is None:
                st.error(f"VAE model not found. Ensure the file exists: {model_path}")
            else:
                if pp_enable:
                    pre = preprocess_digit_image(
                        img,
                        target_size=(28,28) if 'Conv' in model_choice or 'MLP' in model_choice else (14,14),
                        remove_lines=pp_remove_lines,
                        normalize_stroke_thickness=pp_norm_stroke,
                        pixelate=pp_pixelate,
                        return_debug=True,
                    )
                    pre_img = Image.fromarray(pre['preview_uint8'])
                    with st.expander("Preprocessing preview"):
                        show_image(st, pre['preview_uint8'], caption="Normalized input")
                    # Use preprocessed image for scoring
                    out = score_digit(pre_img, vae=vae)
                else:
                    # Use raw image directly
                    out = score_digit(img, vae=vae)
                score = out["quality_score"]
                recon = out["reconstruction"]
                pred_digit = out.get("predicted_digit")

                col2.subheader("VAE Reconstruction")
                show_image(col2, recon, caption=f"Quality Score: {score}", clamp=True)
                if pred_digit is not None:
                    conf = out.get("prediction_confidence")
                    st.success(f"Predicted Digit: {pred_digit}" + (f" (conf: {conf:.2f})" if conf is not None else ""))

                with st.expander("Details"):
                    st.json({k: v for k, v in out.items() if k not in ("reconstruction",)})
                if mode == "Play":
                    # Find a matching digit task, else general digit task
                    tasks = current_tasks(profile.level)
                    target = next((t for t in tasks if t["type"]=="digit" and (t.get("digit")==pred_digit or t.get("digit") is not None)), None)
                    if target is None:
                        target = {"type":"digit","digit":pred_digit,"threshold":60,"xp":15,"title":"Write a digit"}
                    passed, gained_xp, details = evaluate_task(target, out)
                    profile, msg = on_task_completed(profile, target, passed, gained_xp, details)
                    st.info(msg)
                    # Show improvement suggestions
                    sugg = out.get("suggestions")
                    if sugg:
                        with st.expander("Tips to Improve Your Digit"):
                            for s in sugg:
                                st.write("- "+s)
        elif analysis_type == "Line":
            out = score_line(img, line_type=line_type)
            score = out["quality_score"]
            overlay = out["overlay"]

            col2.subheader("Line Detection")
            show_image(col2, overlay, caption=f"Quality Score: {score}", clamp=True)

            with st.expander("Details"):
                st.json({k: v for k, v in out.items() if k not in ("overlay",)})
            if mode == "Play":
                tasks = current_tasks(profile.level)
                target = next((t for t in tasks if t["type"]=="line" and t.get("line_type")==line_type), {"type":"line","line_type":line_type,"threshold":55,"xp":15})
                passed, gained_xp, details = evaluate_task(target, out)
                profile, msg = on_task_completed(profile, target, passed, gained_xp, details)
                st.info(msg)
        else:
            # Alphabet (BN)
            try:
                # Lazy import to avoid blocking the whole app if bn_alphabet has issues
                from bn_alphabet import (
                    load_bn as _load_bn,
                    infer_letter as _infer_letter,
                    suggestions_for_letter as _sugg,
                    train_bn as _train_bn,
                    train_bn_from_hf as _train_bn_from_hf,
                )
            except Exception as e:
                st.error(f"Alphabet BN modules failed to import: {e}")
            else:
                try:
                    if bn_choice == "Train from dataset..." and ds_path:
                        with st.spinner("Training BN on folder dataset..."):
                            model_path = _train_bn(ds_path)
                        st.success(f"Trained and saved BN to {model_path}")
                        model = _load_bn(model_path)
                    elif bn_choice == "Train from Hugging Face..." and hf_id:
                        with st.spinner("Training BN from Hugging Face dataset..."):
                            model_path = _train_bn_from_hf(hf_id, split=(hf_split or 'train'))
                        st.success(f"Trained and saved BN to {model_path}")
                        model = _load_bn(model_path)
                    else:
                        model = _load_bn()
                    top = _infer_letter(model, img, topk=3)
                    best = next(iter(top.items())) if top else (None, None)
                    st.subheader("Alphabet Prediction (BN)")
                    if best[0] is not None:
                        st.success(f"Predicted Letter: {best[0]} (confidence: {best[1]:.2f})")
                    st.json(top)
                    with st.expander("Suggestions"):
                        for s in _sugg():
                            st.write("- "+s)
                    if mode == "Play":
                        tasks = current_tasks(profile.level)
                        # Choose a task for the predicted letter or default to 'A'
                        target_letter = best[0] if best[0] is not None else 'A'
                        target = next((t for t in tasks if t["type"]=="alphabet" and t.get("letter")==target_letter), {"type":"alphabet","letter":target_letter,"threshold":0.6,"xp":25})
                        # Prepare result payload expected by evaluator
                        out_bn = {"top": top}
                        passed, gained_xp, details = evaluate_task(target, out_bn)
                        profile, msg = on_task_completed(profile, target, passed, gained_xp, details)
                        st.info(msg)
                except Exception as e:
                    st.error(f"BN inference failed: {e}")
else:
    st.info("Upload an image to begin. For digits, use MNIST-like 28x28 grayscale; for lines, any simple drawing works.")
