from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from vae_scorer import load_vae, score_digit
from line_scorer import score_line
from vae_handwriting_model import HandwritingQualityAssessor
from bayesian_handwriting_model import BayesianHandwritingAssessor
from game_engine import load_profile, save_profile, current_tasks, evaluate_task, on_task_completed
from preprocess import preprocess_digit_image

st.set_page_config(page_title="Handwriting Quality Assessor", layout="centered")

st.title("Handwriting Quality Assessor")
st.caption("Digit quality via VAE • Line quality via feature scoring • Handwriting quality via VAE or Bayesian • Optional LLM tips")

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

# --- Optional LLM helpers (llama.cpp GGUF or transformers) ---
@cache_resource
def _load_llama_cpp(model_path: str):
    try:
        from llama_cpp import Llama  # type: ignore
        return Llama(model_path=model_path, n_ctx=2048)
    except Exception:
        return None

@cache_resource
def _load_hf_pipeline(model_id: str):
    try:
        from transformers import pipeline  # type: ignore
        return pipeline("text-generation", model=model_id)
    except Exception:
        return None

def _llm_tips(prompt: str, backend: str, model_ref: str) -> str | None:
    if not model_ref:
        return None
    try:
        if backend == "llama.cpp":
            llm = _load_llama_cpp(model_ref)
            if llm is None:
                return None
            out = llm.create_completion(prompt=prompt, max_tokens=256, temperature=0.7)
            text = out.get("choices", [{}])[0].get("text", "").strip()
            return text or None
        else:
            gen = _load_hf_pipeline(model_ref)
            if gen is None:
                return None
            out = gen(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            text = out[0]["generated_text"] if isinstance(out, list) else str(out)
            # Heuristic: remove prompt echo if present
            return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
    except Exception:
        return None

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Analyze", "Play"], index=0)
    options = [
        "Digit (MNIST)",
        "Line",
        "Quality (VAE)",
        "Quality (Bayesian)",
    ]
    analysis_type = st.selectbox("Select Analysis Type", options, index=0)
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
    elif analysis_type == "Quality (VAE)":
        st.markdown("VAE Handwriting Quality")
        hq_vae_model_path = st.text_input("VAE Quality Model Path", value="models/handwriting_vae.pth")
    elif analysis_type == "Quality (Bayesian)":
        st.markdown("Bayesian Handwriting Quality")
        bayes_model_path = st.text_input("Bayesian Model Path", value="models/bayesian_handwriting_model.pkl")
    else:
        line_type = None

    st.divider()
    st.subheader("LLM tips (optional)")
    use_llm = st.checkbox("Enable LLM-generated tips", value=False)
    llm_backend = st.radio("LLM backend", ["llama.cpp", "transformers"], index=0, horizontal=True, disabled=not use_llm)
    llm_model_ref = st.text_input(
        "Model path (GGUF for llama.cpp) or HF model id (for transformers)",
        value="", disabled=not use_llm,
        help="Example (llama.cpp): C:/models/llama-3.2-1B-instruct.Q4_K_M.gguf | Example (transformers): sshleifer/tiny-gpt2"
    )

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
                    names = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
                    name = names[pred_digit] if 0 <= pred_digit < 10 else str(pred_digit)
                    st.success(f"Digit: {name} ({pred_digit}) • Confidence: {conf:.2f}" if conf is not None else f"Digit: {name} ({pred_digit})")

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
                # Tips (heuristics + optional LLM)
                tips = []
                tips += out.get("suggestions") or []
                if use_llm:
                    prompt = (
                        "You are a handwriting coach. Based on this analysis, give 3-5 concrete tips to improve legibility and neatness.\n"
                        f"Predicted digit: {pred_digit}, confidence: {out.get('prediction_confidence')}\n"
                        f"Quality score: {out.get('quality_score')}, recon_error_mse: {out.get('recon_error_mse'):.4f}.\n"
                        "Keep each tip short and actionable."
                    )
                    llm_text = _llm_tips(prompt, llm_backend, llm_model_ref)
                    if llm_text:
                        tips.append(llm_text)
                if tips:
                    with st.expander("Tips to Improve Your Digit"):
                        for s in tips:
                            st.write("- "+s)
        elif analysis_type == "Line":
            out = score_line(img, line_type=line_type)
            score = out["quality_score"]
            overlay = out["overlay"]

            col2.subheader("Line Detection")
            show_image(col2, overlay, caption=f"Quality Score: {score}", clamp=True)

            # Predict line type and confidence from metrics
            angle = out.get("angle_deg", 0.0)
            straight = out.get("straightness", 1.0)
            # compute both candidates
            def _score_for(target_deg: float) -> float:
                dev = abs(((angle - target_deg + 90) % 180) - 90)
                s_angle = max(0.0, 1.0 - dev / 12.0)
                s_straight = max(0.0, 1.0 - straight / 0.03)
                return 0.6 * s_angle + 0.4 * s_straight
            s_sleep = _score_for(0.0)
            s_slant = _score_for(45.0)
            if s_sleep >= s_slant:
                pred_line = "Sleeping line"
                conf = s_sleep / max(1e-6, (s_sleep + s_slant))
            else:
                pred_line = "Slanting line"
                conf = s_slant / max(1e-6, (s_sleep + s_slant))
            st.success(f"Predicted Line Type: {pred_line} • Confidence: {conf:.2f}")

            with st.expander("Details"):
                d = {k: v for k, v in out.items() if k not in ("overlay",)}
                d["predicted_line_type"] = pred_line
                d["prediction_confidence"] = conf
                st.json(d)
            if mode == "Play":
                tasks = current_tasks(profile.level)
                target = next((t for t in tasks if t["type"]=="line" and t.get("line_type")==line_type), {"type":"line","line_type":line_type,"threshold":55,"xp":15})
                passed, gained_xp, details = evaluate_task(target, out)
                profile, msg = on_task_completed(profile, target, passed, gained_xp, details)
                st.info(msg)
            # Tips (heuristics + optional LLM)
            tips = []
            if out.get("angle_dev", 0) > 8:
                tips.append("Align your line closer to the target angle; lightly mark start/end points to guide the stroke.")
            if out.get("straightness", 1) > 0.04:
                tips.append("Keep pressure steady and move your whole arm, not just the wrist, for a straighter stroke.")
            if out.get("length_px", 0) < 40:
                tips.append("Make the line longer and continuous; avoid sketching many short segments.")
            if use_llm:
                prompt = (
                    "You are a handwriting coach. Give 3 short tips to draw cleaner lines in notebooks.\n"
                    f"Measured angle: {angle:.1f}°, straightness: {straight:.3f}, predicted: {pred_line}, confidence: {conf:.2f}.\n"
                    "Focus on angle control, smoothness, and consistency."
                )
                llm_text = _llm_tips(prompt, llm_backend, llm_model_ref)
                if llm_text:
                    tips.append(llm_text)
            if tips:
                with st.expander("Tips to Improve Your Line"):
                    for s in tips:
                        st.write("- "+s)
        elif analysis_type == "Quality (VAE)":
            @cache_resource
            def _load_hq_vae(path: str):
                assessor = HandwritingQualityAssessor()
                try:
                    assessor.load_model(path)
                    assessor.model_trained = True
                    return assessor
                except Exception:
                    return None

            assessor = _load_hq_vae(hq_vae_model_path)
            if assessor is None:
                st.error(f"Could not load VAE handwriting quality model from {hq_vae_model_path}")
            else:
                res = assessor.assess_handwriting_quality(img)
                if "error" in res:
                    st.error(res["error"])
                else:
                    q = res["overall_quality_score"]
                    cat = res["quality_category"]
                    recon = res["raw_metrics"].get("reconstructed")
                    col2.subheader("VAE Handwriting Reconstruction")
                    if recon is not None:
                        show_image(col2, recon, caption=f"Quality: {q:.1f} ({cat})", clamp=True)
                    st.metric("Overall Quality (0-100)", f"{q:.1f}")
                    with st.expander("Details"):
                        # Avoid dumping big arrays directly
                        raw = dict(res)
                        raw["raw_metrics"] = {k: v for k, v in res["raw_metrics"].items() if k != "reconstructed"}
                        st.json(raw)
                    # Tips (heuristics + optional LLM)
                    tips = []
                    if q < 60:
                        tips.append("Use slower, deliberate strokes; keep spacing consistent and avoid over-pressing.")
                    if use_llm:
                        prompt = (
                            "You are a handwriting coach. Give 3-5 tips to improve handwriting quality and readability.\n"
                            f"Quality category: {cat}, score: {q:.1f}.\n"
                            "Be actionable and concise."
                        )
                        llm_text = _llm_tips(prompt, llm_backend, llm_model_ref)
                        if llm_text:
                            tips.append(llm_text)
                    if tips:
                        with st.expander("Tips to Improve Your Handwriting"):
                            for s in tips:
                                st.write("- "+s)
                if mode == "Play":
                    st.info("Play mode scoring isn't wired for generic handwriting quality yet. Use Analyze mode.")
        elif analysis_type == "Quality (Bayesian)":
            @cache_resource
            def _load_bayes(path: str):
                assessor = BayesianHandwritingAssessor()
                try:
                    assessor.load_model(path)
                    return assessor
                except Exception:
                    return None

            assessor = _load_bayes(bayes_model_path)
            if assessor is None or not getattr(assessor, "is_trained", False):
                st.error(f"Could not load Bayesian handwriting model from {bayes_model_path}")
            else:
                res = assessor.assess_quality_bayesian(np.array(img.convert('L')))
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.subheader("Bayesian Handwriting Quality")
                    st.metric("Overall Quality (0-100)", f"{res['overall_quality_score']:.1f}")
                    st.write(f"Predicted: {res['predicted_quality'].title()} (conf: {res['confidence']:.2f})")
                    with st.expander("Class probabilities"):
                        st.json(res.get("quality_probabilities", {}))
                    with st.expander("Extracted features"):
                        st.json(res.get("extracted_features", {}))
                    # Tips
                    tips = []
                    if res.get('overall_quality_score', 0) < 60:
                        tips.append("Practice consistent letter size and spacing; reduce tremor by slowing down.")
                    if use_llm:
                        prompt = (
                            "You are a handwriting coach. Provide 3-5 tips to improve handwriting quality.\n"
                            f"Predicted: {res.get('predicted_quality')}, confidence: {res.get('confidence'):.2f}, score: {res.get('overall_quality_score'):.1f}.\n"
                            "Keep tips brief and actionable."
                        )
                        llm_text = _llm_tips(prompt, llm_backend, llm_model_ref)
                        if llm_text:
                            tips.append(llm_text)
                    if tips:
                        with st.expander("Tips to Improve Your Handwriting"):
                            for s in tips:
                                st.write("- "+s)
                if mode == "Play":
                    st.info("Play mode scoring isn't wired for generic handwriting quality yet. Use Analyze mode.")
else:
    st.info("Upload an image to begin. For digits, use MNIST-like 28x28 grayscale; for lines, any simple drawing works.")