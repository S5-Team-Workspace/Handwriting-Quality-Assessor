# Handwriting Quality Assessor

Digit quality via VAE (MNIST) and line quality via geometric features with optional Bayesian scoring.

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
```

2. Ensure the trained VAE weights exist at `models/vae_mnist.pth` (already present in your repo).

3. Run the app:

```bash
streamlit run app.py
```

## Components

- `vae_scorer.py`: Loads the MNIST VAE and computes a 0-100 quality score using reconstruction error with min-max calibration.
- `line_scorer.py`: Detects lines using OpenCV, extracts angle and straightness, and computes a 0-100 score (rule-based, BN-ready).
- `app.py`: Streamlit GUI to upload an image and select analysis type.
- `requirements.txt`: Project dependencies.

## Notes

- Digit scoring uses MSE recon error normalized between a low/high clip (tune with MNIST test percentiles if desired).
- Line scoring uses a weighted combination of angle deviation and straightness. Replace with a BN later if needed.
- The app caches the VAE to avoid repeated loads.

## Troubleshooting

- If `torch`/`opencv` imports fail, ensure the virtual environment is active and dependencies installed.
- If `models/vae_mnist.pth` is missing, copy your trained file into the `models/` folder.

## Exporting the 14x14 MNIST VAE from the notebook

The notebook `VAE_Digit_Recognition.ipynb` contains a compact VAE for 14x14 MNIST (196-dim input) that performed well. You can export its trained weights and use them in the app:

1. Open the notebook and run the training cells until the model has converged.
2. Run the added save cell (just after the `VAE` class cell). It writes `models/vae_14x14_best.pth`.
3. Load it in code with the helper module:

```python
from models.vae_14x14_mnist import load_state_dict
vae14 = load_state_dict("models/vae_14x14_best.pth", device="cpu")
vae14.eval()
```

The app’s digit scoring pipeline will automatically detect this model type and switch to 14x14 preprocessing and BCE-based reconstruction error if you integrate it in the selection. If you want a selector entry in the sidebar, let me know and I’ll wire it to the new path.
