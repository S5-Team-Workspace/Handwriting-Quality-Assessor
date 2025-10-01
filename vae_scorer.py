import io
import os
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from models.vae_14x14_mnist import VAE14x14
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None

# Lightweight VAE definition matching your saved weights
class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        # Match saved state_dict naming: "fc_log_var"
        self.fc_log_var = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var, z


class MLPVAE(nn.Module):
    """MLP-based VAE matching external checkpoint structure.
    Expected keys: encoder.<layers>, fc_mu, fc_logvar, decoder.<layers>
    Architecture inferred from state_dict shapes (common: 784->256->128->(mu,logvar=latent)
    and  (latent)->128->256->784 with BatchNorm and activations).
    """
    def __init__(self, input_dim: int = 784, hidden_dims=(256, 128), latent_dim: int = 2):
        super().__init__()
        h1, h2 = hidden_dims
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Identity(),  # align indices so next Linear is at .4
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Identity(),  # align indices so next Linear is at .4
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Identity(),  # align indices so final Linear is at .8
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (B,1,28,28) or (B,784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        recon = recon.view(-1, 1, 28, 28)
        return recon, mu, logvar, z


def _to_tensor_28x28(img: Image.Image) -> torch.Tensor:
    # Convert to grayscale, pad/crop to square, resize to 28x28, normalize to [0,1]
    if img.mode != "L":
        img = ImageOps.grayscale(img)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    img = canvas.resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, 1))  # (1,1,28,28)
    return torch.from_numpy(arr)


def _to_tensor_14x14(img: Image.Image) -> torch.Tensor:
    # Resize to 14x14 grayscale, binarize at 0.5 as in notebook preprocessing
    if img.mode != "L":
        img = ImageOps.grayscale(img)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    img = canvas.resize((14, 14), Image.BILINEAR)
    arr = (np.asarray(img).astype(np.float32) / 255.0)
    arr = (arr > 0.5).astype(np.float32)
    arr = arr.reshape(1, -1)  # (1,196)
    return torch.from_numpy(arr)


def load_vae(model_path: str = "models/vae_mnist.pth",
             device: str | torch.device = None,
             latent_dim: int = 20) -> nn.Module:
    """Load a VAE checkpoint. Auto-detect Conv vs MLP architecture by state_dict keys."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(model_path, map_location=device)
    # Unwrap common wrappers
    if isinstance(raw, dict) and any(k in raw for k in ("state_dict", "model_state_dict")):
        state = raw.get("state_dict", raw.get("model_state_dict"))
    else:
        state = raw

    keys = list(state.keys())
    # Heuristic detection
    # - 14x14 VAE: has fc1/fc3 with input_dim=196
    is_14 = ("fc1.weight" in state and hasattr(state["fc1.weight"], 'shape') and state["fc1.weight"].shape[1] == 196)
    # - MLP VAE: has top-level 'fc_mu', 'fc_logvar' and 'encoder.0.weight' shapes (256,784)
    is_mlp = ("fc_mu.weight" in state and "fc_logvar.weight" in state) or any(k.startswith("encoder.0.weight") for k in keys)
    if is_14:
        vae = VAE14x14().to(device)
        vae.load_state_dict(state)
    elif is_mlp:
        # Infer latent_dim from shapes
        ld = state["fc_mu.weight"].shape[0] if "fc_mu.weight" in state else 2
        # Infer hidden sizes from shapes if available
        h2 = state.get("decoder.0.weight", torch.empty(ld, 2)).shape[0] if "decoder.0.weight" in state else 128
        h1 = state.get("decoder.4.weight", torch.empty(256, h2)).shape[0] if "decoder.4.weight" in state else 256
        vae = MLPVAE(latent_dim=ld, hidden_dims=(h1, h2)).to(device)
        vae.load_state_dict(state, strict=False)
    else:
        vae = VAE(latent_dim=latent_dim).to(device)
        vae.load_state_dict(state)
    vae.eval()
    return vae


def score_digit(image: Image.Image,
                vae: nn.Module | None = None,
                device: str | torch.device = None,
                recon_err_clip: Tuple[float, float] = (0.02, 0.20)) -> Dict[str, Any]:
    """
    Compute digit quality using VAE reconstruction error.
    - recon_err_clip: (low, high) for min-max normalization to 0..100
      Tune once using MNIST test percentiles; defaults are robust starting points.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if vae is None:
        vae = load_vae(device=device)

    # Choose preprocessing based on model type
    if isinstance(vae, VAE14x14):
        x = _to_tensor_14x14(image).to(device)
    else:
        x = _to_tensor_28x28(image).to(device)
    with torch.no_grad():
        out = vae(x)
        # Support both 28x28 VAEs and 14x14 notebook VAE
        if isinstance(vae, VAE14x14):
            recon, mu, std = out
            # Convert std to logvar-compatible for reporting
            log_var = torch.log(std**2 + 1e-8)
            z = None
            err = F.binary_cross_entropy(recon, x, reduction="mean").item()
        else:
            recon, mu, log_var, z = out
            err = F.mse_loss(recon, x, reduction="mean").item()

    lo, hi = recon_err_clip
    norm = (err - lo) / max(1e-8, (hi - lo))
    norm = float(np.clip(norm, 0.0, 1.0))
    score = int(round(100 * (1.0 - norm)))

    # Prepare visualization-friendly outputs
    if isinstance(vae, VAE14x14):
        # 14x14 vector -> (14,14) for display
        recon_2d = recon.view(14, 14).cpu().numpy()
        recon_img = (recon_2d * 255).astype(np.uint8)
    else:
        recon_img = (recon.squeeze().cpu().numpy() * 255).astype(np.uint8)
    result = {
        "quality_score": score,
        "recon_error_mse": err,
        "latent": (None if z is None else z.squeeze().cpu().numpy().tolist()),
        "mu": mu.squeeze().cpu().numpy().tolist(),
        "logvar": log_var.squeeze().cpu().numpy().tolist(),
        "reconstruction": recon_img,
    }

    # Optional: predict digit label
    try:
        pred, conf = _predict_digit_label(vae, x, mu if z is None else z)
        if pred is not None:
            result["predicted_digit"] = int(pred)
            result["prediction_confidence"] = float(conf)
    except Exception:
        pass
    # Suggestions to improve handwriting (simple heuristics)
    try:
        result["suggestions"] = _digit_improvement_suggestions(image)
    except Exception:
        pass
    return result


def _predict_digit_label(vae: nn.Module, x_tensor: torch.Tensor, latent_tensor: torch.Tensor | None):
    """Predict digit label.
    - If latent is available (Conv/MLP VAE), use a cached logistic regression on latent z.
    - If using 14x14 VAE (no z), fall back to simple nearest centroid on input.
    Models are trained on-the-fly and cached to disk in .cache/.
    """
    cache_dir = Path('.cache')
    cache_dir.mkdir(exist_ok=True)

    if latent_tensor is not None and LogisticRegression is not None:
        model_path = cache_dir / 'mnist_latent_lr.joblib'
        try:
            import joblib
            if model_path.exists():
                clf = joblib.load(model_path)
            else:
                clf = _train_latent_classifier(vae)
                joblib.dump(clf, model_path)
            z_np = latent_tensor.detach().cpu().numpy().reshape(1, -1)
            probs = getattr(clf, 'predict_proba', None)
            if probs is not None:
                p = probs(z_np)[0]
                return int(np.argmax(p)), float(np.max(p))
            pred = clf.predict(z_np)[0]
            return int(pred), 0.0
        except Exception:
            return None, None
    else:
        # 14x14 fallback: nearest centroid in input space (very rough)
        try:
            from torchvision import datasets, transforms
            import torch.utils.data as data
            ds = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
            # compute simple centroids for 10 classes on first 200 samples
            counts = [0]*10
            sums = [torch.zeros(196) for _ in range(10)]
            for img, label in data.DataLoader(ds, batch_size=64, shuffle=False):
                img = img.view(img.size(0), -1)
                # downscale 28x28 -> 14x14 by average pooling
                img14 = torch.nn.functional.avg_pool2d(img.view(-1,1,28,28), kernel_size=2).view(-1,196)
                for i in range(img14.size(0)):
                    y = int(label[i].item())
                    if counts[y] < 100:  # limit per class for speed
                        sums[y] += img14[i]
                        counts[y] += 1
                if min(counts) >= 100:
                    break
            cents = [sums[c]/max(1,counts[c]) for c in range(10)]
            v = x_tensor.view(1, -1).cpu()
            dists = np.array([torch.norm(v - c.view(1, -1)).item() for c in cents])
            idx = int(np.argmin(dists))
            # pseudo-confidence: inverse-distance normalized
            inv = 1.0 / (dists + 1e-6)
            conf = float(inv[idx] / np.sum(inv))
            return idx, conf
        except Exception:
            return None, None


def _train_latent_classifier(vae: nn.Module):
    from torchvision import datasets, transforms
    import torch.utils.data as data
    ds = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    loader = data.DataLoader(ds, batch_size=256, shuffle=True)
    zs = []
    ys = []
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if isinstance(vae, VAE14x14):
                # downscale to 14x14 and binarize
                imgs14 = torch.nn.functional.avg_pool2d(imgs, kernel_size=2)
                x = imgs14.view(imgs14.size(0), -1)
                mu, std = vae.encoder(x)
                z = mu  # use mu as representation
            else:
                x = imgs
                _, mu, _, z = vae(x)
            zs.append((z if z is not None else mu).detach().cpu().numpy())
            ys.append(labels.numpy())
            if len(ys) * 256 >= 8000:  # limit for speed
                break
    Z = np.concatenate(zs, axis=0)
    Y = np.concatenate(ys, axis=0)
    if Pipeline is not None and StandardScaler is not None:
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=200, multi_class='multinomial')),
        ])
    else:
        # minimal fallback
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, multi_class='multinomial')
    clf.fit(Z, Y)
    return clf


def _digit_improvement_suggestions(pil_image: Image.Image):
    # Simple heuristics based on grayscale image
    img = pil_image.convert('L')
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Threshold to separate foreground/background
    th = 0.5
    fg = (arr > th).astype(np.uint8)
    # Occupancy ratio
    occ = fg.mean()
    suggestions = []
    if occ < 0.05:
        suggestions.append("Make the digit larger or write with thicker strokes so it occupies more of the image.")
    if occ > 0.5:
        suggestions.append("Thin the strokes or reduce size slightly to avoid filling most of the canvas.")
    # Center of mass
    ys, xs = np.where(fg > 0)
    if len(xs) > 0:
        cx, cy = xs.mean() / arr.shape[1], ys.mean() / arr.shape[0]
        if abs(cx - 0.5) > 0.15 or abs(cy - 0.5) > 0.15:
            suggestions.append("Center the digit within the image for better reconstruction and recognition.")
    else:
        suggestions.append("Increase contrast: the digit was not detected well. Use darker ink or higher contrast.")
    # Stroke smoothness proxy: count small holes/noisy pixels
    noise = ((arr > 0.4) & (arr < 0.6)).mean()
    if noise > 0.1:
        suggestions.append("Use smoother strokes and avoid noise; write clearly with steady lines.")
    if not suggestions:
        suggestions.append("Looks good! Keep the digit centered, sufficiently large, and with clear contrast.")
    return suggestions


if __name__ == "__main__":
    # Quick smoke test (expects models/vae_mnist.pth and a sample image path)
    from pathlib import Path
    sample = None
    for p in ["sample_digit.png", "data/sample_digit.png"]:
        if Path(p).exists():
            sample = p
            break
    if sample is None:
        print("No sample image found. Create sample_digit.png to test.")
    else:
        img = Image.open(sample)
        vae = load_vae()
        out = score_digit(img, vae)
        print({k: (type(v).__name__ if k=="reconstruction" else v) for k, v in out.items()})
