"""
app.py Streamlit GUI for the Fashion Denoising Autoencoder
──────────────────────────────────────────────────────────────
Run with:
    streamlit run app.py


"""

import io
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Denoising AE",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;500;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.03em; }

  .stApp { background: #0d0d0f; color: #e8e4dc; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #131318; border-right: 1px solid #2a2a35; }

  /* Cards */
  .card {
    background: #17171f;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }

  /* Metric chips */
  .metric-chip {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #3a3a55;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #a8e6cf;
    margin-right: 0.5rem;
  }

  /* Accent button */
  .stButton > button {
    background: #c8ff57 !important;
    color: #0d0d0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(200,255,87,0.25) !important;
  }

  /* Sliders */
  .stSlider > div > div > div { background: #c8ff57 !important; }

  /* Section titles */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.6rem;
  }

  /* Image captions */
  .img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    text-align: center;
    margin-top: 0.4rem;
  }

  /* Download button */
  .stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #c8ff57 !important;
    color: #c8ff57 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Model import (same directory) ─────────────────────────────────────────────
from autoencoder_model import DenoisingAutoencoder


# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_SIZE = 128
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading model …")
def load_model(checkpoint_path: str):
    model = DenoisingAutoencoder().to(DEVICE)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model, True
    model.eval()
    return model, False          # untrained weights – demo mode


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return t(img.convert("RGB")).unsqueeze(0)          # (1,3,H,W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_gaussian_noise(img_t: torch.Tensor, sigma: float) -> torch.Tensor:
    return (img_t + torch.randn_like(img_t) * sigma).clamp(0, 1)


def add_salt_pepper(img_t: torch.Tensor, amount: float) -> torch.Tensor:
    noisy = img_t.clone()
    mask  = torch.rand_like(img_t)
    noisy[mask < amount / 2]       = 0.0
    noisy[mask > 1 - amount / 2]   = 1.0
    return noisy


def add_speckle(img_t: torch.Tensor, sigma: float) -> torch.Tensor:
    return (img_t + img_t * torch.randn_like(img_t) * sigma).clamp(0, 1)


def psnr(clean: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((clean - recon) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified single-window SSIM (no external deps)."""
    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a  = ((a - mu_a) ** 2).mean()
    sigma_b  = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / den)


@st.cache_data(show_spinner=False)
def load_dataset_paths(csv_path: str, img_dir: str):
    try:
        df  = pd.read_csv(csv_path)
        paths = [os.path.join(img_dir, f) for f in df["filename"].tolist()]
        return [p for p in paths if os.path.exists(p)]
    except Exception:
        return []


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👗 Fashion DAE")
    st.markdown("<div class='section-title'>Model checkpoint</div>", unsafe_allow_html=True)
    ckpt_path = st.text_input("Path to .pth file", value="models/dae_fashion.pth")

    model, loaded = load_model(ckpt_path)
    if loaded:
        st.success("✔ Checkpoint loaded")
    else:
        st.warning("⚠ No checkpoint found — using random weights (demo mode)")

    st.divider()
    st.markdown("<div class='section-title'>Dataset (optional)</div>", unsafe_allow_html=True)
    csv_path = st.text_input("images.csv path",
        value="/kaggle/input/datasets/paramaggarwal/fashion-product-images-dataset/fashion-dataset/images.csv")
    img_dir  = st.text_input("images/ directory",
        value="/kaggle/input/datasets/paramaggarwal/fashion-product-images-dataset/fashion-dataset/images/")

    ds_paths = load_dataset_paths(csv_path, img_dir)
    if ds_paths:
        st.success(f"✔ {len(ds_paths):,} images found")
    else:
        st.info("Dataset not accessible — upload an image below.")

    st.divider()
    st.markdown("<div class='section-title'>Noise settings</div>", unsafe_allow_html=True)
    noise_type = st.selectbox("Noise type", ["Gaussian", "Salt & Pepper", "Speckle"])
    if noise_type == "Gaussian":
        noise_level = st.slider("σ (std)", 0.05, 0.8, 0.35, 0.05)
    elif noise_type == "Salt & Pepper":
        noise_level = st.slider("Amount", 0.01, 0.5, 0.15, 0.01)
    else:
        noise_level = st.slider("σ (speckle)", 0.05, 0.8, 0.30, 0.05)

    st.divider()
    st.markdown(f"<div class='section-title'>Device: {DEVICE}</div>", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# Fashion Denoising Autoencoder")
st.markdown("Upload an image or sample from the dataset · adjust noise · see the model reconstruct.")

tab_demo, tab_train, tab_about = st.tabs(["🖼  Denoise", "📈  Train", "ℹ️  About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DENOISE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    col_src, col_img = st.columns([1, 3])

    with col_src:
        st.markdown("<div class='section-title'>Image source</div>", unsafe_allow_html=True)
        source = st.radio("", ["Upload image", "Random from dataset"], label_visibility="collapsed")

        pil_clean = None

        if source == "Upload image":
            uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                pil_clean = Image.open(uploaded).convert("RGB")

        else:  # random dataset sample
            if st.button("🔀  Pick random image") and ds_paths:
                st.session_state["random_path"] = random.choice(ds_paths)
            rp = st.session_state.get("random_path")
            if rp and os.path.exists(rp):
                try:
                    pil_clean = Image.open(rp).convert("RGB")
                    st.caption(os.path.basename(rp))
                except Exception:
                    st.error("Could not open image.")
            elif not ds_paths:
                st.info("Dataset paths not found. Upload an image instead.")

        run_btn = st.button("⚡  Run Denoising", disabled=(pil_clean is None))

    # ── Results ──────────────────────────────────────────────────────────────
    with col_img:
        if pil_clean is not None:
            # Build tensor & noisy version
            clean_t = pil_to_tensor(pil_clean)

            if noise_type == "Gaussian":
                noisy_t = add_gaussian_noise(clean_t, noise_level)
            elif noise_type == "Salt & Pepper":
                noisy_t = add_salt_pepper(clean_t, noise_level)
            else:
                noisy_t = add_speckle(clean_t, noise_level)

            # Run inference
            with torch.no_grad():
                recon_t = model(noisy_t.to(DEVICE)).cpu()

            # Convert
            pil_noisy = tensor_to_pil(noisy_t)
            pil_recon = tensor_to_pil(recon_t)
            pil_orig  = pil_clean.resize((IMG_SIZE, IMG_SIZE))

            # Metrics
            arr_clean = np.array(pil_orig).astype(float) / 255.0
            arr_noisy = np.array(pil_noisy).astype(float) / 255.0
            arr_recon = np.array(pil_recon).astype(float) / 255.0

            psnr_noisy = psnr(arr_clean, arr_noisy)
            psnr_recon = psnr(arr_clean, arr_recon)
            ssim_noisy = ssim_simple(arr_clean, arr_noisy)
            ssim_recon = ssim_simple(arr_clean, arr_recon)

            # Display
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(pil_orig,  use_container_width=True)
                st.markdown("<div class='img-label'>Original</div>", unsafe_allow_html=True)
            with c2:
                st.image(pil_noisy, use_container_width=True)
                st.markdown(f"<div class='img-label'>Noisy  |  PSNR {psnr_noisy:.1f} dB</div>", unsafe_allow_html=True)
            with c3:
                st.image(pil_recon, use_container_width=True)
                st.markdown(f"<div class='img-label'>Denoised  |  PSNR {psnr_recon:.1f} dB</div>", unsafe_allow_html=True)

            st.divider()

            # Metrics row
            st.markdown(
                f"<span class='metric-chip'>PSNR noisy  {psnr_noisy:.2f} dB</span>"
                f"<span class='metric-chip'>PSNR denoised  {psnr_recon:.2f} dB</span>"
                f"<span class='metric-chip'>SSIM noisy  {ssim_noisy:.3f}</span>"
                f"<span class='metric-chip'>SSIM denoised  {ssim_recon:.3f}</span>",
                unsafe_allow_html=True,
            )

            # Download
            st.download_button(
                label="⬇  Download denoised image",
                data=img_to_bytes(pil_recon),
                file_name="denoised_fashion.png",
                mime="image/png",
            )
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:4rem 2rem; color:#555;'>
                <div style='font-size:3rem;'>👗</div>
                <div style='font-family:Space Mono,monospace; margin-top:1rem;'>
                    Upload or sample an image to begin
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRAINING LAUNCHER
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("### Launch training from the GUI")
    st.info("Configure hyperparameters and run the training script. Progress is shown in the terminal where Streamlit is launched.")

    c1, c2, c3 = st.columns(3)
    with c1:
        t_epochs      = st.number_input("Epochs",       min_value=1,  max_value=200, value=30)
        t_batch       = st.number_input("Batch size",   min_value=8,  max_value=256, value=64, step=8)
    with c2:
        t_lr          = st.number_input("Learning rate", value=1e-3, format="%.5f")
        t_noise       = st.slider("Noise factor (σ)",   0.05, 0.8, 0.4, 0.05)
    with c3:
        t_max_samples = st.number_input("Max samples (0 = all)", min_value=0, max_value=100000, value=0, step=1000)
        t_img_size    = st.selectbox("Image size", [64, 128, 256], index=1)

    t_save = st.text_input("Save checkpoint to", value="models/dae_fashion.pth")

    if st.button("🚀  Start Training"):
        max_s = f"--max_samples {t_max_samples}" if t_max_samples > 0 else ""
        cmd = (
            f"python train.py "
            f"--data_csv \"{csv_path}\" "
            f"--img_dir \"{img_dir}\" "
            f"--epochs {t_epochs} "
            f"--batch_size {t_batch} "
            f"--lr {t_lr} "
            f"--noise_factor {t_noise} "
            f"--img_size {t_img_size} "
            f"--save_path \"{t_save}\" "
            f"{max_s}"
        )
        st.code(cmd, language="bash")
        st.info("Copy the command above and run it in your terminal, OR paste it into a Kaggle notebook cell with `!` prefix.")

    st.divider()
    st.markdown("#### Architecture summary")
    st.code(str(model), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ### What is a Denoising Autoencoder?

    A **Denoising Autoencoder (DAE)** is a neural network trained to reconstruct clean data
    from corrupted (noisy) inputs.  During training:

    1. A clean image **x** is sampled from the dataset.
    2. Noise is added → **x̃** (corrupted input).
    3. The network learns the mapping **x̃ → x** by minimising MSE loss.

    The encoder compresses the image into a compact latent representation;
    the decoder reconstructs the clean image from that representation.

    ### Architecture

    | Stage | Layers | Output channels |
    |-------|--------|-----------------|
    | Encoder block 1 | Conv → BN → ReLU → MaxPool | 32 |
    | Encoder block 2 | Conv → BN → ReLU → MaxPool | 64 |
    | Encoder block 3 | Conv → BN → ReLU → MaxPool | 128 |
    | Bottleneck | Conv → BN → ReLU | 256 |
    | Decoder block 1 | ConvTranspose → BN → ReLU | 128 |
    | Decoder block 2 | ConvTranspose → BN → ReLU | 64 |
    | Decoder block 3 | ConvTranspose → BN → ReLU | 32 |
    | Output | Conv → Sigmoid | 3 |

    ### Noise types supported

    | Type | Description |
    |------|-------------|
    | **Gaussian** | Additive white Gaussian noise – most common |
    | **Salt & Pepper** | Random black/white pixel corruption |
    | **Speckle** | Multiplicative noise (common in medical images) |

    ### Metrics

    - **PSNR** – Peak Signal-to-Noise Ratio (higher = better, dB scale)
    - **SSIM** – Structural Similarity Index (1 = perfect, 0 = no similarity)

    ### Files

    ```
    project/
    ├── autoencoder_model.py   # Model definition
    ├── train.py               # Training script
    ├── app.py                 # This Streamlit app
    └── models/
        └── dae_fashion.pth    # Trained checkpoint (created by train.py)
    ```
    """)
