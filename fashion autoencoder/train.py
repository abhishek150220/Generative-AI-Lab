
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from autoencoder_model import DenoisingAutoencoder


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class FashionDataset(Dataset):
    """Loads fashion product images from the Kaggle dataset."""

    def __init__(self, image_paths, img_size=128, augment=False):
        self.paths = image_paths
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),          # → [0, 1]
        ]
        if augment:
            base_transforms.insert(1, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)
        except Exception:
            # Return a black image if file is corrupt / missing
            return torch.zeros(3, 128, 128)


def add_noise(images: torch.Tensor, noise_factor: float = 0.4) -> torch.Tensor:
    """Add Gaussian noise and clamp to [0, 1]."""
    noise = torch.randn_like(images) * noise_factor
    return (images + noise).clamp(0.0, 1.0)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶  Using device: {device}")

    # --- Build image list ---
    df = pd.read_csv(args.data_csv)
    all_paths = [os.path.join(args.img_dir, f) for f in df["filename"].tolist()]
    all_paths = [p for p in all_paths if os.path.exists(p)]
    print(f"▶  Found {len(all_paths)} valid images")

    # Optional: train on a subset to speed things up
    if args.max_samples and args.max_samples < len(all_paths):
        all_paths = random.sample(all_paths, args.max_samples)
        print(f"▶  Subsampling to {len(all_paths)} images")

    # --- Train / Val split ---
    full_ds = FashionDataset(all_paths, img_size=args.img_size, augment=True)
    val_size = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # --- Model, loss, optimiser ---
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for clean in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            clean = clean.to(device)
            noisy = add_noise(clean, args.noise_factor)

            optimizer.zero_grad()
            recon = model(noisy)
            loss  = criterion(recon, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * clean.size(0)

        train_loss /= train_size

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean in val_loader:
                clean = clean.to(device)
                noisy = add_noise(clean, args.noise_factor)
                recon = model(noisy)
                val_loss += criterion(recon, clean).item() * clean.size(0)
        val_loss /= val_size

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"           ✔  Saved best model → {args.save_path}")

    print(f"\n✅  Training complete.  Best val loss: {best_val_loss:.6f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Denoising Autoencoder")

    parser.add_argument("--data_csv",      default="/kaggle/input/datasets/paramaggarwal/fashion-product-images-dataset/fashion-dataset/images.csv")
    parser.add_argument("--img_dir",       default="/kaggle/input/datasets/paramaggarwal/fashion-product-images-dataset/fashion-dataset/images/")
    parser.add_argument("--save_path",     default="models/dae_fashion.pth")
    parser.add_argument("--img_size",      type=int,   default=128)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--noise_factor",  type=float, default=0.4,
                        help="Std of Gaussian noise added to inputs (0 = clean AE)")
    parser.add_argument("--max_samples",   type=int,   default=None,
                        help="Limit dataset size for quick experiments")
    parser.add_argument("--workers",       type=int,   default=4)

    args = parser.parse_args()
    train(args)
