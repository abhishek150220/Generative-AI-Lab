import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder for fashion product images.
    Encoder compresses the image; Decoder reconstructs the clean version.
    """

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Block 1: (3, H, W) → (32, H/2, W/2)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, H/2, W/2) → (64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, H/4, W/4) → (128, H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bottleneck: (128, H/8, W/8) → (256, H/8, W/8)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            # Block 1: (256, H/8, W/8) → (128, H/4, W/4)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: (128, H/4, W/4) → (64, H/2, W/2)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (64, H/2, W/2) → (32, H, W)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Output: (32, H, W) → (3, H, W)
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),          # pixel values in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
