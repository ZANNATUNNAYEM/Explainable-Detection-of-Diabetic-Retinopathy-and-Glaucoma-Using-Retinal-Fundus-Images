"""
Conditional Generative Adversarial Network (cGAN) for synthetic retinal image balancing.
Generates synthetic fundus samples conditioned on underrepresented severity classes.
"""

import os
import torch
import torch.nn as nn
from torchvision.utils import save_image


class Generator(nn.Module):
    """
    Conditional Generator for 224x224 RGB retinal images.
    Takes latent vector z concatenated with class embedding.
    """
    def __init__(self, latent_dim=100, num_classes=5, channels=3, feature_dim=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        in_dim = latent_dim + num_classes

        # Initial projection to 7x7
        self.init_proj = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 7 * 7),
            nn.BatchNorm1d(feature_dim * 8 * 7 * 7),
            nn.ReLU(True)
        )

        self.feature_dim = feature_dim

        # Transposed convs: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),  # 14x14
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),  # 28x28
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),      # 56x56
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim, feature_dim // 2, 4, 2, 1, bias=False),    # 112x112
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim // 2, channels, 4, 2, 1, bias=False),       # 224x224
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=-1)
        out = self.init_proj(x)
        out = out.view(-1, self.feature_dim * 8, 7, 7)
        img = self.deconv_blocks(out)
        return img


class Discriminator(nn.Module):
    """
    Conditional Discriminator with label conditioning via spatial expansion.
    """
    def __init__(self, num_classes=5, channels=3, feature_dim=64, img_size=224):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels + 1, feature_dim, 4, 2, 1, bias=False),  # 112x112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False), # 56x56
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False), # 28x28
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False), # 14x14
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 8, 1, 14, 1, 0, bias=False),              # 1x1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, c], dim=1)
        out = self.conv_blocks(x)
        return out.view(-1, 1).squeeze(1)


class ConditionalGANTrainer:
    """Trainer encapsulation for conditional GAN on retinal fundus subsets."""
    def __init__(self, latent_dim=100, num_classes=5, lr=2e-4, beta1=0.5, beta2=0.999, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.netG = Generator(latent_dim=latent_dim, num_classes=num_classes).to(self.device)
        self.netD = Discriminator(num_classes=num_classes).to(self.device)

        self.criterion = nn.BCELoss()
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, beta2))
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, beta2))

    def train_step(self, real_imgs, labels):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(self.device)
        labels = labels.to(self.device)

        real_target = torch.ones(batch_size, device=self.device)
        fake_target = torch.zeros(batch_size, device=self.device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optD.zero_grad()
        out_real = self.netD(real_imgs, labels)
        loss_real = self.criterion(out_real, real_target)

        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.netG(noise, labels)
        out_fake = self.netD(fake_imgs.detach(), labels)
        loss_fake = self.criterion(out_fake, fake_target)

        loss_d = (loss_real + loss_fake) / 2.0
        loss_d.backward()
        self.optD.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.optG.zero_grad()
        out_gen = self.netD(fake_imgs, labels)
        loss_g = self.criterion(out_gen, real_target)
        loss_g.backward()
        self.optG.step()

        return {"loss_d": loss_d.item(), "loss_g": loss_g.item()}


def generate_synthetic_samples(generator, class_label, num_samples, latent_dim=100, device="cpu"):
    """Generates synthetic retinal fundus images for a given class."""
    generator.eval()
    generator.to(device)
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        syn_imgs = generator(noise, labels)
        # Rescale from [-1, 1] to [0, 1]
        syn_imgs = (syn_imgs + 1.0) / 2.0
    return syn_imgs
