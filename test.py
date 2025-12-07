import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = "data/abstracts"   
BATCH_SIZE = 32     # smaller batch for faster training
IMAGE_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100
EPOCHS = 100        # fast training for fun output
LR = 0.0002
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Cuda makes it go wayyyyyy faster, installation reccomended if applicable
SAVE_INTERVAL = 20  # display images every N epochs


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*CHANNELS, [0.5]*CHANNELS)  
])

dataset = CustomImageDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)


netG = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
netD = Discriminator(CHANNELS).to(DEVICE)


criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))


real_label = 1.
fake_label = 0.
fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)  # smaller grid for preview

print("Starting Training Loop...")
for epoch in range(EPOCHS):
    for i, (data, _) in enumerate(dataloader):
        netD.zero_grad()
        real_data = data.to(DEVICE)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, device=DEVICE)
        output = netD(real_data)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_data = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_data.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        optimizerD.step()
        lossD = lossD_real + lossD_fake

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_data)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if i % 5 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                  f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")


    if (epoch+1) % SAVE_INTERVAL == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = utils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(6,6))
        plt.imshow(np.transpose(grid, (1,2,0)))
        plt.title(f"Epoch {epoch+1}")
        plt.axis('off')
        plt.show()
        utils.save_image(fake, f"generated_epoch_{epoch+1}.png", normalize=True)

print("Training finished!")