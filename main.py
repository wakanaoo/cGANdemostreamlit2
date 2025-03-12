import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Generator の定義
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4  # 28//4 = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# ハイパーパラメータ
latent_dim = 10
n_classes = 10

# モデルのロード（weights_only=False を明示）
generator = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
generator.to(device)
generator.eval()

# Streamlit UI
def generate_image(target_label):
    z = torch.randn(1, latent_dim, device=device)
    label_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
    with torch.no_grad():
        gen_img = generator(z, label_tensor)
    gen_img = (gen_img + 1) / 2  # [-1,1] を [0,1] にスケール
    return gen_img.cpu().numpy().squeeze()

st.title("GAN MNIST Image Generator")
st.write("Select a digit (0-9) and generate an image!")

label = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, step=1, value=0)

if st.button("Generate Image"):
    img = generate_image(label)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis("off")
    st.pyplot(fig)
