import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

# VAE 모델 정의
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # 인코더 정의
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 잠재 공간의 평균
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 잠재 공간의 로그 분산

        # 디코더 정의
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 학습 설정
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

# 데이터셋 로드 및 학습
data_dir = './data'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(1, 11):
    train(vae, train_loader, optimizer, epoch)

# 생성 결과 시각화
with torch.no_grad():
    z = torch.randn(64, 20)  # 잠재 공간에서 샘플링
    sample = vae.decode(z).cpu()
    sample = sample.view(64, 1, 28, 28)

    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            ax[i, j].imshow(sample[i * 8 + j][0], cmap='gray')
            ax[i, j].axis('off')
    plt.show()