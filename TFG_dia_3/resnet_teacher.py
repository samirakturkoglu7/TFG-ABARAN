import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time

DEVICE = torch.device("cuda")

def generate_dataset(n_samples, input_size, n_classes):
    t = np.linspace(0, 1, input_size, dtype=np.float32)
    X = np.zeros((n_samples, input_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        dominant = np.random.randint(1, 6)
        signal = np.zeros(input_size, dtype=np.float32)
        for h in range(1, 6):
            amp = (1.0 / h) if h != dominant else 2.0
            phase = np.random.uniform(0, 2 * math.pi)
            signal += amp * np.sin(2 * math.pi * h * 50.0 * t + phase)
        signal += np.random.normal(0, 0.1, input_size).astype(np.float32)
        X[i] = signal
        y[i] = (dominant - 1) % n_classes
    return torch.from_numpy(X), torch.from_numpy(y)

class BasicBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)

class ResNet1D(nn.Module):
    def __init__(self, filters=64, n_blocks=4, num_classes=10):
        super().__init__()
        self.conv1  = nn.Conv1d(1, filters, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm1d(filters)
        self.blocks = nn.Sequential(*[BasicBlock1D(filters) for _ in range(n_blocks)])
        self.linear = nn.Linear(filters, num_classes)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = torch.mean(out, dim=2)
        return self.linear(out)

print("Generando dataset...")
X, y = generate_dataset(5000, 1024, 10)
X, y = X.to(DEVICE), y.to(DEVICE)

model = ResNet1D().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print("Entrenando ResNet-1D (teacher)...")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()

for epoch in range(5):
    perm = torch.randperm(5000)
    total_loss, n_batches = 0, 0
    for i in range(0, 5000, 32):
        idx = perm[i:i+32]
        xb = X[idx].unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(xb), y[idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    print(f"  Epoch {epoch+1}/5 loss: {total_loss/n_batches:.4f}")

torch.cuda.synchronize()
t_train = (time.perf_counter() - t0) * 1000
peak_vram = torch.cuda.max_memory_allocated() / 1024**2

# Accuracy por batches
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for i in range(0, 5000, 32):
        xb = X[i:i+32].unsqueeze(1)
        preds = model(xb).argmax(dim=1)
        correct += (preds == y[i:i+32]).sum().item()
        total += y[i:i+32].size(0)
acc = correct / total

print(f"\n── Resultados ──")
print(f"Tiempo entrenamiento: {t_train:.0f} ms")
print(f"VRAM pico:            {peak_vram:.1f} MB")
print(f"Accuracy (train):     {acc*100:.1f}%")

torch.save(model.state_dict(), "resnet_teacher.pth")
print("Modelo guardado en resnet_teacher.pth")
