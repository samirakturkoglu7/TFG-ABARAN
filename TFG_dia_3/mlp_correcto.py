import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import psutil
import os

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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)

print("Generando dataset...")
X, y = generate_dataset(5000, 1024, 10)
X, y = X.to(DEVICE), y.to(DEVICE)

model = MLP().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Entrenando MLP con dataset correcto...")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()

for epoch in range(5):
    perm = torch.randperm(5000)
    total_loss, n_batches = 0, 0
    for i in range(0, 5000, 256):
        idx = perm[i:i+256]
        optimizer.zero_grad()
        loss = criterion(model(X[idx]), y[idx])
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
    for i in range(0, 5000, 256):
        preds = model(X[i:i+256]).argmax(dim=1)
        correct += (preds == y[i:i+256]).sum().item()
        total += y[i:i+256].size(0)
acc = correct / total

print(f"\n── Resultados ──")
print(f"Tiempo entrenamiento: {t_train:.0f} ms")
print(f"VRAM pico:            {peak_vram:.1f} MB")
print(f"Accuracy (train):     {acc*100:.1f}%")

# Guardar el modelo para usarlo como student en KD
torch.save(model.state_dict(), "mlp_student.pth")
print("Modelo guardado en mlp_student.pth")

# Inferencia
print("\nMidiendo inferencia...")
x_sample = X[:1]
latencias = []
with torch.no_grad():
    for _ in range(10):
        _ = model(x_sample)
    torch.cuda.synchronize()
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x_sample)
        torch.cuda.synchronize()
        latencias.append((time.perf_counter() - t0) * 1000)

import statistics
print(f"Latencia media: {statistics.mean(latencias):.3f} ms")
