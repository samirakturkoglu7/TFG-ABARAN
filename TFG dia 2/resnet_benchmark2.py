import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import psutil
import os

DEVICE = torch.device("cuda")

# ── DATASET REAL: señal armónica + ruido (metodología Saida) ──
# Cada muestra es una mezcla de 5 armónicos sinusoidales con ruido gaussiano.
# La clase es el índice del armónico dominante (el de mayor amplitud).
# Esto SÍ tiene relación causal señal -> etiqueta, a diferencia del ruido puro.
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

print("Generando dataset...")
X, y = generate_dataset(5000, 1024, 10)
X, y = X.to(DEVICE), y.to(DEVICE)

class BasicBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)

class ResNet1D(nn.Module):
    def __init__(self, filters=64, n_blocks=4, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(filters)
        self.blocks = nn.Sequential(*[BasicBlock1D(filters) for _ in range(n_blocks)])
        self.linear = nn.Linear(filters, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = torch.mean(out, dim=2)
        return self.linear(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = ResNet1D(filters=64, n_blocks=4).to(DEVICE)
n_params = model.count_parameters()
weight_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
print(f"Parámetros: {n_params:,} | Peso: {weight_mb:.2f} MB")

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print("\nEntrenando...")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()

for epoch in range(5):
    perm = torch.randperm(5000)
    total_loss = 0
    n_batches = 0
    for i in range(0, 5000, 32):
        idx = perm[i:i+32]
        xb = X[idx].unsqueeze(1)
        yb = y[idx]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    print(f"  Epoch {epoch+1}/5 loss media: {total_loss/n_batches:.4f}")

torch.cuda.synchronize()
t_train = (time.perf_counter() - t0) * 1000
peak_vram = torch.cuda.max_memory_allocated() / 1024**2
ram = psutil.Process().memory_info().rss / 1024**2

print(f"\n── Resultados entrenamiento ──")
print(f"Tiempo:      {t_train:.0f} ms")
print(f"VRAM pico:   {peak_vram:.1f} MB")
print(f"RAM proceso: {ram:.1f} MB")

# Accuracy en el propio set de entrenamiento (orientativo)
model.eval()
with torch.no_grad():
    preds = model(X.unsqueeze(1)).argmax(dim=1)
    acc = (preds == y).float().mean().item()
print(f"Accuracy (train set): {acc*100:.1f}%")

# ── INFERENCIA ──
print("\nMidiendo inferencia (100 runs, batch=1)...")
x_sample = X[:1].unsqueeze(1)
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
print(f"\n── Resultados inferencia ──")
print(f"Latencia media: {statistics.mean(latencias):.3f} ms")
print(f"Latencia std:   {statistics.stdev(latencias):.3f} ms")
print(f"Latencia min:   {min(latencias):.3f} ms")
print(f"Latencia max:   {max(latencias):.3f} ms")

curr_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"
if os.path.exists(curr_path):
    mA = int(open(curr_path).read().strip())
    print(f"\nPotencia actual: {mA/1000*5:.2f} W")
