import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time
import statistics
import os

DEVICE = torch.device("cuda")

# ── DATASET ──────────────────────────────────────────────────────────────────
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

# ── ARQUITECTURAS ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)

class BasicBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + x)

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
        return self.linear(torch.mean(out, dim=2))

# ── EARLY STOPPING (adaptado de Unai) ────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_acc = -1
        self.counter  = 0
        self.best_state = None
    def step(self, val_acc, model):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter  = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience
    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

# ── EVALUACIÓN ────────────────────────────────────────────────────────────────
def accuracy(model, X, y, batch_size=256, unsqueeze=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            if unsqueeze:
                xb = xb.unsqueeze(1)
            preds = model(xb).argmax(dim=1)
            correct += (preds == y[i:i+batch_size]).sum().item()
            total   += y[i:i+batch_size].size(0)
    return correct / total

# ── ENTRENAMIENTO KD (lógica de Unai adaptada a señales 1D) ──────────────────
def train_kd(teacher, student, X, y, epochs, T, alpha, label="KD"):
    """
    teacher: ResNet1D (recibe [B, 1, 1024])
    student: MLP      (recibe [B, 1024])
    T:       temperatura — suaviza las probabilidades del teacher
    alpha:   peso de la loss KD frente a CrossEntropy real
             alpha=0 → solo CrossEntropy (sin KD)
             alpha=1 → solo imita al teacher
    """
    ce_loss   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    stopper   = EarlyStopping(patience=5)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"\n── Entrenando student con {label} (T={T}, alpha={alpha}) ──")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        student.train()
        perm = torch.randperm(len(X))
        total_loss, n_batches = 0, 0

        for i in range(0, len(X), 256):
            idx = perm[i:i+256]
            xb  = X[idx]          # [B, 1024]  para MLP
            xb_t = xb.unsqueeze(1) # [B, 1, 1024] para ResNet
            yb  = y[idx]

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(xb_t)  # ResNet recibe señal con canal

            student_logits = student(xb)         # MLP recibe señal plana

            # Soft labels del teacher (con temperatura T)
            log_p  = F.log_softmax(student_logits / T, dim=1)
            q_t    = F.softmax(teacher_logits    / T, dim=1)
            kd_loss = F.kl_div(log_p, q_t, reduction='batchmean') * (T * T)

            # Hard labels (etiquetas reales)
            label_loss = ce_loss(student_logits, yb)

            loss = alpha * kd_loss + (1.0 - alpha) * label_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        val_acc = accuracy(student, X, y)
        print(f"  Epoch {epoch+1}/{epochs} | loss: {total_loss/n_batches:.4f} | acc: {val_acc*100:.1f}%")

        if stopper.step(val_acc, student):
            print(f"  Early stopping en epoch {epoch+1}")
            break

    stopper.restore(student)
    torch.cuda.synchronize()
    t_train  = (time.perf_counter() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    final_acc = accuracy(student, X, y)

    return {"label": label, "T": T, "alpha": alpha,
            "acc": final_acc, "t_train_ms": t_train, "vram_mb": peak_vram}

# ── MEDICIÓN DE LATENCIA ──────────────────────────────────────────────────────
def medir_latencia(model, x_sample, n=100, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_sample)
        torch.cuda.synchronize()
        lats = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x_sample)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(lats), statistics.stdev(lats)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando dataset...")
    X, y = generate_dataset(5000, 1024, 10)
    X, y = X.to(DEVICE), y.to(DEVICE)

    # Cargar teacher preentrenado
    teacher = ResNet1D().to(DEVICE)
    teacher.load_state_dict(torch.load("resnet_teacher.pth", map_location=DEVICE))
    teacher.eval()
    print(f"Teacher cargado — accuracy: {accuracy(teacher, X, y, unsqueeze=True)*100:.1f}%")

    # Barrido de hiperparámetros (adaptado de Unai, reducido para Jetson)
    experimentos = [
        {"T": 2,  "alpha": 0.3},
        {"T": 4,  "alpha": 0.7},
        {"T": 8,  "alpha": 0.9},
    ]

    resultados = []
    for exp in experimentos:
        student = MLP().to(DEVICE)
        r = train_kd(teacher, student, X, y,
                     epochs=10, T=exp["T"], alpha=exp["alpha"],
                     label=f"KD_T{exp['T']}_a{exp['alpha']}")

        # Latencia del student
        x_sample = X[:1]
        lat_mean, lat_std = medir_latencia(student, x_sample)
        r["lat_ms"]  = lat_mean
        r["lat_std"] = lat_std

        # Guardar el mejor modelo KD
        torch.save(student.state_dict(), f"mlp_kd_T{exp['T']}_a{exp['alpha']}.pth")
        resultados.append(r)

    # Tabla resumen
    print("\n" + "="*70)
    print("RESUMEN KNOWLEDGE DISTILLATION")
    print("="*70)
    print(f"{'Config':<20} {'Acc':>8} {'Latencia':>12} {'VRAM':>10} {'Tiempo':>12}")
    print("-"*70)

    # Baseline MLP sin KD (referencia)
    mlp_base = MLP().to(DEVICE)
    mlp_base.load_state_dict(torch.load("mlp_student.pth", map_location=DEVICE))
    acc_base = accuracy(mlp_base, X, y)
    lat_base, _ = medir_latencia(mlp_base, X[:1])
    print(f"{'MLP baseline':<20} {acc_base*100:>7.1f}% {lat_base:>10.3f}ms {'116.7MB':>10} {'(ya medido)':>12}")

    for r in resultados:
        print(f"{r['label']:<20} {r['acc']*100:>7.1f}% {r['lat_ms']:>10.3f}ms {r['vram_mb']:>9.1f}MB {r['t_train_ms']:>10.0f}ms")

    print("="*70)

    # Potencia actual
    curr_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"
    if os.path.exists(curr_path):
        mA = int(open(curr_path).read().strip())
        print(f"\nPotencia actual: {mA/1000*5:.2f} W")
