#!/usr/bin/env python3
"""
entrenar_resnet1d_exp.py
========================
Version parametrizada de entrenar_resnet1d_2*.py para experimentos controlados
de memoria en Jetson Orin Nano + Docker.

Mismos modelo, dataset y bucle de entrenamiento que los scripts originales.
Cambios:
  - Cada factor experimental es un argumento (batch, tamano de dataset, empty_cache).
  - Semilla fija. El dataset de N muestras es prefijo del de 2N con la misma semilla,
    asi que "reducir el dataset" no cambia la distribucion.
  - Instrumentacion de memoria: cgroup (actual, pico del kernel, pico muestreado,
    anon), RSS del proceso y estadisticas CUDA, registradas por etapa en un
    .events.jsonl con fsync. Si el proceso muere (SIGKILL por OOM) queda la ultima
    etapa alcanzada.
  - Si falla con una excepcion, se guarda la etapa y el error en el .json final.
"""

import argparse
import json
import math
import os
import platform
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

T0 = time.time()
MB = 1024 * 1024
CUDA_READY = False

# --------------------------------------------------------------------------
# Medicion de memoria (cgroup v2 con fallback a v1)
# --------------------------------------------------------------------------
CG_CURRENT = ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
CG_PEAK = ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
CG_LIMIT = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")


def mb(x):
    return None if x is None else round(x / MB, 1)


def read_int(paths):
    for p in paths:
        try:
            with open(p) as f:
                s = f.read().strip()
            if s != "max":
                return int(s)
        except (OSError, ValueError):
            continue
    return None


def cg_anon():
    """Memoria anonima del cgroup (excluye page cache, que es recuperable)."""
    for p, key in (("/sys/fs/cgroup/memory.stat", "anon"),
                   ("/sys/fs/cgroup/memory/memory.stat", "total_rss")):
        try:
            with open(p) as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    if k == key:
                        return int(v)
        except (OSError, ValueError):
            continue
    return None


def proc_status(key):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class Sampler(threading.Thread):
    """Muestrea el cgroup cada 50 ms para obtener picos aunque no haya memory.peak."""

    def __init__(self, period=0.01):
        super().__init__(daemon=True)
        self.period = period
        self.peak_current = 0
        self.peak_anon = 0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            c = read_int(CG_CURRENT)
            a = cg_anon()
            if c:
                self.peak_current = max(self.peak_current, c)
            if a:
                self.peak_anon = max(self.peak_anon, a)
            time.sleep(self.period)

    def stop(self):
        self._halt.set()


SAMPLER = Sampler()


def snapshot(stage):
    d = {
        "stage": stage,
        "t": round(time.time() - T0, 2),
        "cg_current_mb": mb(read_int(CG_CURRENT)),
        "cg_anon_mb": mb(cg_anon()),
        "cg_peak_kernel_mb": mb(read_int(CG_PEAK)),
        "cg_peak_sampled_mb": mb(SAMPLER.peak_current),
        "cg_anon_peak_sampled_mb": mb(SAMPLER.peak_anon),
        "rss_mb": mb(proc_status("VmRSS")),
        "rss_peak_mb": mb(proc_status("VmHWM")),
    }
    if CUDA_READY:
        d["cuda_alloc_mb"] = mb(torch.cuda.memory_allocated())
        d["cuda_max_alloc_mb"] = mb(torch.cuda.max_memory_allocated())
        d["cuda_reserved_mb"] = mb(torch.cuda.memory_reserved())
        try:
            free, total = torch.cuda.mem_get_info()
            d["cuda_free_mb"], d["cuda_total_mb"] = mb(free), mb(total)
        except Exception:
            pass
    return d


class EventLog:
    def __init__(self, path):
        self.f = open(path, "w")
        self.last_stage = "none"

    def log(self, stage, **extra):
        d = snapshot(stage)
        d.update(extra)
        self.f.write(json.dumps(d) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        self.last_stage = stage
        print(f"[{d['t']:8.2f}s] {stage:<18} cg={d['cg_current_mb']} MB  "
              f"anon={d['cg_anon_mb']} MB  rss={d['rss_mb']} MB  "
              f"cuda_alloc={d.get('cuda_alloc_mb')} MB", flush=True)
        return d


# --------------------------------------------------------------------------
# Dataset (misma formula que los scripts originales, con RNG con semilla)
# --------------------------------------------------------------------------
def generate_dataset(n_samples, input_size, n_classes, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, input_size, dtype=np.float32)
    X = np.zeros((n_samples, 1, input_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        label = int(rng.integers(0, n_classes))
        f_base = 20.0 + label * 8.0 + rng.uniform(-3.0, 3.0)
        f_sec = 20.0 + ((label + 1) % n_classes) * 8.0 + rng.uniform(-2.0, 2.0)
        amp_main = rng.uniform(0.8, 1.2)
        amp_sec = rng.uniform(0.5, 0.9)
        phase1 = rng.uniform(0, 2 * math.pi)
        phase2 = rng.uniform(0, 2 * math.pi)
        signal = (amp_main * np.sin(2 * math.pi * f_base * t + phase1)
                  + amp_sec * np.sin(2 * math.pi * f_sec * t + phase2))
        signal += rng.normal(0, 0.8, input_size).astype(np.float32)
        X[i, 0, :] = signal
        y[i] = label
    return torch.from_numpy(X), torch.from_numpy(y)


# --------------------------------------------------------------------------
# Modelo (identico al original)
# --------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# --------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Nombre unico de la ejecucion")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--empty-cache", choices=["none", "start", "epoch"], default="none",
                   help="start = como en entrenar_resnet1d_2.py (antes de iniciar CUDA); "
                        "epoch = al final de cada epoca")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Si >0, para tras N pasos de entrenamiento (sonda de huella del runtime)")
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run(args, ev, result):
    global CUDA_READY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result["device"] = str(device)
    ev.log("start")

    # Replica la posicion original: empty_cache() antes de cualquier uso de CUDA.
    # Con CUDA sin inicializar PyTorch lo ignora, asi que esta variante es un no-op real.
    if args.empty_cache == "start" and device.type == "cuda":
        torch.cuda.empty_cache()
        ev.log("empty_cache_start")

    if device.type == "cuda":
        torch.zeros(1, device=device)  # fuerza la creacion del contexto CUDA
        CUDA_READY = True
        result["gpu_name"] = torch.cuda.get_device_name(0)
        ev.log("cuda_init")

    X_train, y_train = generate_dataset(args.n_train, args.input_size, args.n_classes, args.seed)
    X_test, y_test = generate_dataset(args.n_test, args.input_size, args.n_classes, args.seed + 1000)
    ev.log("data_ready")

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                              shuffle=True, generator=g)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = ResNet1D(num_classes=args.n_classes).to(device)
    result["n_params"] = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ev.log("model_on_device")

    step = 0
    stop = False
    t_train = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if step == 0:
                sync(device)
                ev.log("first_forward")
            loss = criterion(outputs, labels)
            loss.backward()
            if step == 0:
                sync(device)
                ev.log("first_backward")
            optimizer.step()
            running += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predicted = model(inputs).argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total
        result["final_val_acc"] = round(val_acc, 2)
        ev.log(f"epoch_{epoch + 1}", loss=round(running / len(train_loader), 4),
               val_acc=round(val_acc, 2))
        if args.empty_cache == "epoch" and device.type == "cuda":
            torch.cuda.empty_cache()

    sync(device)
    result["train_time_s"] = round(time.time() - t_train, 2)
    result["steps"] = step
    if args.save_model and not args.max_steps:
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.tag}.pth"))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ev = EventLog(os.path.join(args.out_dir, f"{args.tag}.events.jsonl"))
    SAMPLER.start()

    result = {
        "tag": args.tag,
        "config": vars(args),
        "status": "ok",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "cg_limit_mb": mb(read_int(CG_LIMIT)),
    }
    try:
        run(args, ev, result)
    except Exception as e:  # fallo "limpio" (p. ej. CUBLAS_STATUS_ALLOC_FAILED)
        result["status"] = "fail"
        result["failed_after_stage"] = ev.last_stage
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"\nFALLO tras la etapa '{ev.last_stage}': {result['error']}", flush=True)
        try:
            ev.log("FAIL")
        except Exception:
            pass

    try:
        result["final"] = snapshot("end")
    except Exception:
        pass
    SAMPLER.stop()
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
