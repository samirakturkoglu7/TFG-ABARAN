import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
import numpy as np
import math
import os
import time

# --- CONFIGURACIÓN E INFRAESTRUCTURA JETSON ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_QUANT = torch.device("cpu") # PyTorch Quantization Backend en ARM
torch.backends.quantized.engine = 'qnnpack'

INPUT_SIZE = 1024
N_CLASSES = 10

# --- 1. SINTESIS DE DATASET (RF/OFDM Overlapping Dataset) ---
def generate_complex_dataset(n_samples, input_size=1024, n_classes=10):
    t = np.linspace(0, 1, input_size, dtype=np.float32)
    X = np.zeros((n_samples, 1, input_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        label = np.random.randint(0, n_classes)
        f_base = 20.0 + (label * 8.0) + np.random.uniform(-3.0, 3.0)
        f_sec = 20.0 + ((label + 1) % n_classes * 8.0) + np.random.uniform(-2.0, 2.0)
        amp_main, amp_sec = np.random.uniform(0.8, 1.2), np.random.uniform(0.5, 0.9)
        phase1, phase2 = np.random.uniform(0, 2 * math.pi), np.random.uniform(0, 2 * math.pi)
        signal = amp_main * np.sin(2 * math.pi * f_base * t + phase1) + \
                 amp_sec * np.sin(2 * math.pi * f_sec * t + phase2)
        signal += np.random.normal(0, 0.8, input_size).astype(np.float32)
        X[i, 0, :] = signal
        y[i] = label
    return torch.from_numpy(X), torch.from_numpy(y)

# --- 2. ARQUITECTURA RESNET-1D (Base) ---
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
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, num_classes=10, base_channels=16):
        super().__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels*2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels*4, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels*4, num_classes)

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

# --- 3. LOSS DE KNOWLEDGE DISTILLATION (KD) ---
def kd_loss_function(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    T: Temperatura para suavizar distribuciones de probabilidad (KD Soft Targets)
    alpha: Balance entre la pérdida del Teacher (KD) y Hard Targets (Cross Entropy)
    """
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_prob = F.log_softmax(student_logits / T, dim=1)
    distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)
    student_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation_loss + (1.0 - alpha) * student_loss

# --- 4. BENCHMARKING DE LATENCIA Y MEMORIA EN JETSON ---
def measure_latency_and_size(model, input_tensor, runs=100, is_quantized=False):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = buf.tell() / (1024 ** 2)
    

    dev = DEVICE_QUANT if is_quantized else DEVICE
    
    model = model.to(dev)
    input_tensor = input_tensor.to(dev)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
            
    if dev.type == 'cuda':
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(input_tensor)
            torch.cuda.synchronize()
        end = time.time()
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(input_tensor)
        end = time.time()
        
    avg_latency_ms = ((end - start) / runs) * 1000.0
    return size_mb, avg_latency_ms

# --- 5. EJECUCIÓN DEL FLUJO COMPLETO ---
if __name__ == "__main__":
    print(f"=== INICIANDO PIPELINE DE KD Y CUANTIZACIÓN EN JETSON ===")
    print(f"Dispositivo actual: {DEVICE}")
    
    # Dataset
    X_train, y_train = generate_complex_dataset(6000)
    X_test, y_test = generate_complex_dataset(1000)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    
    # A) CARGAR / ENTRENAR TEACHER (Modelo grande: base_channels=32)
    print("\n[1] Configurando Modelo Irakasle (Teacher FP32)...")
    teacher = ResNet1D(num_classes=N_CLASSES, base_channels=32).to(DEVICE)
    # Entrenar rápida adaptación del irakasle
    opt_t = optim.Adam(teacher.parameters(), lr=1e-3)
    teacher.train()
    for ep in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_t.zero_grad()
            loss = F.cross_entropy(teacher(bx), by)
            loss.backward()
            opt_t.step()
    teacher.eval()

    # B) ENTRENAR STUDENT CON KD (Modelo compacto: base_channels=16)
    print("\n[2] Entrenando Modelo Ikasle (Student FP32) mediante Knowledge Distillation...")
    student_kd = ResNet1D(num_classes=N_CLASSES, base_channels=16).to(DEVICE)
    opt_s = optim.Adam(student_kd.parameters(), lr=1e-3)
    
    student_kd.train()
    for ep in range(10):
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_s.zero_grad()
            with torch.no_grad():
                t_logits = teacher(bx)
            s_logits = student_kd(bx)
            loss = kd_loss_function(s_logits, t_logits, by, T=4.0, alpha=0.7)
            loss.backward()
            opt_s.step()
            
    # C) EVALUACIÓN BASE DE ACCURACY
    def calc_acc(m, dev):
        m.eval()
        m.to(dev)
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_test), 64):
                bx, by = X_test[i:i+64].to(dev), y_test[i:i+64].to(dev)
                preds = m(bx).argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        return (correct / total) * 100.0

    acc_teacher = calc_acc(teacher, DEVICE)
    acc_student_kd = calc_acc(student_kd, DEVICE)
    
    # D) APLICAR CUANTIZACIÓN PTQ AL STUDENT KD
    print("\n[3] Aplicando PTQ INT8 al Ikasle (Student Distilled)...")
    student_kd_cpu = student_kd.to(DEVICE_QUANT)
    qconfig = QConfigMapping().set_global(tq.get_default_qconfig('qnnpack'))
    example_input = (torch.randn(1, 1, 1024),)
    
    prep_model = prepare_fx(student_kd_cpu, qconfig, example_input)
    # Calibración
    with torch.no_grad():
        prep_model(X_test[:500].to(DEVICE_QUANT))
    quant_student = convert_fx(prep_model)
    
    acc_student_quant = calc_acc(quant_student, DEVICE_QUANT)

    # E) NEURKETAK / MEDIDAS DE RENDIMIENTO
    sample_in = torch.randn(1, 1, 1024)
    size_t, lat_t = measure_latency_and_size(teacher, sample_in)
    size_s, lat_s = measure_latency_and_size(student_kd, sample_in)
    size_q, lat_q = measure_latency_and_size(quant_student, sample_in, is_quantized=True)

    print("\n" + "="*70)
    print("  RESULTADOS DE NEURKETAK EN NVIDIA JETSON ORIN NANO")
    print("="*70)
    print(f"{'Modelo':<25} | {'Accuracy (%)':<12} | {'Tamaño (MB)':<12} | {'Latencia (ms)':<15}")
    print("-" * 70)
    print(f"{'Teacher (FP32 - GPU)':<25} | {acc_teacher:<12.2f} | {size_t:<12.2f} | {lat_t:<15.3f}")
    print(f"{'Student KD (FP32 - GPU)':<25} | {acc_student_kd:<12.2f} | {size_s:<12.2f} | {lat_s:<15.3f}")
    print(f"{'Student KD (INT8 - CPU)':<25} | {acc_student_quant:<12.2f} | {size_q:<12.2f} | {lat_q:<15.3f}")
    print("="*70)
    
# Guardar el modelo cuantizado (TorchScript para despliegue directo)
scripted_model = torch.jit.script(quant_student)
scripted_model.save("student_kd_int8_jetson.pt")
print("Modelo INT8 exportado exitosamente como 'student_kd_int8_jetson.pt'")
