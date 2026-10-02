"""
estudio_cuantizacion_resnet.py
==============================
Estudio completo de cuantización PTQ y QAT sobre el modelo ResNet-1D.
Compara: FP32 vs. INT8 (PTQ) vs. INT8 (QAT)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization import QConfigMapping
import numpy as np
import math
import os
import time

# --- CONFIGURACIÓN ---
DEVICE_QUANT = torch.device("cpu")
torch.backends.quantized.engine = 'qnnpack' # Backend ARM Jetson

# --- 1. DATASET EXACTAMENTE IGUAL QUE EN EL ENTRENAMIENTO 2 ---
def generate_complex_dataset(n_samples, input_size, n_classes):
    t = np.linspace(0, 1, input_size, dtype=np.float32)
    X = np.zeros((n_samples, 1, input_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        label = np.random.randint(0, n_classes)
        f_base = 20.0 + (label * 8.0) + np.random.uniform(-3.0, 3.0)
        f_sec = 20.0 + ((label + 1) % n_classes * 8.0) + np.random.uniform(-2.0, 2.0)
        amp_main = np.random.uniform(0.8, 1.2)
        amp_sec = np.random.uniform(0.5, 0.9) 
        phase1 = np.random.uniform(0, 2 * math.pi)
        phase2 = np.random.uniform(0, 2 * math.pi)
        signal = amp_main * np.sin(2 * math.pi * f_base * t + phase1) + \
                 amp_sec * np.sin(2 * math.pi * f_sec * t + phase2)
        signal += np.random.normal(0, 0.8, input_size).astype(np.float32)
        X[i, 0, :] = signal
        y[i] = label
    return torch.from_numpy(X), torch.from_numpy(y)

# --- 2. ARQUITECTURA RESNET-1D ---
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

# --- 3. UTILIDADES ---
def accuracy(model, X, y, batch_size=64):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE_QUANT)
            yb = y[i:i+batch_size].to(DEVICE_QUANT)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

def tamanyo_mb(model):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024**2

# --- 4. PIPELINE ---
if __name__ == "__main__":
    print("\nGenerando dataset de validación...")
    X_test, y_test = generate_complex_dataset(2000, 1024, 10)
    # Muestras para calibrar PTQ
    X_calib = X_test[:500] 

    # 1. CARGAR FP32 BASE
    model_fp32 = ResNet1D()
    if not os.path.exists("resnet1d_complejo_2.pth"):
        raise FileNotFoundError("¡Falta resnet1d_complejo_2.pth! Ejecuta el entrenamiento primero.")
    
    model_fp32.load_state_dict(torch.load("resnet1d_complejo_2.pth", map_location=DEVICE_QUANT, weights_only=True))
    model_fp32.eval()
    
    acc_fp32 = accuracy(model_fp32, X_test, y_test)
    tam_fp32 = tamanyo_mb(model_fp32)
    print(f"\n{'='*40}")
    print(f"[1] BASELINE FP32")
    print(f"  Accuracy: {acc_fp32*100:.2f}% | Tamaño: {tam_fp32:.2f} MB")
    print(f"{'='*40}")

    # 2. CUANTIZACIÓN PTQ (Estática)
    print("\n[2] APLICANDO PTQ ESTÁTICA...")
    qconfig_ptq = QConfigMapping().set_global(tq.get_default_qconfig('qnnpack'))
    example_inputs = (torch.randn(1, 1, 1024),)
    
    model_ptq_prep = prepare_fx(model_fp32, qconfig_ptq, example_inputs)
    
    # Calibración
    print("  Calibrando con 500 muestras...")
    with torch.no_grad(): 
        model_ptq_prep(X_calib)
        
    model_ptq = convert_fx(model_ptq_prep)
    acc_ptq = accuracy(model_ptq, X_test, y_test)
    tam_ptq = tamanyo_mb(model_ptq)
    print(f"  Accuracy PTQ: {acc_ptq*100:.2f}% | Tamaño: {tam_ptq:.2f} MB")
    print(f"  -> Degradación vs FP32: {(acc_fp32 - acc_ptq)*100:.2f} pp")

    # 3. CUANTIZACIÓN QAT
    print("\n[3] APLICANDO QAT (Reentrenamiento 2 épocas simulación INT8)...")
    # Para QAT el modelo original debe entrar en modo train
    model_fp32.train()
    qconfig_qat = QConfigMapping().set_global(tq.get_default_qat_qconfig('qnnpack'))
    
    model_qat_prep = prepare_qat_fx(model_fp32, qconfig_qat, example_inputs)
    
    # Reentrenamos con un Learning Rate muy bajo para no romper lo aprendido
    optimizer = optim.Adam(model_qat_prep.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    X_train_qat, y_train_qat = generate_complex_dataset(2000, 1024, 10)
    loader_qat = DataLoader(TensorDataset(X_train_qat, y_train_qat), batch_size=64, shuffle=True)
    
    for epoch in range(2):
        model_qat_prep.train()
        for inputs, labels in loader_qat:
            optimizer.zero_grad()
            outputs = model_qat_prep(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"  Época QAT {epoch+1}/2 completada.")
        
    # Convertir a INT8 tras QAT
    model_qat_prep.eval()
    model_qat = convert_fx(model_qat_prep)
    
    acc_qat = accuracy(model_qat, X_test, y_test)
    print(f"\n  Accuracy QAT: {acc_qat*100:.2f}% | Tamaño: {tam_ptq:.2f} MB")
    if acc_ptq < acc_fp32:
        print(f"  -> Recuperación gracias a QAT: +{(acc_qat - acc_ptq)*100:.2f} pp")

    print("\n=== RESUMEN FINAL ===")
    print(f"FP32: {acc_fp32*100:.2f}% | PTQ: {acc_ptq*100:.2f}% | QAT: {acc_qat*100:.2f}%")

