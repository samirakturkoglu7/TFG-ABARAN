"""
entrenar_resnet1d.py
====================
Entrenamiento de un modelo ResNet-1D para señales armónicas complejas.
Este modelo servirá como 'Teacher' o modelo base avanzado para 
pruebas exhaustivas de cuantización (PTQ, QAT, Per-Channel) y Pruning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import time

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001
N_SAMPLES_TRAIN = 8000
N_SAMPLES_TEST = 2000
INPUT_SIZE = 1024
N_CLASSES = 10

# --- 1. DATASET MÁS COMPLEJO (MÁS RUIDO) ---
def generate_complex_dataset(n_samples, input_size, n_classes):
    """
    Genera señales armónicas 1D bajo condiciones de extremo ruido y solapamiento.
    Forzamos un SNR (Signal-to-Noise Ratio) muy pobre para que la ResNet
    no alcance el 100% de Accuracy fácilmente y sirva para evaluar cuantización.
    """
    print(f"Generando {n_samples} muestras de dataset extremo...")
    t = np.linspace(0, 1, input_size, dtype=np.float32)
    X = np.zeros((n_samples, 1, input_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        dominant = np.random.randint(1, 6)
        signal = np.zeros(input_size, dtype=np.float32)
        
        for h in range(1, 6):
            # La diferencia de amplitud entre la dominante y las demás es mínima
            amp = 0.5 if h != dominant else 0.8
            # Ruido de fase severo e inestable a lo largo de la señal
            phase = np.random.uniform(0, 2 * math.pi)
            # Frecuencias ligeramente desplazadas para evitar patrones puros
            freq_shift = np.random.uniform(-2.0, 2.0)
            
            signal += amp * np.sin(2 * math.pi * (h * 50.0 + freq_shift) * t + phase)
            
        # Ruido de fondo brutal (std dev = 2.5) que casi entierra la señal original
        signal += np.random.normal(0, 2.5, input_size).astype(np.float32)
        
        X[i, 0, :] = signal
        y[i] = (dominant - 1) % n_classes
        
    return torch.from_numpy(X), torch.from_numpy(y)
# --- 2. ARQUITECTURA RESNET-1D ---
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
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
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet1D, self).__init__()
        self.in_channels = 16
        # Capa inicial
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Bloques residuales
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        # Clasificador final
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- 3. ENTRENAMIENTO ---
def train_model():
    print(f"Dispositivo de entrenamiento: {DEVICE}")
    
    # Generar datos
    X_train, y_train = generate_complex_dataset(N_SAMPLES_TRAIN, INPUT_SIZE, N_CLASSES)
    X_test, y_test = generate_complex_dataset(N_SAMPLES_TEST, INPUT_SIZE, N_CLASSES)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # Inicializar modelo
    model = ResNet1D(num_classes=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("\nIniciando entrenamiento...")
    tiempo_inicio = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Evaluación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {val_acc:.2f}%")
        
    tiempo_total = time.time() - tiempo_inicio
    print(f"\nEntrenamiento completado en {tiempo_total:.2f} segundos.")
    
    # Guardar modelo
    nombre_archivo = "resnet1d_complejo.pth"
    torch.save(model.state_dict(), nombre_archivo)
    print(f"Modelo guardado exitosamente como '{nombre_archivo}'")

if __name__ == "__main__":
    train_model()
