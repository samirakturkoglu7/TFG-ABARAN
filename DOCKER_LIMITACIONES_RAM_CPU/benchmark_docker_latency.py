import torch # Librería principal de PyTorch
import torch.nn as nn # Módulo de redes neuronales
import numpy as np # Operaciones matriciales
import time # Medición de tiempos

# Selección del dispositivo de cómputo (GPU CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arquitectura ResNet-1D idéntica a los experimentos previos
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

def measure_latency():
    model = ResNet1D(num_classes=10, base_channels=16).to(DEVICE)
    model.eval()
    
    # Tensor de entrada para inferencia individual (Batch Size = 1)
    dummy_input = torch.randn(1, 1, 1024).to(DEVICE)
    
    # Calentamiento (Warmup) para estabilizar la GPU
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            
    # Medición estricta mediante CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    runs = 500
    latencies = []
    
    with torch.no_grad():
        for _ in range(runs):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize() # Sincronización exacta con la GPU
            latencies.append(start_event.elapsed_time(end_event))
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    print(f"Latencia Media (BS=1): {mean_lat:.3f} ms (± {std_lat:.3f} ms)")

if __name__ == "__main__":
    measure_latency()
