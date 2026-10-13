import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import io

# --- CONFIGURACIÓN DE HARDWARE ---
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")
torch.backends.quantized.engine = 'qnnpack'

# --- ARQUITECTURA RESNET-1D (Student KD) ---
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

class ResNet1D_Student(nn.Module):
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

# --- INFERENTZIA PROFILATZEA ---
def profile_model(model, device, is_cuda=True):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 1, 1024).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    # Neurtu latentzia
    runs = 200
    if is_cuda:
        torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy_input)
            torch.cuda.synchronize()
        t_end = time.time()
    else:
        t_start = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy_input)
        t_end = time.time()
        
    avg_latency = ((t_end - t_start) / runs) * 1000.0
    return avg_latency

if __name__ == "__main__":
    print("\n--- RESNET-1D STUDENT (KD) JETSON-EN APLIKATZEN ---")
    
    # Eredua instantziatu
    model_kd = ResNet1D_Student(num_classes=10, base_channels=16)
    
    # Pisuak kargatu existitzen bada (bestela pisu ausazkoekin probatuko du)
    weights_path = "student_kd_resnet1d.pth"
    try:
        model_kd.load_state_dict(torch.load(weights_path, map_location=DEVICE_GPU))
        print(f"Pisuak ondo kargatu dira: {weights_path}")
    except FileNotFoundError:
        print(f"Oharra: Ez da '{weights_path}' aurkitu. Sarea pisu berriekin aztertuko da.")

    # Tamaina neurtu
    buf = io.BytesIO()
    torch.save(model_kd.state_dict(), buf)
    size_mb = buf.tell() / (1024 ** 2)
    print(f"Sarearen tamaina RAM/VRAMean: {size_mb:.2f} MB")

    # Latentzia neurtu GPU-n
    if torch.cuda.is_available():
        lat_gpu = profile_model(model_kd, DEVICE_GPU, is_cuda=True)
        print(f"Latentzia GPU-n (CUDA Ampere): {lat_gpu:.3f} ms")
    else:
        print("GPU ez dago erabilgarri.")

    # Latentzia neurtu CPU-n
    lat_cpu = profile_model(model_kd, DEVICE_CPU, is_cuda=False)
    print(f"Latentzia CPU-n (ARM Cortex): {lat_cpu:.3f} ms")
