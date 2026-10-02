"""
cuantizacion_ptq.py
====================
Post-Training Static Quantization (PTQ) del MLP baseline y del MLP con KD.
Adaptado de Cuantizacion.py de Unai Rodríguez para señales 1D en Jetson.

Pasos:
  1. Cargar MLP entrenado (baseline y con KD)
  2. Preparar con prepare_fx (inserta observers)
  3. Calibrar con un subconjunto de datos reales
  4. Convertir a INT8 con convert_fx
  5. Comparar: accuracy, latencia, tamaño en disco, potencia

Ejecutar en Jetson:
  python3 cuantizacion_ptq.py

Requiere en el mismo directorio:
  - mlp_student.pth          (MLP baseline entrenado)
  - mlp_kd_T4_a0.7.pth       (MLP con KD, mejor variante)
"""

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import prepare_fx, convert_fx, QConfigMapping
import numpy as np
import math
import time
import statistics
import os
import platform

# ── DEVICE: PTQ corre en CPU (la cuantización estática de PyTorch no soporta CUDA)
# Los modelos cuantizados a INT8 se ejecutan en CPU con qnnpack (backend ARM de Jetson)
DEVICE_TRAIN = torch.device("cuda" if torch.cuda.is_avaialble() else "cpu")  # Para generar datos
DEVICE_QUANT = torch.device("cpu")   # Para cuantización

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

# ── ARQUITECTURA MLP ──────────────────────────────────────────────────────────
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

# ── DETECCIÓN DE BACKEND (adaptado de Unai) ───────────────────────────────────
def get_backend():
    """
    Detecta la arquitectura del procesador y selecciona el backend correcto.
    En Jetson (aarch64/ARM) el backend correcto es qnnpack.
    En x86 sería fbgemm.
    """
    machine = platform.machine().lower()
    if machine in ['arm64', 'aarch64', 'arm']:
        backend = 'qnnpack'
    else:
        backend = 'fbgemm'
    print(f"Arquitectura detectada: {machine} → backend: {backend}")
    return backend

# ── EVALUACIÓN ────────────────────────────────────────────────────────────────
def accuracy(model, X, y, batch_size=256, device=DEVICE_QUANT):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(device)
            yb = y[i:i+batch_size].to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return correct / total

def medir_latencia(model, x_sample, n=100, warmup=10, device=DEVICE_QUANT):
    """
    Para modelos cuantizados en CPU no se usa cuda.synchronize().
    Se usa time.perf_counter() directamente porque no hay asincronía en CPU.
    """
    model.eval()
    x_sample = x_sample.to(device)
    
    with torch.no_grad():

        for _ in range(warmup):
            _ = model(x_sample)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        	
        lats = []
        for _ in range(n):
            if device.type == 'cuda':
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
		starter.record()
		_ = model(x_sample)
		ender.record()
		torch.cuda.synchronize()
		lats.append(starter.elapsed_time(ender))
	    else:
	        t0 = time.perf_counter()
		_ = model(x_sample)
		lats.append((time.perf_counter() - t0) * 1000)
		    
    return statistics.mean(lats), statistics.stdev(lats)

def tamanyo_modelo_mb(model):
    """Guarda el modelo en un buffer temporal y mide su tamaño en MB."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024**2

# ── CUANTIZACIÓN PTQ ──────────────────────────────────────────────────────────
def cuantizar_ptq(model_fp32, calibration_data, backend):
    """
    Post-Training Static Quantization usando la API moderna de PyTorch (fx graph mode).
    
    Paso 1: prepare_fx — inserta observers que miden rangos de activaciones
    Paso 2: calibración — pasar datos reales para que los observers aprendan los rangos
    Paso 3: convert_fx — sustituye capas float32 por versiones INT8
    
    El modelo debe estar en CPU para cuantización estática.
    """
    model_fp32.eval()
    model_fp32 = model_fp32.to(DEVICE_QUANT)

    # Configuración del backend
    torch.backends.quantized.engine = backend

    # QConfigMapping: aplica configuración por defecto para el backend elegido
    qconfig_mapping = QConfigMapping().set_global(
        tq.get_default_qconfig(backend)
    )

    # example_inputs: muestra de la forma de entrada — CRÍTICO para prepare_fx
    # Para el MLP: [1, 1024] (batch=1, señal plana)
    example_inputs = (torch.randn(1, 1024),)

    print("  Preparando modelo (insertando observers)...")
    model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs)

    print("  Calibrando (midiendo rangos de activaciones)...")
    with torch.no_grad():
        for i in range(0, len(calibration_data), 256):
            xb = calibration_data[i:i+256]
            model_prepared(xb)

    print("  Convirtiendo a INT8...")
    model_quantized = convert_fx(model_prepared)

    return model_quantized

# ── POTENCIA ──────────────────────────────────────────────────────────────────
def leer_potencia():
    curr_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"
    if os.path.exists(curr_path):
        mA = int(open(curr_path).read().strip())
        return mA / 1000 * 5.0
    return None

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    backend = get_backend()

    print("\nGenerando dataset...")
    X, y = generate_dataset(5000, 1024, 10)
    # Para cuantización todo en CPU
    X_cpu = X
    y_cpu = y

    # Datos de calibración: 500 muestras es suficiente para PTQ
    X_calib = X_cpu[:500]

    resultados = []

    for nombre, pth in [("MLP baseline", "mlp_student.pth"),
                         ("MLP + KD (T=4, a=0.7)", "mlp_kd_T4_a0.7.pth")]:

        print(f"\n{'='*60}")
        print(f"Procesando: {nombre}")
        print('='*60)

        # ── Modelo float32 original (referencia en CPU) ──
        model_fp32 = MLP()
        model_fp32.load_state_dict(
            torch.load(pth, map_location=DEVICE_QUANT, weights_only=True)
        )
        model_fp32.eval()
        
        if torch.cuda.is_available():
        	model_gpu = MLP().to(DEVICE_TRAIN)
        	model_gpu.load_state_dict(torch.load(pth, map_location=DEVICE_TRAIN, weights_only=True))
        	model_gpu.eval()
        	lat_gpu, std_gpu = medir_latencia(model_gpu, X_cpu[:1], device=DEVICE_TRAIN)
        	print(f"\nFP32 (GPU CUDA):")
        	print(f" Latencia: {lat_gpu:.3f} ms ± {std_gpu:.3f}")
        else: 
        	lat_gpu = 0.0

        acc_fp32  = accuracy(model_fp32, X_cpu, y_cpu)
        lat_fp32, std_fp32 = medir_latencia(model_fp32, X_cpu[:1])
        tam_fp32  = tamanyo_modelo_mb(model_fp32)

        print(f"\nFP32 (referencia CPU):")
        print(f"  Accuracy:  {acc_fp32*100:.1f}%")
        print(f"  Latencia:  {lat_fp32:.3f} ms ± {std_fp32:.3f}")
        print(f"  Tamaño:    {tam_fp32:.2f} MB")

        # ── Cuantización PTQ ──
        print(f"\nAplicando PTQ...")
        model_int8 = cuantizar_ptq(model_fp32, X_calib, backend)

        acc_int8  = accuracy(model_int8, X_cpu, y_cpu)
        lat_int8, std_int8 = medir_latencia(model_int8, X_cpu[:1])
        tam_int8  = tamanyo_modelo_mb(model_int8)

        pot = leer_potencia()

        print(f"\nINT8 (cuantizado):")
        print(f"  Accuracy:  {acc_int8*100:.1f}%")
        print(f"  Latencia:  {lat_int8:.3f} ms ± {std_int8:.3f}")
        print(f"  Tamaño:    {tam_int8:.2f} MB")
        if pot:
            print(f"  Potencia:  {pot:.2f} W")

        # Guardar modelo cuantizado
        nombre_archivo = f"mlp_int8_{nombre.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '').replace('.', '')}.pth"
        torch.save(model_int8.state_dict(), nombre_archivo)
        print(f"  Guardado:  {nombre_archivo}")

        resultados.append({
            "nombre": nombre,
            "acc_fp32": acc_fp32, "lat_fp32": lat_fp32, "tam_fp32": tam_fp32,
            "acc_int8": acc_int8, "lat_int8": lat_int8, "tam_int8": tam_int8,
        })

    # ── Tabla resumen ──
    print(f"\n{'='*70}")
    print("RESUMEN CUANTIZACIÓN PTQ")
    print('='*70)
    print(f"{'Modelo':<25} {'Precision':>10} {'Accuracy':>10} {'Latencia':>12} {'Tamaño':>10}")
    print('-'*70)
    for r in resultados:
        print(f"{r['nombre']:<25} {'FP32':>10} {r['acc_fp32']*100:>9.1f}% {r['lat_fp32']:>10.3f}ms {r['tam_fp32']:>8.2f}MB")
        perdida = (r['acc_fp32'] - r['acc_int8']) * 100
        aceleracion = r['lat_fp32'] / r['lat_int8'] if r['lat_int8'] > 0 else 0
        reduccion   = r['tam_fp32'] / r['tam_int8'] if r['tam_int8'] > 0 else 0
        print(f"{'':<25} {'INT8':>10} {r['acc_int8']*100:>9.1f}% {r['lat_int8']:>10.3f}ms {r['tam_int8']:>8.2f}MB")
        print(f"  → Pérdida accuracy: {perdida:.2f}pp | Aceleración: ×{aceleracion:.1f} | Reducción tamaño: ×{reduccion:.1f}")
        print()
    print('='*70)
