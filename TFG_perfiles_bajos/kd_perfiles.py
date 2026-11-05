# kd_perfiles.py - Pipeline Teacher -> KD Student -> PTQ INT8, medido bajo un perfil de hardware
# Uso: python3 kd_perfiles.py --perfil bajo --gpu-frac 0.5
import argparse  # Lee argumentos de la linea de comandos
import copy  # Copia modelos sin modificar el original
import glob  # Busca archivos por patron (ruta del sensor)
import io  # Buffer en memoria para medir el tamano del modelo
import json  # Guarda los resultados en .json
import math  # Constante pi para generar senales
import os  # Rutas y archivos del sistema
import resource  # Mide el pico de RAM del proceso
import threading  # Hilo en segundo plano para leer la potencia
import time  # Cronometros
import numpy as np  # Generacion de senales y estadisticas
import torch  # Framework principal
import torch.nn as nn  # Capas de redes neuronales
import torch.nn.functional as F  # Funciones (softmax, loss)
import torch.optim as optim  # Optimizadores
import torch.ao.quantization as tq  # Herramientas de cuantizacion
from torch.ao.quantization import QConfigMapping  # Configuracion de cuantizacion por capa
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx  # Pasos de PTQ
from torch.utils.data import DataLoader, TensorDataset  # Para trabajar por batches

p = argparse.ArgumentParser()  # Crea el lector de argumentos
p.add_argument("--perfil", default="full")  # Nombre del perfil (etiqueta de los resultados)
p.add_argument("--gpu-frac", type=float, default=1.0)  # Fraccion de memoria GPU que PyTorch puede usar (0-1)
p.add_argument("--batch", type=int, default=64)  # Tamano de batch de entrenamiento
p.add_argument("--epochs-t", type=int, default=5)  # Epocas del teacher
p.add_argument("--epochs-s", type=int, default=10)  # Epocas del student con KD
args = p.parse_args()  # Lee los argumentos

torch.manual_seed(0)  # Semilla de PyTorch: mismos pesos iniciales en todos los perfiles
np.random.seed(0)  # Semilla de NumPy: mismo dataset en todos los perfiles
DEVICE = torch.device("cuda")  # El entrenamiento va en GPU
DEVICE_Q = torch.device("cpu")  # El modelo INT8 corre en CPU
torch.backends.quantized.engine = "qnnpack"  # Backend de cuantizacion para ARM
NUCLEOS = len(os.sched_getaffinity(0))  # Nucleos que el sistema deja usar a este proceso
torch.set_num_threads(NUCLEOS)  # PyTorch usa solo esos nucleos en CPU
if args.gpu_frac < 1.0:  # Si se pide limitar la memoria de GPU
    torch.cuda.set_per_process_memory_fraction(args.gpu_frac, 0)  # Limita el allocator de PyTorch en la GPU 0

RES = {"perfil": args.perfil, "args": vars(args), "etapas": {}, "error": None}  # Diccionario con todos los resultados


def fmt(v, d=2):  # Formatea un numero o devuelve 'n/d' si no hay dato
    return "n/d" if v is None else f"{v:.{d}f}"  # Texto con d decimales


def leer_archivo(ruta):  # Lee un archivo sysfs
    try:  # Intenta leer
        with open(ruta) as f:  # Abre el archivo
            return f.read().strip()  # Devuelve el texto sin espacios
    except Exception:  # Si no existe o falla
        return None  # Devuelve None


def buscar_hwmon():  # Localiza la carpeta del sensor INA3221 (hwmon0/hwmon1 cambia entre arranques)
    r = glob.glob("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*")  # Busca las carpetas hwmon
    return r[0] if r else None  # Devuelve la primera o None


HWMON = buscar_hwmon()  # Ruta del sensor
V_RIEL = 5.0  # Tension del riel asumida (V), igual que en las sesiones anteriores


def potencia_w():  # Potencia instantanea en vatios
    if HWMON is None:  # Si no hay sensor
        return None  # No se puede medir
    mA = leer_archivo(f"{HWMON}/curr1_input")  # Corriente del canal 1 en mA
    return None if mA is None else float(mA) / 1000.0 * V_RIEL  # P = I * V


class MonitorPotencia(threading.Thread):  # Hilo que muestrea la potencia mientras se ejecuta una etapa
    def __init__(self, intervalo=0.2):  # Intervalo entre lecturas en segundos
        super().__init__(daemon=True)  # Hilo daemon: no bloquea la salida del programa
        self.intervalo = intervalo  # Guarda el intervalo
        self.muestras = []  # Lista de (tiempo, vatios)
        self.parar = threading.Event()  # Senal para detener el hilo

    def run(self):  # Cuerpo del hilo
        while not self.parar.is_set():  # Mientras no se pida parar
            w = potencia_w()  # Lee la potencia
            if w is not None:  # Si hay lectura valida
                self.muestras.append((time.perf_counter(), w))  # Guarda la muestra
            time.sleep(self.intervalo)  # Espera hasta la siguiente lectura

    def resumen(self):  # Estadisticas de la etapa
        if len(self.muestras) < 2:  # Si hay muy pocas muestras
            return {"w_medio": None, "w_pico": None, "julios": None}  # No se puede resumir
        ws = [w for _, w in self.muestras]  # Solo los vatios
        dur = self.muestras[-1][0] - self.muestras[0][0]  # Duracion cubierta por las muestras
        medio = sum(ws) / len(ws)  # Potencia media
        return {"w_medio": medio, "w_pico": max(ws), "julios": medio * dur}  # Energia = potencia media * tiempo


def medir_etapa(nombre, fn):  # Ejecuta fn() midiendo tiempo, potencia y memoria
    torch.cuda.reset_peak_memory_stats()  # Reinicia el pico de VRAM
    mon = MonitorPotencia()  # Crea el monitor de potencia
    mon.start()  # Lo arranca en segundo plano
    torch.cuda.synchronize()  # Espera a que la GPU este libre
    t0 = time.perf_counter()  # Inicio del cronometro
    salida = fn()  # Ejecuta la etapa
    torch.cuda.synchronize()  # Espera a que la GPU termine antes de parar el cronometro
    seg = time.perf_counter() - t0  # Duracion en segundos
    mon.parar.set()  # Pide parar al monitor
    mon.join()  # Espera a que termine el hilo
    r = mon.resumen()  # Estadisticas de potencia
    r["segundos"] = seg  # Anade la duracion
    r["vram_pico_mb"] = torch.cuda.max_memory_allocated() / 1024**2  # Pico de VRAM en MB
    r["ram_pico_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Pico de RAM del proceso en MB
    RES["etapas"][nombre] = r  # Guarda en los resultados
    print(f"[{nombre}] {seg:.1f} s | potencia media {fmt(r['w_medio'])} W | pico {fmt(r['w_pico'])} W | VRAM pico {r['vram_pico_mb']:.0f} MB | RAM pico {r['ram_pico_mb']:.0f} MB")  # Resumen en pantalla
    return salida  # Devuelve lo que devolvio la etapa


def ruta_cgroup():  # Carpeta cgroup v2 de este proceso
    try:  # Intenta leer
        for linea in open("/proc/self/cgroup").read().splitlines():  # Recorre las lineas
            if linea.startswith("0::"):  # Linea de cgroup v2
                return "/sys/fs/cgroup" + linea[3:]  # Construye la ruta completa
    except Exception:  # Si falla
        pass  # Ignora el error
    return None  # No se encontro


def estado_hw():  # Foto del hardware realmente aplicado (para verificar el perfil)
    cg = ruta_cgroup()  # Ruta del cgroup
    return {  # Devuelve el diccionario
        "nucleos_proceso": NUCLEOS,  # Nucleos disponibles para este proceso
        "cpu0_khz": leer_archivo("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),  # Frecuencia de CPU0
        "gpu_hz": leer_archivo("/sys/class/devfreq/17000000.gpu/cur_freq"),  # Frecuencia de GPU
        "gpu_max_hz": leer_archivo("/sys/class/devfreq/17000000.gpu/max_freq"),  # Techo de GPU
        "ram_cgroup_max": leer_archivo(cg + "/memory.max") if cg else None,  # Limite de RAM ('max' = sin limite)
        "voltaje_riel_mV": leer_archivo(f"{HWMON}/in1_input") if HWMON else None,  # Tension medida (para comprobar los 5 V)
    }  # Fin del diccionario


def generar_dataset(n, tam=1024, clases=10):  # Genera n senales: 2 frecuencias solapadas + ruido fuerte
    t = np.linspace(0, 1, tam, dtype=np.float32)  # Eje de tiempo de 1 s
    X = np.zeros((n, 1, tam), dtype=np.float32)  # Matriz [muestras, canal, longitud]
    y = np.zeros(n, dtype=np.int64)  # Etiquetas
    for i in range(n):  # Una senal por iteracion
        c = np.random.randint(0, clases)  # Clase aleatoria
        f1 = 20.0 + c * 8.0 + np.random.uniform(-3.0, 3.0)  # Frecuencia principal de la clase con variacion
        f2 = 20.0 + ((c + 1) % clases) * 8.0 + np.random.uniform(-2.0, 2.0)  # Frecuencia de la clase vecina (interferencia)
        a1 = np.random.uniform(0.8, 1.2)  # Amplitud principal
        a2 = np.random.uniform(0.5, 0.9)  # Amplitud de la interferencia
        p1 = np.random.uniform(0, 2 * math.pi)  # Fase de la senal principal
        p2 = np.random.uniform(0, 2 * math.pi)  # Fase de la interferencia
        s = a1 * np.sin(2 * math.pi * f1 * t + p1) + a2 * np.sin(2 * math.pi * f2 * t + p2)  # Suma de las dos senoidales
        s = s + np.random.normal(0, 0.8, tam).astype(np.float32)  # Ruido gaussiano fuerte
        X[i, 0, :] = s  # Guarda la senal
        y[i] = c  # Guarda la etiqueta
    return torch.from_numpy(X), torch.from_numpy(y)  # Convierte a tensores


def cargar_dataset():  # Carga el dataset de disco o lo genera una sola vez
    ruta = "dataset_cache.pt"  # Archivo de cache
    if os.path.exists(ruta):  # Si ya existe
        return torch.load(ruta, weights_only=True)  # Lo carga: mismos datos en todos los perfiles
    Xtr, ytr = generar_dataset(6000)  # 6000 muestras de entrenamiento
    Xte, yte = generar_dataset(2000)  # 2000 de test (mas que antes: reduce el ruido en accuracy)
    d = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}  # Agrupa los tensores
    torch.save(d, ruta)  # Guarda la cache
    return d  # Devuelve el dataset


class ResidualBlock1D(nn.Module):  # Bloque residual: dos convoluciones + atajo
    def __init__(self, c_in, c_out, stride=1):  # Canales de entrada/salida y paso
        super().__init__()  # Inicializa nn.Module
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)  # Primera convolucion
        self.bn1 = nn.BatchNorm1d(c_out)  # Normalizacion tras conv1
        self.relu = nn.ReLU(inplace=True)  # Activacion ReLU
        self.conv2 = nn.Conv1d(c_out, c_out, 3, stride=1, padding=1, bias=False)  # Segunda convolucion
        self.bn2 = nn.BatchNorm1d(c_out)  # Normalizacion tras conv2
        self.shortcut = nn.Sequential()  # Atajo identidad por defecto
        if stride != 1 or c_in != c_out:  # Si cambian las dimensiones
            self.shortcut = nn.Sequential(  # Atajo con convolucion 1x1 para igualar dimensiones
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),  # Convolucion 1x1
                nn.BatchNorm1d(c_out),  # Normalizacion del atajo
            )  # Fin del atajo

    def forward(self, x):  # Paso hacia delante
        out = self.relu(self.bn1(self.conv1(x)))  # conv1 -> bn1 -> relu
        out = self.bn2(self.conv2(out))  # conv2 -> bn2
        return self.relu(out + self.shortcut(x))  # Suma residual y ReLU


class ResNet1D(nn.Module):  # ResNet para senales 1D
    def __init__(self, num_classes=10, base_channels=16):  # base_channels controla el tamano del modelo
        super().__init__()  # Inicializa nn.Module
        self.in_channels = base_channels  # Canales actuales (los usa _make_layer)
        self.conv1 = nn.Conv1d(1, base_channels, 7, stride=2, padding=3, bias=False)  # Convolucion inicial
        self.bn1 = nn.BatchNorm1d(base_channels)  # Normalizacion inicial
        self.relu = nn.ReLU(inplace=True)  # Activacion
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)  # Reduce la longitud a la mitad
        self.layer1 = self._make_layer(base_channels, 2, 1)  # Etapa 1: 2 bloques
        self.layer2 = self._make_layer(base_channels * 2, 2, 2)  # Etapa 2: 2 bloques, duplica canales
        self.layer3 = self._make_layer(base_channels * 4, 2, 2)  # Etapa 3: 2 bloques, duplica canales
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Promedio global sobre la longitud
        self.fc = nn.Linear(base_channels * 4, num_classes)  # Capa final de clasificacion

    def _make_layer(self, c_out, blocks, stride):  # Construye una etapa de bloques
        layers = [ResidualBlock1D(self.in_channels, c_out, stride)]  # Primer bloque (puede cambiar dimensiones)
        self.in_channels = c_out  # Actualiza los canales actuales
        for _ in range(1, blocks):  # Bloques restantes
            layers.append(ResidualBlock1D(c_out, c_out))  # Bloques sin cambio de dimensiones
        return nn.Sequential(*layers)  # Agrupa los bloques

    def forward(self, x):  # Paso hacia delante
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Entrada: conv -> bn -> relu -> pool
        x = self.layer3(self.layer2(self.layer1(x)))  # Las tres etapas residuales
        return self.fc(torch.flatten(self.avgpool(x), 1))  # Promedio global -> logits


def kd_loss(s_logits, t_logits, labels, T=4.0, alpha=0.7):  # Loss de Knowledge Distillation
    soft_t = F.softmax(t_logits / T, dim=1)  # Probabilidades suavizadas del teacher
    log_s = F.log_softmax(s_logits / T, dim=1)  # Log-probabilidades suavizadas del student
    dist = F.kl_div(log_s, soft_t, reduction="batchmean") * (T ** 2)  # Distancia teacher-student (escalada por T^2)
    ce = F.cross_entropy(s_logits, labels)  # Error frente a las etiquetas reales
    return alpha * dist + (1.0 - alpha) * ce  # Mezcla de ambos terminos


def entrenar_teacher(teacher, loader, epochs):  # Entrena el modelo grande
    opt = optim.Adam(teacher.parameters(), lr=1e-3)  # Optimizador Adam
    teacher.train()  # Modo entrenamiento
    for ep in range(epochs):  # Una vuelta por epoca
        for bx, by in loader:  # Un batch por iteracion
            bx, by = bx.to(DEVICE), by.to(DEVICE)  # Mueve el batch a la GPU
            opt.zero_grad()  # Borra gradientes anteriores
            loss = F.cross_entropy(teacher(bx), by)  # Forward + loss
            loss.backward()  # Calcula gradientes
            opt.step()  # Actualiza pesos
        print(f"  teacher epoca {ep + 1}/{epochs} loss={loss.item():.3f}")  # Progreso
    teacher.eval()  # Pasa a modo evaluacion


def entrenar_student(student, teacher, loader, epochs):  # Entrena el modelo pequeno con KD
    opt = optim.Adam(student.parameters(), lr=1e-3)  # Optimizador Adam
    student.train()  # Modo entrenamiento
    for ep in range(epochs):  # Una vuelta por epoca
        for bx, by in loader:  # Un batch por iteracion
            bx, by = bx.to(DEVICE), by.to(DEVICE)  # Mueve el batch a la GPU
            opt.zero_grad()  # Borra gradientes anteriores
            with torch.no_grad():  # El teacher no se entrena
                t_logits = teacher(bx)  # Salida del teacher
            loss = kd_loss(student(bx), t_logits, by)  # Loss combinada KD + CrossEntropy
            loss.backward()  # Calcula gradientes del student
            opt.step()  # Actualiza pesos del student
        print(f"  student epoca {ep + 1}/{epochs} loss={loss.item():.3f}")  # Progreso
    student.eval()  # Pasa a modo evaluacion


def cuantizar(student, calib):  # PTQ INT8 del student
    m = copy.deepcopy(student).to(DEVICE_Q).eval()  # Copia en CPU y modo evaluacion (el original no se toca)
    qc = QConfigMapping().set_global(tq.get_default_qconfig("qnnpack"))  # Configuracion global para ARM
    ejemplo = (torch.randn(1, 1, 1024),)  # Forma de entrada para trazar el grafo
    prep = prepare_fx(m, qc, ejemplo)  # Inserta observadores
    with torch.no_grad():  # Sin gradientes durante la calibracion
        for i in range(0, len(calib), 100):  # Calibra en batches de 100
            prep(calib[i:i + 100])  # Los observadores miden los rangos
    return convert_fx(prep)  # Convierte a INT8


def precision(m, dev, X, y):  # Accuracy en % sobre el conjunto dado
    m = m.to(dev).eval()  # Mueve el modelo al dispositivo
    ok = 0  # Aciertos acumulados
    with torch.no_grad():  # Sin gradientes
        for i in range(0, len(X), 64):  # Batches de 64
            bx = X[i:i + 64].to(dev)  # Batch de entrada
            by = y[i:i + 64].to(dev)  # Etiquetas del batch
            ok += (m(bx).argmax(dim=1) == by).sum().item()  # Cuenta aciertos
    return 100.0 * ok / len(X)  # Porcentaje


def tamano_mb(m):  # Tamano del state_dict en MB
    buf = io.BytesIO()  # Buffer en memoria
    torch.save(m.state_dict(), buf)  # Guarda en el buffer
    return buf.tell() / 1024**2  # Bytes a MB


def latencia(m, dev, runs=100):  # Latencia con batch=1
    m = m.to(dev).eval()  # Mueve el modelo al dispositivo
    x = torch.randn(1, 1, 1024, device=dev)  # Una sola muestra
    lat = []  # Latencias medidas
    with torch.no_grad():  # Sin gradientes
        for _ in range(20):  # Calentamiento
            m(x)  # Ejecuta sin medir
        for _ in range(runs):  # Mediciones
            if dev.type == "cuda":  # Si es GPU
                torch.cuda.synchronize()  # Espera antes de empezar
            t0 = time.perf_counter()  # Inicio
            m(x)  # Inferencia
            if dev.type == "cuda":  # Si es GPU
                torch.cuda.synchronize()  # Espera a que termine antes de parar el cronometro
            lat.append((time.perf_counter() - t0) * 1000.0)  # Guarda ms
    lat.sort()  # Ordena para el percentil
    return {"ms_medio": float(np.mean(lat)), "ms_std": float(np.std(lat)), "ms_p95": lat[int(0.95 * len(lat)) - 1]}  # Estadisticas


def guardar():  # Escribe los resultados en resultados/<perfil>.json
    os.makedirs("resultados", exist_ok=True)  # Crea la carpeta si no existe
    cg = ruta_cgroup()  # Ruta del cgroup
    RES["cgroup_events"] = leer_archivo(cg + "/memory.events") if cg else None  # Contadores de OOM del cgroup
    with open(f"resultados/{args.perfil}.json", "w") as f:  # Abre el archivo de salida
        json.dump(RES, f, indent=2)  # Escribe el JSON


def main():  # Flujo completo
    RES["hw"] = estado_hw()  # Registra el hardware aplicado
    print(f"=== PERFIL {args.perfil} === {RES['hw']}")  # Lo muestra para verificar
    d = cargar_dataset()  # Carga el dataset fijo
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]  # Separa train y test
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)  # Loader de entrenamiento
    teacher = ResNet1D(base_channels=32).to(DEVICE)  # Teacher: 32 canales base
    medir_etapa("teacher_train", lambda: entrenar_teacher(teacher, loader, args.epochs_t))  # Entrena y mide
    student = ResNet1D(base_channels=16).to(DEVICE)  # Student: 16 canales base
    medir_etapa("student_kd_train", lambda: entrenar_student(student, teacher, loader, args.epochs_s))  # Entrena con KD y mide
    q = medir_etapa("ptq_int8", lambda: cuantizar(student, Xtr[:500]))  # Cuantiza con datos de train (no de test)
    RES["accuracy"] = {  # Precision de cada modelo
        "teacher": precision(teacher, DEVICE, Xte, yte),  # Teacher en GPU
        "student": precision(student, DEVICE, Xte, yte),  # Student FP32 en GPU
        "student_int8": precision(q, DEVICE_Q, Xte, yte),  # Student INT8 en CPU
    }  # Fin del diccionario
    RES["tamano_mb"] = {  # Tamano de cada modelo
        "teacher": tamano_mb(teacher),  # Teacher
        "student": tamano_mb(student),  # Student FP32
        "student_int8": tamano_mb(q),  # Student INT8
    }  # Fin del diccionario
    s_cpu = copy.deepcopy(student).to(DEVICE_Q)  # Copia del student en CPU para comparar FP32 vs INT8 en el mismo dispositivo
    RES["latencia"] = {  # Latencias con batch=1
        "teacher_gpu": latencia(teacher, DEVICE),  # Teacher en GPU
        "student_gpu": latencia(student, DEVICE),  # Student FP32 en GPU
        "student_cpu": latencia(s_cpu, DEVICE_Q),  # Student FP32 en CPU
        "student_int8_cpu": latencia(q, DEVICE_Q),  # Student INT8 en CPU
    }  # Fin del diccionario
    try:  # Exportar es opcional
        torch.jit.script(q).save(f"student_int8_{args.perfil}.pt")  # Guarda el modelo INT8 para despliegue
    except Exception as e:  # Si falla
        print("Aviso: no se pudo exportar TorchScript:", str(e)[:100])  # No detiene el experimento
    guardar()  # Guarda los resultados
    print(json.dumps({k: RES[k] for k in ("accuracy", "tamano_mb", "latencia")}, indent=1))  # Resumen en pantalla


if __name__ == "__main__":  # Punto de entrada
    try:  # Captura errores de memoria
        main()  # Ejecuta el pipeline
    except (RuntimeError, MemoryError) as e:  # OOM de CUDA, NvMap error 12, etc.
        RES["error"] = str(e)[:300]  # Guarda el motivo
        guardar()  # Guarda lo medido hasta el fallo
        print("ERROR (posible OOM):", RES["error"])  # Lo muestra
        raise SystemExit(2)  # Sale con codigo de error
