# comparar_resultados.py - Tabla comparativa de todos los perfiles ejecutados
import glob  # Busca los archivos de resultados
import json  # Lee los .json

ORDEN = ["full", "medio", "bajo", "minimo"]  # Orden de presentacion (el resto va al final)


def fmt(v, d=2):  # Formatea numero o 'n/d'
    return "n/d" if v is None else f"{v:.{d}f}"  # Texto con d decimales


datos = []  # Lista de resultados cargados
for ruta in glob.glob("resultados/*.json"):  # Recorre los archivos
    with open(ruta) as f:  # Abre cada archivo
        datos.append(json.load(f))  # Lo carga
datos.sort(key=lambda r: ORDEN.index(r["perfil"]) if r["perfil"] in ORDEN else 99)  # Ordena por perfil

print("\nTIEMPO, ENERGIA Y MEMORIA (entrenamiento)")  # Titulo tabla 1
print(f"{'perfil':<9}{'t_teacher s':>12}{'t_student s':>12}{'ptq s':>8}{'W medio':>9}{'W pico':>8}{'J total':>9}{'VRAM MB':>9}{'RAM MB':>8}")  # Cabecera
for r in datos:  # Una fila por perfil
    e = r["etapas"]  # Resultados por etapa
    t = e.get("teacher_train", {})  # Etapa teacher
    s = e.get("student_kd_train", {})  # Etapa student
    q = e.get("ptq_int8", {})  # Etapa PTQ
    julios = sum(x.get("julios") or 0 for x in (t, s, q))  # Energia total de las tres etapas
    wm = s.get("w_medio")  # Potencia media durante el student
    wp = max([x.get("w_pico") or 0 for x in (t, s, q)] or [0])  # Pico de potencia de todo el proceso
    vram = max([x.get("vram_pico_mb") or 0 for x in (t, s, q)] or [0])  # Pico de VRAM
    ram = max([x.get("ram_pico_mb") or 0 for x in (t, s, q)] or [0])  # Pico de RAM
    print(f"{r['perfil']:<9}{fmt(t.get('segundos'), 1):>12}{fmt(s.get('segundos'), 1):>12}{fmt(q.get('segundos'), 1):>8}{fmt(wm):>9}{fmt(wp):>8}{julios:>9.0f}{vram:>9.0f}{ram:>8.0f}")  # Fila

print("\nPRECISION Y LATENCIA (batch=1)")  # Titulo tabla 2
print(f"{'perfil':<9}{'acc T':>7}{'acc S':>7}{'acc Q':>7}{'lat T gpu':>10}{'lat S gpu':>10}{'lat S cpu':>10}{'lat Q cpu':>10}{'MB S':>7}{'MB Q':>7}")  # Cabecera
for r in datos:  # Una fila por perfil
    if r.get("error"):  # Si el perfil fallo
        print(f"{r['perfil']:<9}FALLO: {r['error'][:90]}")  # Muestra el error
        continue  # Pasa al siguiente
    a, l, m = r["accuracy"], r["latencia"], r["tamano_mb"]  # Atajos a los bloques
    print(f"{r['perfil']:<9}{a['teacher']:>7.2f}{a['student']:>7.2f}{a['student_int8']:>7.2f}{l['teacher_gpu']['ms_medio']:>10.2f}{l['student_gpu']['ms_medio']:>10.2f}{l['student_cpu']['ms_medio']:>10.2f}{l['student_int8_cpu']['ms_medio']:>10.2f}{m['student']:>7.2f}{m['student_int8']:>7.2f}")  # Fila

print("\nHARDWARE APLICADO (verifica que coincide con el perfil)")  # Titulo tabla 3
for r in datos:  # Una fila por perfil
    print(r["perfil"], r.get("hw"), "| cgroup:", (r.get("cgroup_events") or "n/d").replace("\n", " "))  # Estado y contadores OOM
