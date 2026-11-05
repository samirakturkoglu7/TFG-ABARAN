#!/bin/bash
# perfil.sh - Aplica un perfil de hardware limitado a la Jetson Orin Nano
# Uso: sudo ./perfil.sh full|medio|bajo|minimo|estado

PERFIL="${1:-estado}"  # Perfil pedido (por defecto solo muestra el estado)
SOLO_FREQ="${2:-}"  # Si vale 'solofreq' no se apagan nucleos (los limitara Docker con --cpuset-cpus)
CPU=/sys/devices/system/cpu  # Carpeta sysfs de la CPU
GPU=/sys/class/devfreq/17000000.gpu  # Carpeta sysfs de la GPU
EMC=/sys/kernel/nvpmodel_clk_cap/emc  # Nodo para limitar la frecuencia de la memoria (EMC)

if [ "$EUID" -ne 0 ]; then  # Comprueba que se ejecuta como root
  echo "Ejecuta con sudo"  # Aviso al usuario
  exit 1  # Sale con error
fi  # Fin de la comprobacion

escribir() {  # Funcion: escribe un valor ($2) en un archivo sysfs ($1)
  echo "$2" > "$1" 2>/dev/null || echo "AVISO: no se pudo escribir $2 en $1"  # Escribe y avisa si falla
}  # Fin de la funcion

estado() {  # Funcion: muestra el estado real del hardware
  nvpmodel -q 2>/dev/null | tail -n 2  # Modo de potencia activo
  echo "CPUs online: $(cat $CPU/online)"  # Nucleos activos
  echo "CPU0 kHz: $(cat $CPU/cpu0/cpufreq/scaling_cur_freq) (max $(cat $CPU/cpu0/cpufreq/scaling_max_freq))"  # Frecuencia de CPU
  echo "GPU Hz: $(cat $GPU/cur_freq) (min $(cat $GPU/min_freq) / max $(cat $GPU/max_freq))"  # Frecuencia de GPU
  jetson_clocks --show 2>/dev/null | grep -E "Active GPU TPCs|EMC"  # TPC de GPU activos y estado de la memoria
}  # Fin de la funcion

if [ "$PERFIL" = "estado" ]; then  # Si solo se pide el estado
  estado  # Lo muestra
  exit 0  # Y termina sin cambiar nada
fi  # Fin del if

case "$PERFIL" in  # Elige los valores segun el perfil
  full)   MODO=2; NUCLEOS=6; CPU_KHZ=1728000; GPU_HZ=1020000000; EMC_HZ=-1 ;;  # Maximo: MAXN SUPER, 6 nucleos, sin limites extra
  medio)  MODO=0; NUCLEOS=6; CPU_KHZ=1497600; GPU_HZ=612000000;  EMC_HZ=-1 ;;  # Modo 15W: 6 nucleos, CPU 1.5 GHz, GPU 612 MHz
  bajo)   MODO=3; NUCLEOS=4; CPU_KHZ=960000;  GPU_HZ=408000000;  EMC_HZ=-1 ;;  # Modo 7W: 4 nucleos, CPU 960 MHz, GPU 408 MHz (tipo Jetson Nano)
  minimo) MODO=3; NUCLEOS=2; CPU_KHZ=729600;  GPU_HZ=306000000;  EMC_HZ=665600000 ;;  # Extremo: 2 nucleos, GPU al minimo, memoria a 665 MHz
  *) echo "Perfil desconocido: $PERFIL"; exit 1 ;;  # Cualquier otro nombre es un error
esac  # Fin del case

nvpmodel -m "$MODO"  # Aplica el modo de potencia base (tambien resetea limites anteriores)
sleep 2  # Espera a que el kernel aplique el modo

for i in 1 2 3 4 5; do  # Recorre los nucleos 1 a 5 (el 0 no se puede apagar)
  if [ "$i" -lt "$NUCLEOS" ]; then  # Si el nucleo debe quedar activo
    escribir $CPU/cpu$i/online 1  # Lo enciende
  else  # Si sobra
    escribir $CPU/cpu$i/online 0  # Lo apaga
  fi  # Fin del if
done  # Fin del bucle

FMIN=$(cat $CPU/cpu0/cpufreq/cpuinfo_min_freq)  # Frecuencia minima que admite la CPU
for c in $CPU/cpu[0-5]; do  # Recorre cada nucleo
  [ -f "$c/cpufreq/scaling_max_freq" ] || continue  # Salta los nucleos apagados
  escribir $c/cpufreq/scaling_min_freq "$FMIN"  # Baja el minimo primero (evita error al bajar el maximo)
  escribir $c/cpufreq/scaling_max_freq "$CPU_KHZ"  # Fija el techo de frecuencia
  escribir $c/cpufreq/scaling_min_freq "$CPU_KHZ"  # Iguala el minimo al techo: frecuencia fija
done  # Fin del bucle

GMIN=$(tr ' ' '\n' < $GPU/available_frequencies | grep -E '^[0-9]+$' | sort -n | head -1)  # Frecuencia minima de la GPU
escribir $GPU/min_freq "$GMIN"  # Baja el minimo primero
escribir $GPU/max_freq "$GPU_HZ"  # Fija el techo de la GPU
escribir $GPU/min_freq "$GPU_HZ"  # Iguala el minimo al techo: frecuencia fija

if [ "$EMC_HZ" != "-1" ]; then  # Si el perfil limita la memoria
  escribir $EMC "$EMC_HZ"  # Aplica el limite de frecuencia a la RAM
fi  # Fin del if

if [ "$PERFIL" = "full" ]; then  # Solo en el perfil maximo
  jetson_clocks  # Bloquea todas las frecuencias al maximo
fi  # Fin del if

sleep 2  # Deja estabilizar las frecuencias
echo "=== Perfil aplicado: $PERFIL ==="  # Confirmacion
estado  # Muestra el estado real para verificar
