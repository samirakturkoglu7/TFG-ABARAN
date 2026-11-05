#!/bin/bash
# ejecutar_perfiles.sh - Ejecuta el pipeline en cada perfil y restaura el hardware al final
# Uso: ./ejecutar_perfiles.sh   (sin sudo; lo pedira al llamar a perfil.sh y systemd-run)

cd "$(dirname "$0")"  # Va a la carpeta donde estan los scripts
mkdir -p resultados  # Carpeta de salida de resultados
PY="$(command -v python3)"  # Ruta completa de python3 (systemd-run no usa tu PATH)

declare -A RAM=( [full]="" [medio]="6G" [bajo]="4G" [minimo]="2G" )  # Limite de RAM del proceso por perfil (vacio = sin limite)
declare -A GPUF=( [full]="1.0" [medio]="0.75" [bajo]="0.5" [minimo]="0.25" )  # Fraccion de memoria GPU permitida a PyTorch

for P in full medio bajo minimo; do  # Recorre los perfiles de mas a menos recursos
  echo "=================== PERFIL: $P ==================="  # Cabecera
  rm -f "resultados/$P.json"  # Borra resultados antiguos de este perfil
  sudo ./perfil.sh "$P"  # Aplica el perfil de hardware
  sleep 5  # Deja estabilizar frecuencias y temperatura
  if [ -n "${RAM[$P]}" ]; then  # Si el perfil tiene limite de RAM
    sudo systemd-run --scope --quiet -p MemoryMax="${RAM[$P]}" -p MemorySwapMax=0 --uid="$(id -u)" --gid="$(id -g)" --setenv=HOME="$HOME" "$PY" kd_perfiles.py --perfil "$P" --gpu-frac "${GPUF[$P]}"  # Ejecuta dentro de un cgroup con RAM limitada y sin swap
  else  # Si no tiene limite
    "$PY" kd_perfiles.py --perfil "$P" --gpu-frac "${GPUF[$P]}"  # Ejecuta normal
  fi  # Fin del if
  [ -f "resultados/$P.json" ] || echo "AVISO: $P no genero resultados (posible OOM kill del cgroup)"  # Detecta ejecuciones abortadas
  sleep 60  # Enfria la placa entre perfiles
done  # Fin del bucle

sudo ./perfil.sh full  # Restaura el perfil maximo
echo "Terminado. Ejecuta: python3 comparar_resultados.py"  # Siguiente paso
