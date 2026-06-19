import time

CURR_PATH = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"
VOLTAJE = 5.0  # Voltios

print("Leyendo potencia cada segundo (Ctrl+C para parar)...\n")
try:
    while True:
        mA = int(open(CURR_PATH).read().strip())
        W = (mA / 1000.0) * VOLTAJE
        print(f"Corriente: {mA} mA | Potencia: {W:.2f} W")
        time.sleep(1)
except KeyboardInterrupt:
    print("Parado.")
