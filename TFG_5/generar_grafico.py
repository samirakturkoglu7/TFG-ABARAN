import matplotlib.pyplot as plt

modelos = ['Teacher\n(FP32 GPU)', 'Student KD\n(FP32 GPU)', 'Student KD\n(INT8 CPU)']
tamanios = [0.96, 0.27, 0.08]
latencias = [5.267, 5.098, 3.487]

fig, ax1 = plt.subplots(figsize=(8, 4.5))

# Barras para el tamaño
color_bar = '#2b5c8f'
bars = ax1.bar(modelos, tamanios, color=color_bar, alpha=0.7, width=0.4, label='Tamaño (MB)')
ax1.set_ylabel('Tamaño en Disco / RAM (MB)', color=color_bar, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_bar)
ax1.set_ylim(0, 1.2)

# Valores sobre las barras
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f} MB',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Línea para la latencia
ax2 = ax1.twinx()
color_line = '#d9534f'
line = ax2.plot(modelos, latencias, color=color_line, marker='o', linewidth=2.5, markersize=8, label='Latencia (ms)')
ax2.set_ylabel('Latencia de Inferencia (ms)', color=color_line, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_line)
ax2.set_ylim(0, 7)

# Valores sobre los puntos de latencia
for i, txt in enumerate(latencias):
    ax2.annotate(f'{txt:.3f} ms', (modelos[i], latencias[i]), xytext=(0, 8),
                 textcoords="offset points", ha='center', fontsize=9, color=color_line, fontweight='bold')

plt.title('Impacto de Knowledge Distillation y PTQ INT8 en Jetson Orin Nano', fontsize=11, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('grafico_resultados_kd_ptq.png', dpi=300)
print("Gráfico generado exitosamente: grafico_resultados_kd_ptq.png")
