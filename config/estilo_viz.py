# -*- coding: utf-8 -*-
"""Tema visual compartido por todas las figuras del proyecto.

Una sola paleta y un solo estilo para las 15 gráficas, de modo que el informe se
lea como un conjunto y no como quince figuras sueltas.
"""

# Paleta categórica validada (orden fijo; no se cicla). Un color por posición.
PALETA_POS = {"GK": "#2a78d6", "DF": "#eb6834", "MF": "#1baf7a", "FW": "#eda100"}

# Tinta separada del color de serie: el texto nunca compite con los datos.
TINTA = "#0b0b0b"    # títulos y anotaciones destacadas
TINTA2 = "#52514e"   # etiquetas de eje y anotaciones secundarias
GRIS = "#c9c8c3"     # elementos neutros (barras "fuera", líneas de conexión)

ESTILO_VIZ = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "text.color": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "font.size": 9,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
}

# Resolución de guardado de las figuras en reports/figuras/.
DPI_GUARDADO = 150


def aplicar_estilo() -> None:
    """Aplica ESTILO_VIZ a matplotlib. La llama src.visualizacion al importarse."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(ESTILO_VIZ)
