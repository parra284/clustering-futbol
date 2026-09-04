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


# ======================================================================================
# Tema del dashboard (Altair)
# ======================================================================================
# Los mismos cuatro tonos de PALETA_POS, más los pasos equivalentes para fondo oscuro.
# No es un volteo automático del claro: son escalones elegidos para cada superficie.
# Las dos series de la comparación (slots 1 y 2) están validadas en ambos modos.

TEMA_CLARO = {
    "posiciones": PALETA_POS,
    "serie_1": "#2a78d6", "serie_2": "#eb6834",
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b", "tinta2": "#52514e", "apagado": "#898781",
    "rejilla": "#e1e0d9", "eje": "#c3c2b7",
}

TEMA_OSCURO = {
    "posiciones": {"GK": "#3987e5", "DF": "#d95926", "MF": "#199e70", "FW": "#c98500"},
    "serie_1": "#3987e5", "serie_2": "#d95926",
    "superficie": "#1a1a19",
    "tinta": "#ffffff", "tinta2": "#c3c2b7", "apagado": "#898781",
    "rejilla": "#2c2c2a", "eje": "#383835",
}


def tema(modo: str = "light") -> dict:
    """Devuelve la paleta del dashboard para 'light' u 'dark'."""
    return TEMA_OSCURO if modo == "dark" else TEMA_CLARO


def aplicar_estilo() -> None:
    """Aplica ESTILO_VIZ a matplotlib. La llama src.visualizacion al importarse."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(ESTILO_VIZ)
