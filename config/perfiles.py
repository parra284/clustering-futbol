# -*- coding: utf-8 -*-
"""Nombre futbolístico de cada uno de los 15 clusters.

Salen de las interpretaciones de las secciones 4 a 7 del notebook. K-Means **no
numera los clusters igual al reentrenar**, así que si se reentrena con otros datos
o con otro `k` hay que revisarlos uno por uno.

Para que ese riesgo no llegue al dashboard, `src.modelado.exportar()` escribe el
nombre resuelto como columna `perfil` dentro del CSV en el momento de exportar: la
etiqueta viaja con los datos, y el dashboard no vuelve a mapear nada por su cuenta.
"""

PERFILES = {
    "GK": {0: "Portero de área",
           1: "Portero muy exigido",
           2: "Portero de bloque sólido"},
    "DF": {0: "Lateral de combate",
           1: "Central",
           2: "Carrilero ofensivo"},
    "MF": {0: "Carrilero",
           1: "Interior de conducción",
           2: "Interior ofensivo",
           3: "Pivote destructor"},
    "FW": {0: "Delantero asociativo",
           1: "Nueve de choque",
           2: "Extremo de presión",
           3: "Extremo centrador",
           4: "Nueve de área"},
}

# Columnas que el dashboard lee de cada CSV exportado.
COLUMNAS_DASHBOARD = ["league", "team", "player", "pos", "age",
                      "Playing Time_Min", "cluster", "perfil",
                      "confianza", "dist_centroide"]

# Equipo que aparece seleccionado al abrir el dashboard.
EQUIPO_POR_DEFECTO = "Arsenal"
