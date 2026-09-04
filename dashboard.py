# -*- coding: utf-8 -*-
"""Dashboard de perfiles de jugadores — BORRADOR.

Lee los CSV que exporta la sección 8 del notebook y muestra, para un equipo,
a qué perfil pertenece cada uno de sus jugadores.

Ejecutar con:   streamlit run dashboard.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

CARPETA = Path(__file__).parent / "resultados_clustering"
POSICIONES = ["GK", "DF", "MF", "FW"]
NOMBRE_POS = {"GK": "Porteros", "DF": "Defensas",
              "MF": "Mediocampistas", "FW": "Delanteros"}

# Nombres de los clusters, tomados de las interpretaciones de las secciones 4 a 7
# del notebook. Si se reentrena con otros datos o con otro `k`, hay que revisarlos.
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

COLUMNAS = ["league", "team", "player", "pos", "age",
            "Playing Time_Min", "cluster", "confianza", "dist_centroide"]


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """Une los cuatro CSV en una sola tabla, con el nombre del perfil resuelto."""
    partes = []
    for pos in POSICIONES:
        ruta = CARPETA / f"jugadores_{pos}_con_cluster.csv"
        if not ruta.exists():
            st.error(f"No encuentro {ruta.name}. Ejecuta la sección 8 del notebook primero.")
            st.stop()
        d = pd.read_csv(ruta)[COLUMNAS].copy()
        d["modelo"] = pos
        d["perfil"] = d["cluster"].map(PERFILES[pos])
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


st.set_page_config(page_title="Perfiles de jugadores", page_icon="⚽", layout="wide")

st.title("⚽ Perfiles de jugadores por posición")
st.caption(
    "Borrador. Cada jugador se compara solo contra los de su propia posición. "
    "Los polivalentes aparecen una vez por cada línea en la que se les evaluó."
)

datos = cargar_datos()

equipos = sorted(datos["team"].unique())
por_defecto = equipos.index("Arsenal") if "Arsenal" in equipos else 0
equipo = st.selectbox("Equipo", equipos, index=por_defecto)

plantilla = datos[datos["team"] == equipo]

if plantilla.empty:
    st.warning("Ese equipo no tiene jugadores que superen el mínimo de 5 partidos.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jugadores", plantilla["player"].nunique())
col2.metric("Apariciones", len(plantilla),
            help="Un polivalente cuenta una vez por cada posición en la que se le evaluó.")
col3.metric("Perfiles distintos", plantilla["perfil"].nunique())
col4.metric("Confianza baja", int((plantilla["confianza"] == "baja").sum()),
            help="Jugó menos de 10 partidos (15 si es portero): recibió etiqueta, "
                 "pero no ayudó a definir los perfiles.")

st.divider()

for pos in POSICIONES:
    bloque = plantilla[plantilla["modelo"] == pos]
    if bloque.empty:
        continue

    st.subheader(f"{NOMBRE_POS[pos]}  ·  {len(bloque)} jugadores")

    tabla = (bloque
             .sort_values(["perfil", "Playing Time_Min"], ascending=[True, False])
             .rename(columns={"player": "Jugador", "pos": "Posición FBref",
                              "age": "Edad", "Playing Time_Min": "Minutos",
                              "perfil": "Perfil", "confianza": "Confianza",
                              "dist_centroide": "Distancia al centroide"})
             [["Jugador", "Perfil", "Posición FBref", "Edad", "Minutos",
               "Confianza", "Distancia al centroide"]])

    st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Edad": st.column_config.NumberColumn(format="%d"),
            "Minutos": st.column_config.NumberColumn(format="%d"),
            "Distancia al centroide": st.column_config.NumberColumn(
                format="%.2f",
                help="Qué tan cerca está del arquetipo. Menor = encaja más limpio."),
        },
    )

    reparto = bloque["perfil"].value_counts()
    if len(reparto) > 1:
        st.bar_chart(reparto, horizontal=True, height=max(120, 40 * len(reparto)))

st.divider()
st.caption(
    "Los perfiles salen de cuatro modelos K-Means independientes, uno por posición, "
    "entrenados sobre las 5 grandes ligas europeas 2023-24. "
    "Una etiqueta de *confianza baja* dice que el jugador se parece a ese arquetipo "
    "según los minutos que lleva, no que sea su rol confirmado."
)
