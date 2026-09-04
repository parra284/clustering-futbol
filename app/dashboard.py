# -*- coding: utf-8 -*-
"""Dashboard de perfiles de jugadores.

Lee los CSV que exportan los cuatro modelos y muestra, para un equipo, a qué perfil
pertenece cada uno de sus jugadores.

Ejecutar con:   streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

# Streamlit pone app/ en sys.path[0], no la raíz del proyecto: sin esto no encuentra
# config/ ni src/.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pandas as pd
import streamlit as st

from config import parametros as P
from config import rutas as R
from config.perfiles import COLUMNAS_DASHBOARD, EQUIPO_POR_DEFECTO


@st.cache_data
def cargar_resultados() -> pd.DataFrame:
    """Une los cuatro CSV en una sola tabla.

    El nombre del perfil llega ya resuelto en la columna `perfil`, escrita al exportar
    el modelo. Antes se mapeaba aquí por número de cluster, y como K-Means no numera
    igual al reentrenar, cualquier refit dejaba los nombres mal en silencio.
    """
    partes = []
    for pos in P.POSICIONES:
        ruta = R.csv_jugadores(pos)
        if not ruta.exists():
            st.error(f"No encuentro {ruta.name}. Ejecuta antes `python -m models.entrenar_todos`.")
            st.stop()
        d = pd.read_csv(ruta)[COLUMNAS_DASHBOARD].copy()
        d["modelo"] = pos
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


# Los umbrales se leen de config y se redactan aquí: escritos a mano se quedaban
# obsoletos en cuanto alguien tocaba MINUTOS_ENTRENAMIENTO.
PART_INCLUSION = P.MINUTOS_INCLUSION // P.PARTIDO
PART_CAMPO = P.MINUTOS_ENTRENAMIENTO["DF"] // P.PARTIDO
PART_GK = P.MINUTOS_ENTRENAMIENTO["GK"] // P.PARTIDO

st.set_page_config(page_title="Perfiles de jugadores", page_icon="⚽", layout="wide")

st.title("⚽ Perfiles de jugadores por posición")
st.caption(
    "Cada jugador se compara solo contra los de su propia posición. "
    "Los polivalentes aparecen una vez por cada línea en la que se les evaluó."
)

datos = cargar_resultados()

equipos = sorted(datos["team"].unique())
por_defecto = equipos.index(EQUIPO_POR_DEFECTO) if EQUIPO_POR_DEFECTO in equipos else 0
equipo = st.selectbox("Equipo", equipos, index=por_defecto)

plantilla = datos[datos["team"] == equipo]

if plantilla.empty:
    st.warning(f"Ese equipo no tiene jugadores que superen el mínimo de "
               f"{PART_INCLUSION} partidos.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jugadores", plantilla["player"].nunique())
col2.metric("Apariciones", len(plantilla),
            help="Un polivalente cuenta una vez por cada posición en la que se le evaluó.")
col3.metric("Perfiles distintos", plantilla["perfil"].nunique())
col4.metric("Confianza baja", int((plantilla["confianza"] == "baja").sum()),
            help=f"Jugó menos de {PART_CAMPO} partidos ({PART_GK} si es portero): recibió "
                 f"etiqueta, pero no ayudó a definir los perfiles.")

st.divider()

for pos in P.POSICIONES:
    bloque = plantilla[plantilla["modelo"] == pos]
    if bloque.empty:
        continue

    st.subheader(f"{P.NOMBRE_POS[pos]}  ·  {len(bloque)} jugadores")

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
    f"Los perfiles salen de cuatro modelos K-Means independientes, uno por posición, "
    f"entrenados sobre las 5 grandes ligas europeas 2023-24. "
    f"Una etiqueta de *confianza baja* dice que el jugador se parece a ese arquetipo "
    f"según los minutos que lleva, no que sea su rol confirmado."
)
