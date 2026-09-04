# -*- coding: utf-8 -*-
"""Dashboard de perfiles de jugadores.

Cuatro vistas sobre los resultados que exportan los modelos:

  Jugador   · buscar a uno y ver su ficha, su perfil y a quién se parece
  Equipo    · la plantilla completa repartida por perfiles
  Explorar  · filtrar las 2 509 apariciones por perfil, edad, liga y minutos
  Comparar  · dos jugadores de la misma posición, variable a variable

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

from app import graficos
from config import estilo_viz
from config import parametros as P
from config.perfiles import EQUIPO_POR_DEFECTO
from src import perfilado

st.set_page_config(page_title="Perfiles de jugadores", page_icon="⚽", layout="wide")


# ======================================================================================
# Datos y tema
# ======================================================================================

@st.cache_data(show_spinner="Cargando resultados…")
def cargar():
    """Los cuatro CSV, sus valores por 90 y sus percentiles. Se calcula una sola vez."""
    datos = perfilado.cargar_etiquetados()
    por90 = {p: perfilado.tabla_por90(datos[p], p) for p in P.POSICIONES}
    pcts = {p: perfilado.percentiles(por90[p]) for p in P.POSICIONES}
    return datos, por90, pcts


@st.cache_data(show_spinner=False)
def escalada(pos: str):
    """La matriz que entra al K-Means, para medir parecido entre jugadores."""
    return perfilado.matriz_escalada(cargar()[0][pos], pos)


def tema_actual() -> dict:
    """Paleta acorde al tema que tenga puesto quien mira."""
    try:
        modo = st.context.theme.type or "light"
    except Exception:
        modo = "light"
    return estilo_viz.tema(modo)


try:
    DATOS, POR90, PCTS = cargar()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

T = tema_actual()

# Los umbrales se redactan desde config: escritos a mano se quedaban obsoletos en
# cuanto alguien tocaba MINUTOS_ENTRENAMIENTO.
PART_INCLUSION = P.MINUTOS_INCLUSION // P.PARTIDO
PART_CAMPO = P.MINUTOS_ENTRENAMIENTO["DF"] // P.PARTIDO
PART_GK = P.MINUTOS_ENTRENAMIENTO["GK"] // P.PARTIDO

AYUDA_CONFIANZA = (f"«baja» = jugó menos de {PART_CAMPO} partidos ({PART_GK} si es "
                   f"portero). Recibió etiqueta, pero no ayudó a definir los perfiles.")
AYUDA_DISTANCIA = "Qué tan cerca está del arquetipo. Menor = encaja más limpio."

COLS_TABLA = {
    "player": "Jugador", "team": "Equipo", "league": "Liga", "pos": "Posición FBref",
    "age": "Edad", "Playing Time_Min": "Minutos", "perfil": "Perfil",
    "confianza": "Confianza", "dist_centroide": "Distancia al centroide",
}

FORMATO_TABLA = {
    "Edad": st.column_config.NumberColumn(format="%d"),
    "Minutos": st.column_config.NumberColumn(format="%d"),
    "Distancia al centroide": st.column_config.NumberColumn(format="%.2f",
                                                            help=AYUDA_DISTANCIA),
}


def tabla_jugadores(d: pd.DataFrame, columnas: list[str]):
    """Renderiza un bloque de jugadores con los nombres y formatos de siempre."""
    st.dataframe(d[columnas].rename(columns=COLS_TABLA), hide_index=True,
                 width="stretch", column_config=FORMATO_TABLA)


st.title("⚽ Perfiles de jugadores por posición")
st.caption(
    f"5 grandes ligas europeas, temporada 2023-24. Cada jugador se compara solo contra "
    f"los de su propia posición, con las métricas de esa posición. Un polivalente "
    f"aparece una vez por cada línea en la que se le evaluó."
)

tab_jugador, tab_equipo, tab_explorar, tab_comparar = st.tabs(
    ["🔍 Jugador", "👥 Equipo", "🧭 Explorar", "⚖️ Comparar"])


# ======================================================================================
# 1 · Jugador
# ======================================================================================

with tab_jugador:
    texto = st.text_input("Buscar jugador", placeholder="Rodri, Bellingham, Saka…",
                          key="busca_jugador")
    encontrados = perfilado.buscar(DATOS, texto)

    if not texto:
        st.info("Escribe un nombre para ver su ficha.")
    elif encontrados.empty:
        st.warning(f"Ningún jugador coincide con «{texto}».")
    else:
        opciones = {
            f"{r.player} · {r.team} — {P.NOMBRE_POS[r.modelo]}": (r.modelo, int(r.fila))
            for r in encontrados.itertuples()
        }
        elegido = st.selectbox(f"{len(opciones)} coincidencia(s)", list(opciones),
                               key="sel_jugador")
        pos, fila = opciones[elegido]
        d = DATOS[pos]
        jug = d.iloc[fila]
        color = T["posiciones"][pos]

        st.subheader(f"{jug['player']}  ·  {jug['perfil']}")

        c = st.columns(6)
        c[0].metric("Equipo", jug["team"])
        c[1].metric("Liga", str(jug["league"]).split("-")[-1])
        c[2].metric("País", jug["nation"] if pd.notna(jug["nation"]) else "—")
        c[3].metric("Edad", f"{jug['age']:.0f}" if pd.notna(jug["age"]) else "—",
                    help=f"Nacido en {jug['born']:.0f}" if pd.notna(jug["born"]) else None)
        c[4].metric("Minutos", f"{jug['Playing Time_Min']:.0f}",
                    help=f"{jug['Playing Time_90s']:.1f} partidos completos")
        c[5].metric("Confianza", jug["confianza"], help=AYUDA_CONFIANZA)

        st.caption(
            f"Modelo de {P.NOMBRE_POS[pos].lower()} · posición en FBref: {jug['pos']} · "
            f"cluster {jug['cluster']} de {P.K_ELEGIDO[pos]} · "
            f"distancia al centroide {jug['dist_centroide']:.2f} — {AYUDA_DISTANCIA}"
        )

        resumen = perfilado.resumen_variable(d, POR90[pos], pos, fila)

        izq, der = st.columns([3, 2])
        with izq:
            st.markdown("**Dónde está fuerte y dónde flojo**")
            st.caption("La **barra** es el percentil dentro de su posición; la línea "
                       "punteada es la mediana. El **número** es su valor real por 90.")
            st.altair_chart(graficos.percentiles_jugador(resumen, T, color),
                            theme="streamlit")
        with der:
            st.markdown("**Las cifras**")
            st.dataframe(
                resumen.rename(columns={"Jugador": "Por 90"}),
                hide_index=True, width="stretch", height=len(resumen) * 35 + 40,
                column_config={
                    "Por 90": st.column_config.NumberColumn(format="%.2f"),
                    "Media del perfil": st.column_config.NumberColumn(format="%.2f"),
                    "Media de la posición": st.column_config.NumberColumn(format="%.2f"),
                    "Percentil": st.column_config.ProgressColumn(
                        format="%.0f", min_value=0, max_value=100),
                })

        st.divider()
        st.markdown("**Jugadores más parecidos**")
        sim = perfilado.similares(escalada(pos), fila, n=8)
        vecinos = d.iloc[sim["fila"].values][
            ["player", "team", "league", "age", "Playing Time_Min", "perfil",
             "confianza"]].copy()
        vecinos.insert(0, "Distancia", sim["distancia"].values)
        st.dataframe(
            vecinos.rename(columns=COLS_TABLA), hide_index=True, width="stretch",
            column_config={**FORMATO_TABLA,
                           "Distancia": st.column_config.NumberColumn(format="%.2f")})
        st.caption(
            f"«Parecido» significa cercano en las {len(P.FEATURES[pos])} variables de "
            f"este modelo, no parecido en todo. El modelo no mide pases, conducción ni "
            f"posicionamiento: dos jugadores pueden salir juntos aquí y jugar distinto."
        )


# ======================================================================================
# 2 · Equipo
# ======================================================================================

with tab_equipo:
    todos = pd.concat(
        [DATOS[p].assign(modelo=p) for p in P.POSICIONES], ignore_index=True)

    equipos = sorted(todos["team"].unique())
    idx = equipos.index(EQUIPO_POR_DEFECTO) if EQUIPO_POR_DEFECTO in equipos else 0
    equipo = st.selectbox("Equipo", equipos, index=idx, key="sel_equipo")

    plantilla = todos[todos["team"] == equipo]

    if plantilla.empty:
        st.warning(f"Ese equipo no tiene jugadores con {PART_INCLUSION} partidos o más.")
    else:
        c = st.columns(4)
        c[0].metric("Jugadores", plantilla["player"].nunique())
        c[1].metric("Apariciones", len(plantilla),
                    help="Un polivalente cuenta una vez por cada posición evaluada.")
        c[2].metric("Perfiles distintos", plantilla["perfil"].nunique())
        c[3].metric("Confianza baja", int((plantilla["confianza"] == "baja").sum()),
                    help=AYUDA_CONFIANZA)

        st.divider()

        for pos in P.POSICIONES:
            bloque = plantilla[plantilla["modelo"] == pos]
            if bloque.empty:
                continue

            st.subheader(f"{P.NOMBRE_POS[pos]}  ·  {len(bloque)} jugadores")
            izq, der = st.columns([3, 2])

            with izq:
                tabla_jugadores(
                    bloque.sort_values(["perfil", "Playing Time_Min"],
                                       ascending=[True, False]),
                    ["player", "perfil", "pos", "age", "Playing Time_Min",
                     "confianza", "dist_centroide"])
            with der:
                reparto = bloque["perfil"].value_counts()
                if len(reparto) > 1:
                    st.altair_chart(
                        graficos.reparto_perfiles(reparto, T, T["posiciones"][pos]),
                        theme="streamlit")
                else:
                    st.caption(f"Todos en un mismo perfil: {reparto.index[0]}.")


# ======================================================================================
# 3 · Explorar
# ======================================================================================

with tab_explorar:
    f = st.columns([1, 2, 1.4, 1.4])

    posiciones = f[0].multiselect(
        "Posición", P.POSICIONES, default=P.POSICIONES,
        format_func=lambda p: P.NOMBRE_POS[p], key="exp_pos")

    # Sin posiciones no hay nada que filtrar, pero no se puede usar st.stop(): eso
    # aborta el script entero y se llevaría por delante la pestaña de comparación.
    if not posiciones:
        st.info("Elige al menos una posición.")

    universo = pd.concat(
        [DATOS[p].assign(modelo=p) for p in (posiciones or P.POSICIONES)],
        ignore_index=True)

    perfiles_disp = sorted(universo["perfil"].unique())
    perfiles = f[1].multiselect("Perfil", perfiles_disp, default=perfiles_disp,
                                key="exp_perfil")

    edad_min = int(universo["age"].min())
    edad_max = int(universo["age"].max())
    rango_edad = f[2].slider("Edad", edad_min, edad_max, (edad_min, edad_max),
                             key="exp_edad")

    min_max = int(universo["Playing Time_Min"].max())
    minutos = f[3].slider("Minutos mínimos", 0, min_max, P.MINUTOS_INCLUSION, step=90,
                          key="exp_min")

    ligas = st.multiselect("Liga", sorted(universo["league"].unique()),
                           default=sorted(universo["league"].unique()), key="exp_liga")

    sel = universo[
        universo["perfil"].isin(perfiles)
        & universo["age"].between(*rango_edad)
        & (universo["Playing Time_Min"] >= minutos)
        & universo["league"].isin(ligas)
    ].sort_values("Playing Time_Min", ascending=False) if posiciones else universo.iloc[:0]

    st.divider()

    if sel.empty:
        st.warning("Ningún jugador cumple esos filtros.")
    else:
        c = st.columns(4)
        c[0].metric("Apariciones", len(sel))
        c[1].metric("Jugadores", sel["player"].nunique())
        c[2].metric("Edad mediana", f"{sel['age'].median():.0f}")
        c[3].metric("Confianza baja", int((sel["confianza"] == "baja").sum()),
                    help=AYUDA_CONFIANZA)

        st.markdown("**Ordenados por minutos jugados**")
        tabla_jugadores(sel, ["player", "team", "league", "perfil", "pos", "age",
                              "Playing Time_Min", "confianza", "dist_centroide"])

        st.markdown("**Edad frente a minutos**")
        usados = sorted(sel["perfil"].unique())

        # Un panel por perfil solo mientras se puedan comparar de un vistazo. Con los 15
        # perfiles a la vez salen quince nubes que nadie lee: por encima de MAX_PANELES
        # se dibuja una sola, y se dice cómo separarlas.
        MAX_PANELES = 6
        facetar = 1 < len(usados) <= MAX_PANELES

        if len(usados) > MAX_PANELES:
            st.caption(f"{len(usados)} perfiles en una sola nube. Filtra a "
                       f"{MAX_PANELES} o menos para verlos en paneles separados.")
        else:
            st.caption("Abajo a la izquierda están los jóvenes con pocos minutos: el "
                       "caso de scouting que motiva el umbral de inclusión bajo.")

        st.altair_chart(
            graficos.edad_vs_minutos(
                sel, T,
                T["posiciones"][posiciones[0]] if len(posiciones) == 1 else T["serie_1"],
                facetar=facetar),
            theme="streamlit")


# ======================================================================================
# 4 · Comparar
# ======================================================================================

with tab_comparar:
    st.caption("Solo se comparan jugadores de la misma posición: cada modelo usa sus "
               "propias variables, y un percentil de defensa no significa lo mismo que "
               "uno de delantero.")

    pos = st.selectbox("Posición", P.POSICIONES, format_func=lambda p: P.NOMBRE_POS[p],
                       key="cmp_pos")
    d = DATOS[pos]
    etiquetas = (d["player"] + " · " + d["team"]).tolist()

    izq, der = st.columns(2)
    i_a = izq.selectbox("Jugador A", range(len(etiquetas)),
                        format_func=lambda i: etiquetas[i], key="cmp_a")
    i_b = der.selectbox("Jugador B", range(len(etiquetas)),
                        index=min(1, len(etiquetas) - 1),
                        format_func=lambda i: etiquetas[i], key="cmp_b")

    if i_a == i_b:
        st.info("Elige dos jugadores distintos.")
    else:
        ja, jb = d.iloc[i_a], d.iloc[i_b]
        nombres = (f"{ja['player']} ({ja['team']})", f"{jb['player']} ({jb['team']})")

        c = st.columns(2)
        for col, j, nombre in zip(c, (ja, jb), nombres):
            col.markdown(f"### {j['player']}")
            col.markdown(f"**{j['perfil']}**  ·  cluster {j['cluster']}")
            col.caption(
                f"{j['team']} · {str(j['league']).split('-')[-1]} · "
                f"{j['nation'] if pd.notna(j['nation']) else '—'} · "
                f"{j['age']:.0f} años · {j['Playing Time_Min']:.0f} min · "
                f"confianza {j['confianza']}")

        if ja["cluster"] == jb["cluster"]:
            st.success(f"Los dos caen en el mismo perfil: **{ja['perfil']}**.")
        else:
            st.info(f"Perfiles distintos: **{ja['perfil']}** vs **{jb['perfil']}**.")

        pct = PCTS[pos]
        p90 = POR90[pos]
        tabla = pd.DataFrame({
            "Variable": p90.columns,
            nombres[0]: pct.iloc[i_a].values,
            nombres[1]: pct.iloc[i_b].values,
        })

        st.divider()
        st.markdown("**Variable a variable, en percentiles**")
        st.altair_chart(graficos.comparar_jugadores(tabla, T, nombres), theme="streamlit")

        st.markdown("**Las cifras reales por 90 minutos**")
        crudo = pd.DataFrame({
            "Variable": p90.columns,
            nombres[0]: p90.iloc[i_a].values,
            nombres[1]: p90.iloc[i_b].values,
        })
        crudo["Diferencia"] = crudo[nombres[0]] - crudo[nombres[1]]
        st.dataframe(
            crudo, hide_index=True, width="stretch",
            column_config={n: st.column_config.NumberColumn(format="%.2f")
                           for n in (*nombres, "Diferencia")})


st.divider()
st.caption(
    "Los perfiles salen de cuatro modelos K-Means independientes, uno por posición, "
    "entrenados sobre las 5 grandes ligas europeas 2023-24. Una etiqueta de "
    "*confianza baja* dice que el jugador se parece a ese arquetipo según los minutos "
    "que lleva, no que sea su rol confirmado."
)
