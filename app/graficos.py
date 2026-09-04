# -*- coding: utf-8 -*-
"""Gráficos del dashboard, en Altair.

Tres reglas que se aplican en todos:

- **Un solo eje.** Nunca dos escalas en la misma gráfica. Por eso las comparaciones se
  dibujan en percentiles y no en unidades reales: `Crs`/90 llega a 5 y `Ast`/90 a 0.5,
  y en un eje común la segunda sería invisible. Las unidades reales van en la tabla.
- **El color no lleva información que no esté también escrita.** Todas las barras
  llevan su valor al lado, y las dos series de la comparación llevan leyenda.
- **El tema lo pone Streamlit.** Aquí solo se eligen los colores de las *marcas*
  (`config.estilo_viz`). El fondo, los ejes y el texto los colorea Streamlit, que es
  quien sabe con certeza si la página se ve en claro o en oscuro.
"""
import altair as alt
import pandas as pd

ALTURA_FILA = 42          # px por fila del eje Y
ALTURA_CHROME = 90        # eje X, su título y la leyenda
ALTURA_MINIMA = 220


def _altura(n_filas: int) -> int:
    """Alto del bloque para `n_filas` categorías en el eje Y.

    En Vega-Lite `height` es solo el área de dibujo, pero Streamlit renderiza con
    `autosize: fit`, y ahí el alto pedido tiene que dar también para el eje X, su
    título y la leyenda. Sin ese margen, el modelo de porteros (5 variables) deja unos
    17 px por fila y Vega directamente **borra** las etiquetas que no le caben: se veían
    3 de 5 variables, sin ningún aviso.
    """
    return max(ALTURA_MINIMA, ALTURA_FILA * n_filas + ALTURA_CHROME)


def _base(chart, t: dict):
    """Ajustes de tamaño del gráfico de nivel superior.

    Deliberadamente **no** se pinta el fondo ni el color de los ejes: de eso se encarga
    el tema de Streamlit (`st.altair_chart(..., theme="streamlit")`), que sabe con
    certeza si la página se está viendo en claro o en oscuro. Detectarlo desde Python
    con `st.context.theme` no es fiable —llega a decir «oscuro» mientras la página se
    dibuja en claro— y el resultado era un recuadro negro sobre fondo blanco.

    Lo único que se fija aquí son los colores de las marcas, que sí son decisión del
    proyecto y no del tema.
    """
    return chart.configure_view(strokeWidth=0)


def percentiles_jugador(resumen: pd.DataFrame, t: dict, color: str):
    """Percentil del jugador en cada variable, dentro de su posición.

    Un percentil se lee sin conocer la escala de la métrica. La línea del 50 es la
    mediana de la posición: a la derecha está por encima de sus pares, a la izquierda
    por debajo. El número al lado de la barra es el valor real por 90.
    """
    d = resumen.copy()
    d["etiqueta"] = d["Jugador"].map(lambda v: f"{v:.2f}")

    barras = (alt.Chart(d).mark_bar(size=13, cornerRadiusEnd=3, color=color)
              .encode(
                  x=alt.X("Percentil:Q", scale=alt.Scale(domain=[0, 100]),
                          axis=alt.Axis(tickCount=6),
                          title="percentil dentro de su posición"),
                  y=alt.Y("Variable:N", sort="-x", title=None),
                  tooltip=[alt.Tooltip("Variable:N"),
                           alt.Tooltip("Jugador:Q", title="por 90", format=".2f"),
                           alt.Tooltip("Media del perfil:Q", format=".2f"),
                           alt.Tooltip("Media de la posición:Q", format=".2f"),
                           alt.Tooltip("Percentil:Q", format=".0f")]))

    valores = (alt.Chart(d).mark_text(align="left", dx=5, fontSize=10)
               .encode(x=alt.X("Percentil:Q"), y=alt.Y("Variable:N", sort="-x"),
                       text="etiqueta:N"))

    mediana = (alt.Chart(pd.DataFrame({"x": [50]}))
               .mark_rule(strokeDash=[4, 4], color=t["apagado"], size=1)
               .encode(x="x:Q"))

    return _base(alt.layer(mediana, barras, valores)
                 .properties(height=_altura(len(d)), width="container"), t)


def reparto_perfiles(conteo: pd.Series, t: dict, color: str):
    """Cuántos jugadores hay en cada perfil. Magnitud: un solo tono."""
    d = conteo.rename_axis("Perfil").reset_index(name="Jugadores")

    barras = (alt.Chart(d).mark_bar(size=15, cornerRadiusEnd=3, color=color)
              .encode(x=alt.X("Jugadores:Q", title=None,
                              axis=alt.Axis(tickMinStep=1)),
                      y=alt.Y("Perfil:N", sort="-x", title=None),
                      tooltip=["Perfil:N", "Jugadores:Q"]))

    valores = (alt.Chart(d).mark_text(align="left", dx=5, fontSize=10)
               .encode(x="Jugadores:Q", y=alt.Y("Perfil:N", sort="-x"),
                       text="Jugadores:Q"))

    return _base(alt.layer(barras, valores)
                 .properties(height=_altura(len(d)), width="container"), t)


def edad_vs_minutos(d: pd.DataFrame, t: dict, color: str, facetar: bool):
    """Edad frente a minutos jugados.

    Sirve para lo que motiva el umbral de inclusión bajo: encontrar jugadores jóvenes
    con minutos suficientes dentro de un perfil concreto. Cuando hay varios perfiles se
    dibuja uno por panel en lugar de colorearlos: con más de tres categorías el color
    deja de distinguirse con fiabilidad, y separar los paneles no tiene ese límite.
    """
    # Altair serializa el DataFrame entero dentro de la especificación. Con las 79
    # columnas del CSV, 2 500 filas se convierten en varios MB de JSON que el navegador
    # tiene que tragarse: se recorta a lo que la gráfica realmente usa.
    d = d[["player", "team", "perfil", "age", "Playing Time_Min", "confianza"]]

    # Con cientos de puntos encima los círculos grandes se tapan entre sí y la nube deja
    # de decir dónde está la masa: el tamaño y la opacidad bajan con el número de puntos.
    denso = len(d) > 400
    tam, opac = (28, 0.45) if denso else (70, 0.75)

    puntos = (alt.Chart(d).mark_circle(size=tam, opacity=opac, color=color)
              .encode(
                  x=alt.X("age:Q", title="edad", scale=alt.Scale(zero=False, nice=True)),
                  y=alt.Y("Playing Time_Min:Q", title="minutos jugados"),
                  tooltip=[alt.Tooltip("player:N", title="Jugador"),
                           alt.Tooltip("team:N", title="Equipo"),
                           alt.Tooltip("perfil:N", title="Perfil"),
                           alt.Tooltip("age:Q", title="Edad", format=".0f"),
                           alt.Tooltip("Playing Time_Min:Q", title="Minutos", format=".0f"),
                           alt.Tooltip("confianza:N", title="Confianza")]))

    if facetar:
        grafico = puntos.properties(height=210, width=250).facet(
            facet=alt.Facet("perfil:N", title=None,
                            header=alt.Header(labelFontSize=11)),
            columns=3)
    else:
        grafico = puntos.properties(height=320, width="container")

    return _base(grafico, t)


def comparar_jugadores(tabla: pd.DataFrame, t: dict, nombres: tuple[str, str]):
    """Dos jugadores, variable a variable, en percentiles.

    Forma de mancuerna: los dos puntos de cada fila unidos por una línea. La longitud
    de la línea *es* la diferencia, que es justo lo que se viene a mirar. En percentiles
    y no en unidades reales para que todas las variables compartan un mismo eje.
    """
    a, b = nombres
    largo = tabla.melt(id_vars=["Variable"], value_vars=[a, b],
                       var_name="Jugador", value_name="Percentil")

    union = (alt.Chart(tabla).mark_rule(color=t["apagado"], size=3)
             .encode(x=alt.X(f"{a}:Q", title="percentil dentro de su posición",
                             scale=alt.Scale(domain=[0, 100]),
                             axis=alt.Axis(tickCount=6)),
                     x2=f"{b}:Q",
                     y=alt.Y("Variable:N", sort="-x", title=None)))

    puntos = (alt.Chart(largo)
              .mark_point(size=130, filled=True, opacity=1)
              .encode(
                  x=alt.X("Percentil:Q", scale=alt.Scale(domain=[0, 100])),
                  y=alt.Y("Variable:N", sort="-x", title=None),
                  color=alt.Color("Jugador:N",
                                  scale=alt.Scale(domain=[a, b],
                                                  range=[t["serie_1"], t["serie_2"]]),
                                  legend=alt.Legend(title=None, orient="top")),
                  tooltip=["Jugador:N", "Variable:N",
                           alt.Tooltip("Percentil:Q", format=".0f")]))

    return _base(alt.layer(union, puntos)
                 .properties(height=_altura(len(tabla)), width="container"), t)
