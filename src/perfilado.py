# -*- coding: utf-8 -*-
"""Lectura y consulta de los resultados ya exportados.

Todo lo que el dashboard necesita saber sobre un jugador vive aquí, no en la capa de
interfaz: así el notebook, un script o cualquier otro consumidor obtienen exactamente
las mismas cifras que se ven en pantalla.

Nada de este módulo reentrena: parte de los CSV de `reports/`.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from config import parametros as P
from config import rutas as R
from src.preparacion import _es_tasa, _etiqueta, preparar_features

# Columnas de identidad, en el orden en que se muestran.
IDENTIDAD = ["player", "team", "league", "season", "nation", "pos",
             "age", "born", "Playing Time_Min", "Playing Time_90s"]

NOMBRE_IDENTIDAD = {
    "player": "Jugador", "team": "Equipo", "league": "Liga", "season": "Temporada",
    "nation": "País", "pos": "Posición FBref", "age": "Edad", "born": "Nacido",
    "Playing Time_Min": "Minutos", "Playing Time_90s": "Partidos (90s)",
}


def cargar_etiquetados() -> dict[str, pd.DataFrame]:
    """Los cuatro CSV de `reports/`, tal cual, indexados por posición."""
    datos = {}
    for pos in P.POSICIONES:
        ruta = R.csv_jugadores(pos)
        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta {ruta}. Ejecuta antes: python -m models.entrenar_todos")
        datos[pos] = pd.read_csv(ruta)
    return datos


def etiquetas_features(pos: str) -> list[str]:
    """Nombres cortos de las variables del modelo: 'Performance_Int' -> 'Int/90'."""
    return [_etiqueta(c, not _es_tasa(c)) for c in P.FEATURES[pos]]


def features_comunes(posiciones: list[str]) -> list[str]:
    """Variables de modelo que comparten TODAS las posiciones dadas.

    Cada posición se segmenta con sus propias métricas, así que al mirar varias a la vez
    solo se puede ordenar por lo que todas midieron: pedir «ordenar por Save%» en una
    lista que incluye defensas no tiene respuesta. Entre posiciones de campo quedan 7-9
    variables; en cuanto entran los porteros, solo `Fld/90`.

    El orden es el de la primera posición, para que la lista no baile entre selecciones.
    """
    if not posiciones:
        return []
    comunes = set.intersection(*(set(etiquetas_features(p)) for p in posiciones))
    return [e for e in etiquetas_features(posiciones[0]) if e in comunes]


def con_por90(df_pos: pd.DataFrame, pos: str) -> pd.DataFrame:
    """El dataset de una posición con sus variables por 90 añadidas como columnas.

    Las etiquetas cortas (`Int/90`, `Save%`) no chocan con los nombres de FBref
    (`Performance_Int`, `Performance_Save%`), así que ambas conviven en la misma tabla:
    las de FBref son totales de temporada y estas son las tasas que vio el modelo.
    """
    return df_pos.join(tabla_por90(df_pos, pos))


def tabla_por90(df_pos: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Las variables del modelo en unidades reales por 90 minutos.

    Es lo que hay que enseñar a una persona: el modelo no ve "23 intercepciones en la
    temporada" sino "0.68 intercepciones por 90". A diferencia de `preparar_features`,
    aquí no se imputa nada — un dato que falta se muestra como que falta.
    """
    X = df_pos[P.FEATURES[pos]].astype(float).copy()
    totales = [c for c in P.FEATURES[pos] if not _es_tasa(c)]
    X[totales] = X[totales].div(df_pos[P.COL_90S], axis=0)
    X.columns = etiquetas_features(pos)
    return X


def percentiles(por90: pd.DataFrame) -> pd.DataFrame:
    """Percentil (0-100) de cada jugador en cada variable, dentro de su posición.

    Un percentil se lee sin conocer la escala de la métrica: "está en el 92 de
    intercepciones" dice más que "0.68 Int/90" a quien no vive en estos datos.
    """
    return por90.rank(pct=True, na_option="keep").mul(100)


def matriz_escalada(df_pos: pd.DataFrame, pos: str) -> pd.DataFrame:
    """La misma matriz que entra al K-Means. Se usa para medir parecido."""
    X, _, _ = preparar_features(df_pos, P.FEATURES[pos], verbose=False)
    return X


def similares(X: pd.DataFrame, fila: int, n: int = 8) -> pd.DataFrame:
    """Los `n` jugadores más parecidos a `fila`, en el espacio del modelo.

    La distancia es la misma que usa K-Means para asignar clusters, así que "parecido"
    aquí significa exactamente lo que significa dentro del modelo. Se buscan en toda la
    posición, no solo dentro del cluster: dos jugadores muy parecidos pueden caer a
    ambos lados de una frontera, y ese caso es justo el interesante.
    """
    d = cdist(X.values[[fila]], X.values)[0]
    orden = np.argsort(d)
    orden = orden[orden != fila][:n]
    return pd.DataFrame({"fila": orden, "distancia": d[orden]})


def resumen_variable(df_pos: pd.DataFrame, por90: pd.DataFrame, pos: str,
                     fila: int) -> pd.DataFrame:
    """Tabla comparativa de un jugador: su valor, el de su perfil y el de la posición.

    Las tres cifras juntas son lo que convierte un número en una lectura: 0.68 Int/90
    no dice nada hasta que se sabe que la media de su perfil es 0.51 y la de todos los
    de su posición 0.44.
    """
    cluster = df_pos["cluster"].iloc[fila]
    del_perfil = df_pos["cluster"].values == cluster
    pcts = percentiles(por90)

    return pd.DataFrame({
        "Variable": por90.columns,
        "Jugador": por90.iloc[fila].values,
        "Media del perfil": por90[del_perfil].mean().values,
        "Media de la posición": por90.mean().values,
        "Percentil": pcts.iloc[fila].values,
    })


def buscar(datos: dict[str, pd.DataFrame], texto: str) -> pd.DataFrame:
    """Busca un texto en los nombres de las cuatro posiciones.

    Devuelve una fila por (jugador, equipo, modelo): un polivalente aparece una vez por
    cada línea en la que se le evaluó, que es justo lo que hay que poder elegir.
    """
    texto = (texto or "").strip().lower()
    if not texto:
        return pd.DataFrame(columns=["player", "team", "pos", "modelo", "fila"])

    trozos = []
    for pos, d in datos.items():
        m = d["player"].str.lower().str.contains(texto, regex=False, na=False)
        if m.any():
            t = d.loc[m, ["player", "team", "league", "pos", "cluster", "perfil"]].copy()
            t["modelo"] = pos
            t["fila"] = np.flatnonzero(m)
            trozos.append(t)

    if not trozos:
        return pd.DataFrame(columns=["player", "team", "pos", "modelo", "fila"])
    return pd.concat(trozos, ignore_index=True).sort_values(["player", "team"])
