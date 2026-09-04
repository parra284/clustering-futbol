# -*- coding: utf-8 -*-
"""Fase 3 de CRISP-DM: separación por posición y preparación de las variables."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

from config import parametros as P
from src.utils import mostrar


def _es_tasa(col: str) -> bool:
    """True si FBref ya entrega la columna como tasa por 90 o como porcentaje."""
    return (col.startswith("Per 90") or col.endswith("/90")
            or col.endswith("90") or col.endswith("%"))


def _etiqueta(col: str, convertida: bool) -> str:
    """Nombre corto para las tablas de perfil: 'Performance_Int' -> 'Int/90'."""
    corto = col.split("_", 1)[1] if "_" in col else col
    if convertida or col.startswith("Per 90"):
        corto += "/90"
    return corto


def construir_datasets(df_base: pd.DataFrame, verbose: bool = True) -> dict:
    """Separa la base limpia en los cuatro datasets de posición.

    Se usa `str.contains(pos)` y no `== pos`: así un "DF,MF" cae en el dataset de
    defensas Y en el de mediocampistas, que es exactamente lo que queremos.

    'entrena' marca a quién se usa para ajustar imputer, scaler y centroides. El resto
    recibe etiqueta por predicción, y se marca como confianza baja para poder filtrarlo
    después en cualquier informe.
    """
    datasets = {}
    for pos in P.POSICIONES:
        d = df_base[df_base["pos"].str.contains(pos)
                    & (df_base["Playing Time_Min"] >= P.MINUTOS_INCLUSION)
                    ].reset_index(drop=True).copy()
        d["entrena"] = d["Playing Time_Min"] >= P.MINUTOS_ENTRENAMIENTO[pos]
        d["confianza"] = np.where(d["entrena"], "alta", "baja")
        datasets[pos] = d

    if verbose:
        resumen = pd.DataFrame({
            "código": P.POSICIONES,
            "posición": [P.NOMBRE_POS[p] for p in P.POSICIONES],
            "umbral_entren.": [f"{P.MINUTOS_ENTRENAMIENTO[p] // P.PARTIDO} part."
                               for p in P.POSICIONES],
            "n_etiquetados": [len(datasets[p]) for p in P.POSICIONES],
            "n_entrenan": [int(datasets[p]["entrena"].sum()) for p in P.POSICIONES],
            "n_conf_baja": [int((~datasets[p]["entrena"]).sum()) for p in P.POSICIONES],
        })
        resumen["% conf. baja"] = (100 * resumen["n_conf_baja"]
                                   / resumen["n_etiquetados"]).round(0)
        print(f"\n--- Los 4 datasets (inclusión: {P.MINUTOS_INCLUSION} min) ---")
        print(resumen.to_string(index=False))

    return datasets


def universo_analizado(df_base: pd.DataFrame, datasets: dict,
                       verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jugadores que entran al menos a un modelo, y el subconjunto de polivalentes.

    Devuelve (analizados, polivalentes). Es el contexto que la sección 2 del notebook
    reporta antes de modelar: reparto por liga, cobertura de scouting y dobles roles.
    """
    en_algun_modelo = (pd.concat([datasets[p][["player", "team"]] for p in P.POSICIONES])
                       .drop_duplicates())
    analizados = df_base.merge(en_algun_modelo, on=["player", "team"], how="inner")
    polivalentes = analizados[analizados["pos"].str.contains(",")]

    if verbose:
        print("\n--- Reparto por liga (jugadores en al menos un modelo) ---")
        print(analizados["league"].value_counts().to_string())

        jovenes = analizados[analizados["age"] <= P.EDAD_JOVEN]
        print(f"\nJugadores de {P.EDAD_JOVEN} años o menos con etiqueta: {len(jovenes)} "
              f"de {len(analizados)} analizados")

        total_apariciones = sum(len(datasets[p]) for p in P.POSICIONES)
        print(f"\nSuma de los 4 datasets: {total_apariciones} apariciones "
              f"para {len(analizados)} jugadores distintos")
        print(f"Jugadores polivalentes: {len(polivalentes)}")
        print(polivalentes["pos"].value_counts().to_string())

    return analizados, polivalentes


def preparar_features(df_pos, cols, escalado=None, col_90s=None,
                      col_entrena="entrena", verbose=True):
    """Fase 3 para el dataset de UNA posición: por-90 → imputación → escalado.

    1. **Por 90 minutos.** Todo total de temporada se divide entre los 90s jugados. Sin
       esto el modelo separa titulares de suplentes en vez de perfiles de juego: un lateral
       con 30 partidos siempre acumulará más centros que uno con 8, jueguen igual o no.
       Las columnas que FBref ya entrega como tasa (`Per 90 Minutes_*`, `Sh/90`, `GA90`) o
       como porcentaje (`Save%`, `CS%`, `SoT%`) se dejan intactas.

    2. **Imputación** por mediana, robusta a outliers.

    3. **Escalado.** K-Means mide distancias euclídeas, así que sin escalar mandan las
       variables de magnitud grande (`SoT%` va de 0 a 100; `Ast`/90 vive entre 0 y 0.4) y
       las de cola larga. Dos opciones:
         · "yeo-johnson" (por defecto): transformación de potencia que simetriza la
           distribución y después estandariza a media 0 y desviación 1. Las tasas por 90
           son conteos divididos entre minutos, con muchos ceros y cola derecha larga:
           en defensas, `Ast`/90 tiene asimetría 1.60 y su valor más extremo queda a 4.8
           desviaciones de la media, de modo que ese único jugador pesa en la distancia
           más que varias variables juntas. Yeo-Johnson deja esa misma cola en ~2.1
           desviaciones y todas las variables entran al modelo con un peso comparable.
         · "standard": StandardScaler puro (media 0, desviación 1), sin corregir asimetría.

    El imputer y el scaler se ajustan **solo con los jugadores de entrenamiento** (los que
    superan el umbral alto de minutos) y después se aplican a todos. Así la mediana de
    imputación y los parámetros de la transformación salen de tasas bien medidas, y un
    juvenil con 600 minutos no desplaza la escala con la que se mide a los demás.

    Con `verbose=False` no imprime nada: así las celdas que la llaman en bucle (las
    figuras de 3.1 y 3.3) emiten solo su gráfica.

    Devuelve (X_escalado, X_por90, objetos), los tres alineados con df_pos.index y con
    TODAS las filas (entrenan o no):
      - X_escalado : matriz que entra al modelo
      - X_por90    : mismos datos sin escalar, para leer los perfiles en unidades reales
      - objetos    : imputer y scaler ajustados, para reutilizarlos con jugadores nuevos
    """
    escalado = P.ESCALADO if escalado is None else escalado
    col_90s = P.COL_90S if col_90s is None else col_90s

    ya_por90 = [c for c in cols if _es_tasa(c)]
    totales = [c for c in cols if c not in ya_por90]
    etiquetas = [_etiqueta(c, c in totales) for c in cols]

    X = df_pos[cols].astype(float).copy()
    X[totales] = X[totales].div(df_pos[col_90s], axis=0)
    n_nan = int(X.isna().sum().sum())

    entrena = (df_pos[col_entrena].values if col_entrena in df_pos
               else np.ones(len(df_pos), dtype=bool))

    imputer = SimpleImputer(strategy="median").fit(X[entrena])
    X_por90 = pd.DataFrame(imputer.transform(X), columns=etiquetas, index=df_pos.index)

    if escalado == "yeo-johnson":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    elif escalado == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("escalado debe ser 'yeo-johnson' o 'standard'")

    scaler.fit(X_por90[entrena])
    X_escalado = pd.DataFrame(scaler.transform(X_por90),
                              columns=etiquetas, index=df_pos.index)

    if verbose:
        print(f"  {len(totales)} de {len(cols)} columnas convertidas a por-90 | "
              f"{n_nan} faltantes imputados | escalado: {escalado}")
        print(f"  Matriz final: {X_escalado.shape[0]} jugadores × {X_escalado.shape[1]} variables "
              f"({int(entrena.sum())} de entrenamiento, {int((~entrena).sum())} de confianza baja)")

        # Diagnóstico del escalado. 'max |z|' es lo que decide si una variable domina: mide a
        # cuántas desviaciones queda el jugador más extremo, y por tanto cuánto puede tirar él
        # solo de un centroide. Cuanto más parecidos sean los max |z| entre variables, más
        # equilibrado es el peso de cada una dentro de la distancia.
        diag = pd.DataFrame({
            "% ceros": (X_por90 == 0).mean().mul(100).round(1),
            "asimetría por-90": X_por90[entrena].skew().round(2),
            "asimetría escalada": X_escalado[entrena].skew().round(2),
            "max |z| (entren.)": X_escalado[entrena].abs().max().round(2),
            "max |z| (todos)": X_escalado.abs().max().round(2),
        })
        mostrar(diag)

    objetos = {"imputer": imputer, "scaler": scaler, "escalado": escalado,
               "cols": cols, "etiquetas": etiquetas, "entrena": entrena,
               "convertidas_a_90": totales, "ya_por90": ya_por90}
    return X_escalado, X_por90, objetos
