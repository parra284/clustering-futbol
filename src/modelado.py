# -*- coding: utf-8 -*-
"""Fases 4, 5 y 6 de CRISP-DM: ajuste, evaluación y despliegue de cada modelo.

Este módulo no dibuja. El ajuste devuelve resultados y delega el dibujo en
`src.visualizacion`, que solo se importa si se piden gráficas: así
`python -m models.modelo_gk` funciona sin backend gráfico y sin bloquearse en
un `plt.show()`.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import parametros as P
from config import rutas as R
from config.perfiles import PERFILES
from src.utils import mostrar


def ajustar_y_evaluar(df_pos, X_escalado, X_por90, k, pos,
                      random_state=None, n_representativos=None,
                      graficar=True, guardar_fig=None):
    """Fases 4 y 5 para una posición: ajusta el K-Means final y lo evalúa.

    Evaluación cuantitativa (silueta) y cualitativa: proyección PCA en 3D, perfil medio
    de cada cluster en unidades reales por 90, reparto de posiciones dentro de cada
    cluster y jugadores más cercanos a cada centroide.

    La PCA se ajusta siempre, se dibuje o no: forma parte del bundle exportado, porque
    sin ella no se podría situar a un jugador nuevo en los mismos ejes.
    """
    random_state = P.RANDOM_STATE if random_state is None else random_state
    n_representativos = (P.N_REPRESENTATIVOS if n_representativos is None
                         else n_representativos)

    nombre = P.NOMBRE_POS[pos]
    entrena = (df_pos["entrena"].values if "entrena" in df_pos
               else np.ones(len(df_pos), dtype=bool))

    # Los centroides se ajustan SOLO con los jugadores bien medidos; el resto recibe
    # etiqueta por predicción, sin haber podido desplazar ningún perfil.
    kmeans = KMeans(n_clusters=k, n_init=P.N_INIT, random_state=random_state)
    kmeans.fit(X_escalado[entrena])
    etiquetas = kmeans.predict(X_escalado)

    df_out = df_pos.copy()
    df_out["cluster"] = etiquetas

    # La silueta describe la partición tal como se ajustó, así que se mide sobre el
    # subconjunto de entrenamiento. La de todos se informa aparte: si cae mucho, es señal
    # de que los jugadores de confianza baja caen entre clusters y no dentro de ellos.
    sil = silhouette_score(X_escalado[entrena], etiquetas[entrena])
    sil_todos = silhouette_score(X_escalado, etiquetas)
    print(f"=== {nombre} — k={k} | n={len(df_out)} | silhouette={sil:.3f} ===")
    print(f"    {int(entrena.sum())} de entrenamiento (silueta {sil:.3f}) | "
          f"{int((~entrena).sum())} etiquetados por predicción "
          f"(silueta del conjunto completo {sil_todos:.3f})")
    print("\nTamaño de cada cluster (etiquetados / de los cuales entrenan):")
    tam = pd.DataFrame({"etiquetados": df_out["cluster"].value_counts().sort_index(),
                        "entrenan": df_out.loc[entrena, "cluster"].value_counts().sort_index()})
    print(tam.to_string())

    # --- PCA: los ejes se definen con los jugadores bien medidos, igual que los
    # centroides, y después se proyecta a todos sobre ellos.
    n_comp = min(3, X_escalado.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state).fit(X_escalado[entrena])

    if graficar:
        from src import visualizacion
        visualizacion.proyectar_pca(pca, X_escalado, etiquetas, kmeans.cluster_centers_,
                                    k, nombre, entrena=entrena, guardar=guardar_fig)

    # --- Perfil de cada cluster, en unidades reales por 90 ---
    # Se promedia sobre TODOS los etiquetados, porque es la segmentación que se exporta.
    perfil = X_por90.groupby(etiquetas).mean().T
    perfil.columns = [f"cluster {c}" for c in perfil.columns]
    perfil.insert(0, "media global", X_por90.mean())
    print("Perfil medio de cada cluster (valores por 90 minutos, sin escalar):")
    mostrar(perfil.round(3))

    # --- Reparto de posiciones reportadas dentro de cada cluster ---
    # Útil sobre todo en DF/MF/FW: muestra si un cluster concentra a los polivalentes
    # (p. ej. si el cluster ofensivo de mediocampistas está lleno de "MF,FW").
    cruce = pd.crosstab(df_out["cluster"], df_out["pos"], normalize="index").mul(100).round(1)
    if cruce.shape[1] > 1:
        print("Composición de cada cluster por posición reportada (%):")
        mostrar(cruce)

    # --- Jugadores más representativos: los más cercanos a su centroide ---
    # Se buscan solo entre los de entrenamiento: un jugador con pocos minutos puede caer
    # cerca de un centroide por azar, y como arquetipo del perfil no sirve.
    distancias = cdist(X_escalado.values, kmeans.cluster_centers_)
    df_out["dist_centroide"] = distancias[np.arange(len(df_out)), etiquetas]
    print("Jugadores más representativos de cada cluster (solo de entrenamiento):")
    for c in range(k):
        filas = np.where((etiquetas == c) & entrena)[0]   # posiciones, no etiquetas
        cercanos = filas[np.argsort(distancias[filas, c])][:n_representativos]
        nombres = ", ".join(df_out.iloc[cercanos]["player"])
        print(f"  cluster {c}: {nombres}")

    # --- Hallazgos de scouting: los de confianza baja mejor encajados en su perfil ---
    # Es el caso de uso que motiva el umbral de inclusión bajo: jugadores con pocos
    # minutos cuyo perfil, aun así, cae limpiamente dentro de un arquetipo.
    bajos = df_out[~entrena]
    if len(bajos):
        jov = bajos.nsmallest(P.N_HALLAZGOS, "dist_centroide")[
            ["player", "team", "age", "Playing Time_Min", "cluster", "dist_centroide"]]
        print("\nConfianza baja mejor encajados en su perfil (candidatos a revisar):")
        mostrar(jov.round(2).reset_index(drop=True))

    return {"pos": pos, "k": k, "kmeans": kmeans, "pca": pca, "entrena": entrena,
            "labels": etiquetas, "silhouette": sil, "silhouette_todos": sil_todos,
            "perfil": perfil, "df": df_out}


def exportar(pos: str, res: dict, prep: dict, verbose: bool = True) -> dict:
    """Fase 6: dataset etiquetado + tabla de perfiles + modelo reutilizable.

    El CSV lleva la columna `perfil` ya resuelta. Antes el nombre de cada arquetipo
    vivía en el dashboard, mapeado por número de cluster, y K-Means no numera igual al
    reentrenar: cualquier refit dejaba los quince nombres mal en silencio. Escribiéndolo
    aquí, la etiqueta viaja con los datos y con el modelo.
    """
    R.asegurar_carpetas()
    df = res["df"].copy()
    nombres = PERFILES.get(pos, {})
    df.insert(df.columns.get_loc("cluster") + 1, "perfil", df["cluster"].map(nombres))

    ruta_csv = R.csv_jugadores(pos)
    df.to_csv(ruta_csv, index=False, encoding=P.ENCODING_CSV)

    ruta_perfil = R.csv_perfil(pos)
    res["perfil"].to_csv(ruta_perfil, encoding=P.ENCODING_CSV)

    # Modelo completo para reutilizar sin reentrenar. Además de imputer/scaler/kmeans se
    # guarda qué columnas de 'features' hay que convertir a por-90 (y sobre qué columna de
    # minutos) antes de imputar y escalar: sin ese detalle, el bundle no bastaría para
    # etiquetar a un jugador nuevo siguiendo exactamente el mismo pipeline de la fase 3.
    bundle = {
        "posicion": pos,
        "nombre": P.NOMBRE_POS[pos],
        "features": P.FEATURES[pos],
        "minutos_entrenamiento": P.MINUTOS_ENTRENAMIENTO[pos],
        "minutos_inclusion": P.MINUTOS_INCLUSION,
        "col_90s": P.COL_90S,
        "convertidas_a_90": prep["convertidas_a_90"],
        "ya_por90": prep["ya_por90"],
        "etiquetas": prep["etiquetas"],
        "perfiles": nombres,          # nombre de cada cluster, junto al modelo que los produjo
        "imputer": prep["imputer"],
        "scaler": prep["scaler"],
        "kmeans": res["kmeans"],
        "pca": res["pca"],
        "k": res["k"],
        "silhouette": res["silhouette"],
    }
    ruta_modelo = R.joblib_modelo(pos)
    joblib.dump(bundle, ruta_modelo)

    if verbose:
        print(f"{P.NOMBRE_POS[pos]:15s} -> {ruta_csv.name}, "
              f"{ruta_perfil.name}, {ruta_modelo.name}")

    return bundle


def comparar_modelos(modelos: dict) -> pd.DataFrame:
    """Tabla comparativa de los cuatro modelos (sección 8 del notebook)."""
    return pd.DataFrame([
        {
            "posición": P.NOMBRE_POS[p],
            "n_etiquetados": len(modelos[p]["df"]),
            "n_entrenan": int(modelos[p]["entrena"].sum()),
            "n_variables": len(P.FEATURES[p]),
            "k": modelos[p]["k"],
            "sil_entren.": round(modelos[p]["silhouette"], 3),
            "sil_todos": round(modelos[p]["silhouette_todos"], 3),
            "cluster más grande": int(modelos[p]["df"]["cluster"].value_counts().max()),
            "cluster más pequeño": int(modelos[p]["df"]["cluster"].value_counts().min()),
        }
        for p in P.POSICIONES if p in modelos
    ])


def clusters_de_polivalentes(modelos: dict, polivalentes: pd.DataFrame) -> pd.DataFrame:
    """Cada polivalente con su cluster en cada uno de los modelos que lo evaluaron.

    Un jugador "DF,MF" tiene un cluster como defensa y otro como mediocampista. Esta
    tabla los pone lado a lado: es la lectura más útil del enfoque por posición, porque
    muestra qué rol cumple el mismo futbolista según desde qué línea se le mire.
    """
    filas = []
    for _, jug in polivalentes.iterrows():
        fila = {"player": jug["player"], "team": jug["team"], "pos": jug["pos"]}
        for p in P.POSICIONES:
            if p not in modelos:
                continue
            d = modelos[p]["df"]
            # player + team: hay nombres repetidos entre clubes y ligas.
            coincide = d.loc[(d["player"] == jug["player"])
                             & (d["team"] == jug["team"]), "cluster"]
            fila[f"cluster_{p}"] = int(coincide.iloc[0]) if len(coincide) else None
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    # Ningún polivalente es portero, así que la columna cluster_GK sale entera vacía: se
    # quita. Las demás se dejan como enteros que admiten nulos (un "DF,MF" no tiene
    # cluster de delantero).
    vacias = [c for c in tabla.columns if tabla[c].isna().all()]
    tabla = tabla.drop(columns=vacias)
    for c in [c for c in tabla.columns if c.startswith("cluster_")]:
        tabla[c] = tabla[c].astype("Int64")

    return tabla
