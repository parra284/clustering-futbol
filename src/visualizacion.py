# -*- coding: utf-8 -*-
"""Todas las figuras del proyecto, con el mismo tema visual.

Cada función acepta `guardar: Path | None`. Si se le pasa una ruta, escribe el PNG
en `reports/figuras/`; dentro de Jupyter además la muestra inline, y fuera cierra la
figura para que un script no se quede bloqueado en `plt.show()`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registra la proyección '3d'
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import PowerTransformer, StandardScaler

from config import parametros as P
from config import rutas as R
from config.estilo_viz import (DPI_GUARDADO, GRIS, PALETA_POS, TINTA, TINTA2,
                               aplicar_estilo)
from src.preparacion import _es_tasa, _etiqueta, preparar_features
from src.utils import en_notebook, mostrar

aplicar_estilo()


def _cerrar(fig, guardar):
    """Guarda la figura si se pidió, la muestra en Jupyter y la cierra fuera de él."""
    if guardar is not None:
        R.FIGURAS.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar, dpi=DPI_GUARDADO, bbox_inches="tight")
    if en_notebook():
        plt.show()
    else:
        plt.close(fig)


# ======================================================================================
# Fase 4 · elección de k
# ======================================================================================

def elegir_k(X, titulo="", k_min=None, k_max=None, random_state=None,
             entrena=None, guardar=None):
    """Fase 4: método del codo (inercia) + coeficiente de silueta para un rango de k.

    Se calcula solo sobre los jugadores de entrenamiento: el `k` debe elegirse sobre la
    estructura que forman los perfiles bien medidos.

    Devuelve el k con mejor silueta, pero SOLO como referencia: la silueta suele premiar
    particiones muy gruesas (k=2), y para scouting casi siempre son más útiles unos
    cuantos perfiles más aunque estén algo peor separados. La decisión final es
    futbolística y está fijada en `config.parametros.K_ELEGIDO`.
    """
    k_min = P.K_MIN if k_min is None else k_min
    k_max = P.K_MAX if k_max is None else k_max
    random_state = P.RANDOM_STATE if random_state is None else random_state

    if entrena is not None:
        X = X[entrena]
    k_max = min(k_max, len(X) - 1)
    rango = list(range(k_min, k_max + 1))
    inercias, siluetas = [], []

    for k in rango:
        km = KMeans(n_clusters=k, n_init=P.N_INIT, random_state=random_state)
        etiquetas = km.fit_predict(X)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X, etiquetas))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(rango, inercias, marker="o")
    axes[0].set_title(f"Método del codo — {titulo}")
    axes[0].set_xlabel("k (número de clusters)")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(rango, siluetas, marker="o", color="darkorange")
    axes[1].set_title(f"Coeficiente de silueta — {titulo}")
    axes[1].set_xlabel("k (número de clusters)")
    axes[1].set_ylabel("Silhouette score")
    plt.tight_layout()
    _cerrar(fig, guardar)

    tabla = pd.DataFrame({"k": rango,
                          "inercia": np.round(inercias, 1),
                          "silueta": np.round(siluetas, 3)})
    print(tabla.to_string(index=False))

    mejor = rango[int(np.argmax(siluetas))]
    print(f"\nk con mejor silueta (solo referencia): {mejor}")
    return mejor


# ======================================================================================
# Fase 5 · proyección del resultado
# ======================================================================================

def proyectar_pca(pca, X_escalado, etiquetas, centros, k, titulo,
                  entrena=None, guardar=None):
    """Dibuja la proyección PCA a 3 componentes de un clustering ya ajustado.

    Tres componentes son el máximo representable en una gráfica y recogen bastante más
    varianza que dos, así que dos clusters que en el plano se solapan pueden verse
    separados en profundidad. El modelo sigue trabajando en el espacio completo de
    variables: la PCA es solo la ventana por la que se mira.

    Se dibuja la nube en 3D con los centroides marcados y, debajo, los tres planos
    (PC1-PC2, PC1-PC3, PC2-PC3): una vista 3D estática esconde puntos por oclusión y
    engaña con la profundidad, y los planos permiten leer sin ambigüedad.

    El objeto PCA llega ya ajustado desde `src.modelado`, que lo necesita para el bundle
    exportado: aquí solo se proyecta y se dibuja.
    """
    if entrena is None:
        entrena = np.ones(len(X_escalado), dtype=bool)
    n_comp = pca.n_components_
    coords = pca.transform(X_escalado)
    # Los centroides llegan como ndarray; se les devuelven los nombres de columna para
    # que sklearn no avise de que la PCA se ajustó con nombres y aquí no los recibe.
    cen = pca.transform(pd.DataFrame(centros, columns=X_escalado.columns))
    var = pca.explained_variance_ratio_

    paleta = plt.get_cmap("tab10")
    color_pt = [paleta(c % 10) for c in etiquetas]
    ejes = [f"PC{i+1} ({var[i]:.0%})" for i in range(n_comp)]

    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.7, 1], hspace=0.05, wspace=0.30)

    # --- Nube en 3D ---
    ax = fig.add_subplot(gs[0, :], projection="3d")
    col_arr = np.array(color_pt)
    ax.scatter(coords[entrena, 0], coords[entrena, 1], coords[entrena, 2],
               c=col_arr[entrena], s=28, alpha=0.80, edgecolor="none")
    # Los de confianza baja se dibujan más pequeños y translúcidos: reciben etiqueta pero
    # no participaron en definir ni los centroides ni los ejes.
    if (~entrena).any():
        ax.scatter(coords[~entrena, 0], coords[~entrena, 1], coords[~entrena, 2],
                   c=col_arr[~entrena], s=11, alpha=0.35, edgecolor="none")
    ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], c="black", marker="X",
               s=190, edgecolor="white", linewidth=1.4, depthshade=False)
    ax.set_xlabel(ejes[0]); ax.set_ylabel(ejes[1]); ax.set_zlabel(ejes[2])
    ax.set_title(f"{titulo} — k={k}\n"
                 f"{var.sum():.0%} de la varianza en estas {n_comp} componentes")
    ax.view_init(elev=18, azim=48)
    # Las axes 3D dejan mucho margen interno; el zoom agranda el cubo dentro de su hueco.
    ax.set_box_aspect(None, zoom=1.30)

    marcas = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=paleta(c % 10),
                     markeredgecolor="none", markersize=8, label=f"cluster {c}")
              for c in range(k)]
    marcas.append(Line2D([0], [0], marker="X", linestyle="", markerfacecolor="black",
                         markeredgecolor="white", markersize=10, label="centroide"))
    if (~entrena).any():
        marcas.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="gray",
                             markeredgecolor="none", markersize=5, alpha=0.4,
                             label="confianza baja"))
    ax.legend(handles=marcas, loc="upper left", fontsize=9, framealpha=0.9)

    # --- Los tres planos, para leer sin oclusión ---
    for j, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax2 = fig.add_subplot(gs[1, j])
        ax2.scatter(coords[entrena, a], coords[entrena, b], c=col_arr[entrena], s=14,
                    alpha=0.75, edgecolor="none")
        if (~entrena).any():
            ax2.scatter(coords[~entrena, a], coords[~entrena, b], c=col_arr[~entrena], s=7,
                        alpha=0.30, edgecolor="none")
        ax2.scatter(cen[:, a], cen[:, b], c="black", marker="X", s=90,
                    edgecolor="white", linewidth=1.0)
        ax2.set_xlabel(ejes[a], fontsize=9)
        ax2.set_ylabel(ejes[b], fontsize=9)
        ax2.tick_params(labelsize=8)

    _cerrar(fig, guardar)

    # Cuánto aporta cada componente por separado.
    resumen = pd.DataFrame({
        "varianza explicada": [f"{v:.1%}" for v in var],
        "acumulada": [f"{v:.1%}" for v in np.cumsum(var)],
    }, index=[f"PC{i+1}" for i in range(n_comp)])
    mostrar(resumen)

    # Cargas: cuánto pesa cada variable original en cada componente. Es lo que convierte
    # los ejes en algo con nombre futbolístico ("PC1 = eje defensa-ataque", etc.).
    cargas = pd.DataFrame(pca.components_.T,
                          index=X_escalado.columns,
                          columns=[f"PC{i+1}" for i in range(n_comp)])
    print("Cargas de cada variable en las componentes (signo = dirección del eje):")
    mostrar(cargas.round(2))


# ======================================================================================
# Fase 3 · Decisión 1 — el umbral de minutos
# ======================================================================================

def _clusters_con_umbral(datasets, pos, partidos):
    """Repite el pipeline completo de la posición con otro umbral de entrenamiento.
    Devuelve la etiqueta de cada jugador, indexada por (player, team)."""
    d = datasets[pos].copy()
    d["entrena"] = d["Playing Time_Min"] >= partidos * P.PARTIDO
    if d["entrena"].sum() < P.K_ELEGIDO[pos] * 5:      # muestra insuficiente para ese k
        return None
    Xs, _, _ = preparar_features(d, P.FEATURES[pos], verbose=False)
    km = KMeans(P.K_ELEGIDO[pos], n_init=P.N_INIT,
                random_state=P.RANDOM_STATE).fit(Xs[d["entrena"].values])
    return pd.Series(km.predict(Xs), index=pd.MultiIndex.from_frame(d[["player", "team"]]))


def figura_umbral(df_base, datasets, guardar=None):
    """Evidencia de la decisión 1: a quién deja fuera la regla y si mover el corte
    cambia la partición (ARI contra el umbral elegido)."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    fig.subplots_adjust(wspace=0.28)

    # --- (a) a quién deja cada tramo de la regla ---
    a = ax[0]
    ys = np.arange(len(P.POSICIONES))[::-1]
    for y, p in zip(ys, P.POSICIONES):
        total_pos = int(df_base["pos"].str.contains(p).sum())
        ent = int(datasets[p]["entrena"].sum())
        baja = len(datasets[p]) - ent
        fuera = total_pos - len(datasets[p])
        a.barh(y, ent, color=PALETA_POS[p], height=0.55)
        a.barh(y, baja, left=ent + 14, color=GRIS, height=0.55)
        a.barh(y, fuera, left=ent + baja + 28, color="#eeede9", height=0.55)
        if ent > 200:
            a.annotate(f"{ent}", (ent / 2, y), ha="center", va="center",
                       fontsize=8.5, color="white", weight="bold")
        else:
            a.annotate(f"{ent}", (ent + 22, y), va="center", fontsize=8, color=TINTA2)
        a.annotate(f"≥{P.MINUTOS_ENTRENAMIENTO[p] // P.PARTIDO} part.", (-30, y),
                   ha="right", va="center", fontsize=7.5, color=TINTA2)
    a.set_yticks(ys); a.set_yticklabels([P.NOMBRE_POS[p] for p in P.POSICIONES], fontsize=9)
    a.set_xlabel("jugadores"); a.set_xlim(-280, 1750); a.grid(axis="y", visible=False)
    a.legend(handles=[Line2D([0], [0], marker="s", linestyle="", markersize=8,
                             markerfacecolor=c, markeredgecolor="none", label=t)
                      for c, t in [(PALETA_POS["MF"], "entrenan (definen los perfiles)"),
                                   (GRIS, "solo etiquetados (confianza baja)"),
                                   ("#eeede9", f"fuera (< {P.MINUTOS_INCLUSION // P.PARTIDO} partidos)")]],
             frameon=False, fontsize=8, loc="lower right")
    a.set_title("a · A quién deja cada tramo de la regla", loc="left", fontsize=10, color=TINTA)

    # --- (b) ¿cambia el resultado si movemos el umbral? ---
    a = ax[1]
    for p in P.POSICIONES:
        ref = _clusters_con_umbral(datasets, p, P.MINUTOS_ENTRENAMIENTO[p] // P.PARTIDO)
        curva = []
        for q in P.RANGO_UMBRALES:
            alt = _clusters_con_umbral(datasets, p, q)
            comun = ref.index.intersection(alt.index)
            curva.append(adjusted_rand_score(ref.loc[comun], alt.loc[comun]))
        a.plot(P.RANGO_UMBRALES, curva, marker="o", ms=6, lw=2,
               color=PALETA_POS[p], label=P.NOMBRE_POS[p])
    a.axhspan(0.85, 1.05, color=GRIS, alpha=0.3)
    a.annotate("misma partición", (P.RANGO_UMBRALES[0], 0.86), fontsize=8,
               color=TINTA2, va="bottom")
    a.axvspan(8, 12, color=PALETA_POS["MF"], alpha=0.07)
    a.annotate("±2 partidos", (10, 1.005), ha="center", fontsize=8, color=TINTA2)
    for _x, _txt in [(10, "10 part.\n(campo)"), (15, "15 part.\n(porteros)")]:
        a.axvline(_x, color=TINTA2, lw=1, ls="--")
        a.annotate(_txt, (_x, 0.42), ha="center", fontsize=8, color=TINTA2)
    a.set_xticks(P.RANGO_UMBRALES)
    a.set_xlabel("umbral de entrenamiento (partidos completos)")
    a.set_ylabel("ARI contra el umbral elegido"); a.set_ylim(0.37, 1.04)
    a.legend(frameon=False, fontsize=8.5, loc="lower left")
    a.set_title("b · Estabilidad al mover el corte", loc="left", fontsize=10, color=TINTA)

    fig.suptitle("Fase 3 · El umbral de minutos: la regla y su sensibilidad",
                 fontsize=12.5, y=1.0, color=TINTA)
    _cerrar(fig, guardar)


# ======================================================================================
# Fase 3 · Decisión 2 — el escalado
# ======================================================================================

def figura_escalado(datasets, pos_demo=None, guardar=None):
    """Evidencia de la decisión 2: StandardScaler deja una cola que domina la distancia;
    Yeo-Johnson la comprime y ninguna variable manda sobre las demás."""
    pos_demo = P.POS_DEMO_ESCALADO if pos_demo is None else pos_demo

    d = datasets[pos_demo]; ent = d["entrena"].values
    cols = P.FEATURES[pos_demo]
    conteos = [c for c in cols if not _es_tasa(c)]

    X_raw = d[cols].astype(float).copy()
    X_raw[conteos] = X_raw[conteos].div(d[P.COL_90S], axis=0)
    X_raw = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_raw), columns=cols)
    etiq = [_etiqueta(c, c in conteos) for c in cols]   # mismos nombres cortos que el modelo

    # Ambos escaladores se ajustan con el subconjunto de entrenamiento, como en el modelo real.
    Z_std = pd.DataFrame(StandardScaler().fit(X_raw[ent]).transform(X_raw), columns=etiq)
    Z_yeo = pd.DataFrame(PowerTransformer("yeo-johnson", standardize=True)
                         .fit(X_raw[ent]).transform(X_raw), columns=etiq)
    peor = Z_std.abs().max().idxmax()

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    fig.subplots_adjust(wspace=0.32)

    for a, (Z, nombre, letra) in zip(ax[:2], [(Z_std, "StandardScaler", "a"),
                                              (Z_yeo, "Yeo-Johnson", "b")]):
        a.hist(Z[peor], bins=34, color=PALETA_POS[pos_demo], alpha=0.85)
        a.set_xlabel("desviaciones desde la media")
        a.set_title(f"{letra} · {peor} con {nombre}", loc="left", fontsize=10, color=TINTA)
        a.annotate(f"máx |z| = {Z[peor].abs().max():.1f}", (0.97, 0.88),
                   xycoords="axes fraction", ha="right", fontsize=8.5, color=TINTA)
    ax[0].set_ylabel(P.NOMBRE_POS[pos_demo].lower())

    a = ax[2]
    m_std, m_yeo = Z_std.abs().max(), Z_yeo.abs().max()
    orden = m_std.sort_values().index
    ys = np.arange(len(orden))
    a.hlines(ys, m_yeo[orden], m_std[orden], color=GRIS, lw=2, zorder=1)
    a.scatter(m_std[orden], ys, s=52, color=TINTA2, zorder=3, label="StandardScaler")
    a.scatter(m_yeo[orden], ys, s=52, color=PALETA_POS[pos_demo], zorder=3, label="Yeo-Johnson")
    a.set_yticks(ys); a.set_yticklabels(orden, fontsize=8.5)
    a.set_xlabel("máx |z| — cuánto puede tirar el jugador más extremo")
    a.set_title("c · Ninguna variable domina la distancia", loc="left", fontsize=10, color=TINTA)
    a.legend(frameon=False, fontsize=8.5, loc="lower right"); a.grid(axis="y", visible=False)

    fig.suptitle("Fase 3 · Por qué el escalado es Yeo-Johnson y no una estandarización simple",
                 fontsize=12.5, y=1.04, color=TINTA)
    _cerrar(fig, guardar)


# ======================================================================================
# Fase 3 · Decisión 3 — qué variables entran
# ======================================================================================

def figura_ablacion(datasets, guardar=None):
    """Evidencia de la decisión 3: se quita cada variable y se mide cuánto cambia la
    partición (ARI). Por encima de UMBRAL_ARI, la variable no aporta.

    El conjunto candidato son las variables del modelo más las que se probaron y
    quedaron fuera por este mismo test, para que la figura muestre la decisión y no
    solo su resultado.
    """
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.subplots_adjust(wspace=0.55)

    for a, p in zip(ax, P.POSICIONES):
        entrena = datasets[p]["entrena"].values
        k = P.K_ELEGIDO[p]
        X_cand, _, _ = preparar_features(datasets[p], P.FEATURES[p] + P.RETIRADAS[p],
                                         verbose=False)
        Z = X_cand[entrena]
        referencia = KMeans(k, n_init=P.N_INIT,
                            random_state=P.RANDOM_STATE).fit_predict(Z)
        ari = {c: adjusted_rand_score(referencia,
                                      KMeans(k, n_init=P.N_INIT,
                                             random_state=P.RANDOM_STATE)
                                      .fit_predict(Z.drop(columns=[c]))) for c in Z.columns}
        s = pd.Series(ari).sort_values()
        ys = np.arange(len(s))
        a.barh(ys, s.values, height=0.62,
               color=[GRIS if v > P.UMBRAL_ARI else PALETA_POS[p] for v in s.values])
        a.axvline(P.UMBRAL_ARI, color=TINTA2, lw=1.2, ls="--")
        for y, v in zip(ys, s.values):
            a.annotate(f"{v:.2f}", (v, y), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=8, color=TINTA2)
        a.set_yticks(ys); a.set_yticklabels(s.index, fontsize=8.5)
        a.set_xlim(0, 1.2); a.set_xticks([0, 0.5, P.UMBRAL_ARI])
        a.set_title(f"{P.NOMBRE_POS[p]} (k={k})", loc="left", fontsize=10, color=TINTA)
        a.grid(axis="y", visible=False)
    ax[0].set_xlabel("ARI al quitar la variable")

    fig.suptitle("Fase 3 · Qué aporta cada variable: quitarla y medir cuánto cambia la partición",
                 fontsize=12.5, y=1.10, color=TINTA)
    fig.text(0.5, 1.015,
             f"gris = ARI por encima de {P.UMBRAL_ARI:.2f}: la partición no cambia, "
             f"la variable no aporta",
             ha="center", fontsize=9, color=TINTA2)
    _cerrar(fig, guardar)


# ======================================================================================
# Secciones 4 a 7 · por qué quedó fuera cada candidata
# ======================================================================================

def figura_descartadas(pos, df_pos, guardar=None):
    """Compara las variables del modelo con las candidatas descartadas en los dos
    criterios que se pueden medir:

      · Redundancia — máx |correlación| con alguna variable que sí entró. Si supera a
        todas las incluidas, la candidata dice casi lo mismo que una que ya está.
      · Escasez — % de jugadores con cero en toda la temporada. Si supera a todas las
        incluidas, no hay sucesos suficientes para que la tasa signifique algo.

    La banda sombreada es el rango de las variables del modelo: una barra gris que la
    rebasa está descartada por ese criterio. Las candidatas que caben dentro de ambas
    bandas se descartaron por otra razón —no aportar a la partición (sección 3.3) o por
    criterio conceptual—, y se indica en el texto de cada sección.
    """
    d = df_pos
    entrena = d["entrena"].values
    desc = {c: n for c, n in P.DESCARTADAS[pos].items() if c in d.columns}
    cols = list(P.FEATURES[pos]) + list(desc)
    lab = [_etiqueta(c, not _es_tasa(c)) for c in P.FEATURES[pos]] + list(desc.values())

    X = d[cols].astype(float).copy()
    for c in cols:
        if not _es_tasa(c):
            X[c] = X[c] / d[P.COL_90S]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=lab)
    Z = pd.DataFrame(PowerTransformer("yeo-johnson", standardize=True)
                     .fit(X[entrena]).transform(X), columns=lab)[entrena]

    dentro = lab[:len(P.FEATURES[pos])]
    redundancia = {n: Z[[c for c in dentro if c != n]].corrwith(Z[n]).abs().max() for n in lab}
    escasez = {n: (X[n][entrena] == 0).mean() * 100 for n in lab}

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    fig.subplots_adjust(wspace=0.42)
    for a, (datos, titulo, xlab, fmt) in zip(ax, [
            (redundancia, "a · Redundancia",
             "máx |correlación| con una variable del modelo", "{:.2f}"),
            (escasez, "b · Escasez",
             "% de jugadores con cero en toda la temporada", "{:.0f}%")]):
        s = pd.Series(datos).sort_values()
        tope = max(v for k, v in datos.items() if k in dentro)
        ys = np.arange(len(s))
        a.axvspan(0, tope, color=PALETA_POS[pos], alpha=0.07)
        a.barh(ys, s.values, height=0.62,
               color=[PALETA_POS[pos] if k in dentro else GRIS for k in s.index])
        a.axvline(tope, color=TINTA2, lw=1.2, ls="--")
        for y, (k, v) in zip(ys, s.items()):
            rebasa = (k not in dentro) and v > tope
            a.annotate(fmt.format(v), (v, y), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=7.5, color=TINTA if rebasa else TINTA2,
                       weight="bold" if rebasa else "normal")
        a.set_yticks(ys); a.set_yticklabels(s.index, fontsize=8)
        a.set_title(titulo, loc="left", fontsize=10, color=TINTA)
        a.set_xlabel(xlab, fontsize=8.5)
        a.grid(axis="y", visible=False)
        a.set_xlim(0, max(s.values) * 1.24)
        a.annotate("rango del modelo →", (tope, len(s) - 0.3), xytext=(-5, 0),
                   textcoords="offset points", ha="right", fontsize=7.5, color=TINTA2)

    fig.suptitle(f"{P.NOMBRE_POS[pos]} · por qué quedó fuera cada candidata"
                 f"   (color = en el modelo · gris = descartada)",
                 fontsize=11.5, y=1.03, color=TINTA)
    _cerrar(fig, guardar)
