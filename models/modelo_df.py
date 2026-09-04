# -*- coding: utf-8 -*-
"""Modelo de defensas (DF) — K-Means, k=3, 8 variables (todas por 90 minutos).

Ejes: acción defensiva (Int/90, TklW/90), roce y disciplina (Fls/90, Fld/90, CrdY/90)
y aporte ofensivo (Crs/90, Ast/90, Sh/90). El eje dominante es central vs. lateral, y
los centros por 90 son su marcador más nítido.

`G-PK`/90 se probó y se retiró: dejaba la partición prácticamente idéntica (ARI 0.91)
y bajaba la silueta, así que solo añadía una dimensión de ruido a la distancia.

Resultado de referencia: silueta 0.172 (645 de entrenamiento) · 0.161 (779 etiquetados).

    python -m models.modelo_df
"""
from config import parametros as P
from config import rutas as R
from src import datos, modelado, preparacion

POS = "DF"


def main(datasets=None, graficar=True, guardar=True):
    """Ejecuta las fases 3, 4, 5 y 6 para defensas. Devuelve el dict de resultados."""
    if datasets is None:
        datasets = preparacion.construir_datasets(datos.cargar_base())
    d = datasets[POS]

    # --- FASE 3 — preparación del dataset ---
    print(f"Dataset {P.NOMBRE_POS[POS]}: {len(d)} jugadores "
          f"({int(d['entrena'].sum())} de entrenamiento)")
    X, X_90, prep = preparacion.preparar_features(d, P.FEATURES[POS])

    if graficar:
        from src import visualizacion
        visualizacion.figura_descartadas(
            POS, d, guardar=R.figura(f"descartadas_{POS}.png"))
        # FASE 4 — ¿cuántos clusters? El k de mejor silueta es solo referencia.
        k_ref = visualizacion.elegir_k(
            X, titulo=P.NOMBRE_POS[POS], entrena=prep["entrena"],
            guardar=R.figura(f"eleccion_k_{POS}.png"))
        print(f"k sugerido por silueta: {k_ref}  |  k elegido: {P.K_ELEGIDO[POS]}\n")

    # --- FASES 4 y 5 — modelo final y evaluación ---
    res = modelado.ajustar_y_evaluar(d, X, X_90, P.K_ELEGIDO[POS], POS,
                                     graficar=graficar,
                                     guardar_fig=R.figura(f"pca_{POS}.png"))

    # --- FASE 6 — despliegue ---
    if guardar:
        modelado.exportar(POS, res, prep)

    return res


if __name__ == "__main__":
    main()
