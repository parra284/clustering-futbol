# -*- coding: utf-8 -*-
"""Modelo de mediocampistas (MF) — K-Means, k=4, 10 variables.

Tensan el eje destrucción ↔ creación ↔ llegada, donde vive la variedad de roles del
mediocampo: pivote destructor, interior creador, carrilero y mediapunta llegador
conviven bajo la misma etiqueta de FBref.

`SoT%` se probó y se retiró por el test de ablación (ARI 0.91 al quitarla).

Es el k más alto de los tres modelos de campo y la silueta más baja de los cuatro
(0.145), algo esperable en la línea con el abanico de roles más ancho.

Resultado de referencia: silueta 0.145 (887 de entrenamiento) · 0.135 (1097 etiquetados).

    python -m models.modelo_mf
"""
from config import parametros as P
from config import rutas as R
from src import datos, modelado, preparacion

POS = "MF"


def main(datasets=None, graficar=True, guardar=True):
    """Ejecuta las fases 3, 4, 5 y 6 para mediocampistas. Devuelve el dict de resultados."""
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
