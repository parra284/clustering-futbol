# -*- coding: utf-8 -*-
"""Modelo de delanteros (FW) — K-Means, k=5, 10 variables.

Ejes: finalización (G-PK/90, Sh/90, SoT%), movimiento de ruptura (Off/90), juego de
banda (Crs/90, Fld/90) y trabajo sin balón (Fls/90, Int/90, TklW/90). El k más alto de
los cuatro: dos familias —extremos y nueves— más un perfil que las cruza.

Resultado de referencia: silueta 0.132 (385 de entrenamiento) · 0.122 (496 etiquetados).

    python -m models.modelo_fw
"""
from config import parametros as P
from config import rutas as R
from src import datos, modelado, preparacion

POS = "FW"


def main(datasets=None, graficar=True, guardar=True):
    """Ejecuta las fases 3, 4, 5 y 6 para delanteros. Devuelve el dict de resultados."""
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
