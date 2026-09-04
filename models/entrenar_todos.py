# -*- coding: utf-8 -*-
"""Punto de entrada: ejecuta el pipeline completo de las cuatro posiciones.

    python -m models.entrenar_todos                 # con figuras
    python -m models.entrenar_todos --sin-figuras   # solo los CSV y los .joblib

La base limpia y los cuatro datasets se construyen una sola vez y se pasan a cada
modelo: son idénticos para los cuatro, y recalcularlos cuatro veces solo añade tiempo.
"""
import argparse

from config import parametros as P
from config import rutas as R
from src import datos, modelado, preparacion
from src.utils import mostrar

from models import modelo_df, modelo_fw, modelo_gk, modelo_mf

ENTRENADORES = {"GK": modelo_gk, "DF": modelo_df, "MF": modelo_mf, "FW": modelo_fw}


def main(posiciones=None, graficar=True, guardar=True):
    """Entrena las posiciones pedidas (todas por defecto) y compara los resultados."""
    posiciones = P.POSICIONES if posiciones is None else posiciones

    # --- Fase 2: recolección y limpieza base ---
    df_base = datos.cargar_base()
    datasets = preparacion.construir_datasets(df_base)
    _, polivalentes = preparacion.universo_analizado(df_base, datasets)

    if graficar:
        from src import visualizacion
        visualizacion.figura_umbral(df_base, datasets,
                                    guardar=R.figura("31_umbral_minutos.png"))
        visualizacion.figura_escalado(datasets, guardar=R.figura("32_escalado.png"))
        visualizacion.figura_ablacion(datasets,
                                      guardar=R.figura("33_ablacion_variables.png"))

    # --- Fases 3 a 6, una posición a la vez ---
    modelos = {}
    for pos in posiciones:
        print(f"\n{'=' * 86}\n{P.NOMBRE_POS[pos].upper()} ({pos})\n{'=' * 86}")
        modelos[pos] = ENTRENADORES[pos].main(datasets=datasets, graficar=graficar,
                                              guardar=guardar)

    # --- Sección 8: comparación de los cuatro modelos ---
    print(f"\n{'=' * 86}\nCOMPARACIÓN DE LOS {len(modelos)} MODELOS\n{'=' * 86}")
    mostrar(modelado.comparar_modelos(modelos))
    total = sum(modelos[p]["k"] for p in modelos)
    print(f"\nTotal de perfiles identificados: {total}")

    tabla_poli = modelado.clusters_de_polivalentes(modelos, polivalentes)
    print(f"\nJugadores polivalentes con doble etiqueta: {len(tabla_poli)}")
    mostrar(tabla_poli.head(15))

    if guardar:
        print(f"\n✅ CSV y perfiles en '{R.REPORTS}'")
        print(f"✅ Modelos en '{R.MODELOS_ENTRENADOS}'")
        if graficar:
            print(f"✅ Figuras en '{R.FIGURAS}'")

    return modelos


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pos", choices=P.POSICIONES, action="append",
                    help="entrenar solo esta posición (repetible)")
    ap.add_argument("--sin-figuras", action="store_true",
                    help="no generar las gráficas (más rápido)")
    ap.add_argument("--sin-guardar", action="store_true",
                    help="no escribir CSV ni .joblib")
    args = ap.parse_args()

    main(posiciones=args.pos, graficar=not args.sin_figuras,
         guardar=not args.sin_guardar)
