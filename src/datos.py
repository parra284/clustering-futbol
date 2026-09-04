# -*- coding: utf-8 -*-
"""Fase 2 de CRISP-DM: recolección y limpieza base de los datos.

Antes había dos copias de este scraper (`cargar_datos.py` en la raíz y la celda 3
del notebook). Esta es la única versión.
"""
import pandas as pd

from config import parametros as P
from config import rutas as R


def recolectar_datos_permitidos(forzar_descarga: bool = False) -> pd.DataFrame:
    """Descarga (o carga desde caché) estadísticas de jugadores de FBref
    para las ligas y temporada configuradas en `config.parametros`."""

    if R.CSV_CRUDO.exists() and not forzar_descarga:
        print(f"📂 Cargando datos desde caché local: '{R.CSV_CRUDO.name}'")
        return pd.read_csv(R.CSV_CRUDO)

    # Import perezoso: soccerdata solo hace falta si de verdad hay que descargar.
    import soccerdata as sd

    print("Inicializando conexión con FBref...")
    fbref = sd.FBref(leagues=P.FBREF_LIGAS, seasons=P.FBREF_TEMPORADAS)

    lista_dfs = []
    for stat in P.FBREF_TABLAS:
        print(f"Descargando estadísticas de tipo: '{stat}'...")
        try:
            lista_dfs.append(fbref.read_player_season_stats(stat_type=stat))
        except Exception as e:
            print(f"⚠️ Error descargando {stat}: {e}")

    print("\nUniendo todas las métricas en una sola tabla...")
    df_completo = pd.concat(lista_dfs, axis=1)

    # Eliminamos columnas duplicadas (como edad, minutos o equipo que vienen repetidas).
    # Nota: esto solo detecta las que comparten nombre exacto; las que FBref repite con
    # nombre distinto en cada tabla (p. ej. Gls en 'standard' y en 'shooting') se
    # descartan más adelante, en la fase 3.
    df_completo = df_completo.loc[:, ~df_completo.columns.duplicated()]
    df_completo = df_completo.reset_index()

    # Aplanamos los nombres de las columnas para evitar problemas en pandas
    df_completo.columns = [
        "_".join([str(c) for c in col if c]).strip() if isinstance(col, tuple) else col
        for col in df_completo.columns.values
    ]

    print("\n--- VOLUMEN DE DATOS ---")
    print(f"Jugadores procesados: {len(df_completo)}")
    print(f"Características (features) obtenidas: {len(df_completo.columns)}")

    R.asegurar_carpetas()
    df_completo.to_csv(R.CSV_CRUDO, index=False, encoding=P.ENCODING_CSV)
    print(f"\n✅ ¡Los datos se han guardado en '{R.CSV_CRUDO}'!")
    print("\n🔍 Primeras métricas obtenidas:", list(df_completo.columns[:12]))

    return df_completo


def cargar_base(df_raw: pd.DataFrame | None = None, guardar: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """Limpieza base compartida por los cuatro modelos.

    Dos arreglos, ambos sobre defectos del origen y no sobre el modelado:

    1. La tabla combinada de las 5 grandes ligas llega con el bloque alemán sin
       etiquetar: 'league' viene vacío para los 18 equipos de la Bundesliga, aunque
       sus filas están completas (posición, minutos y todas las métricas). Sin este
       relleno esos jugadores entran igual a los modelos —el filtro es por posición y
       minutos, no por liga— pero aparecen sin liga en las tablas exportadas.
    2. Las filas sin 'pos' son jugadores de plantilla sin un solo minuto disputado:
       FBref los lista pero llegan con todas las métricas vacías.

    El resultado se persiste en `data/processed/jugadores_base.csv`: es la salida de
    la fase 2 y la entrada de los cuatro modelos.
    """
    if df_raw is None:
        df_raw = recolectar_datos_permitidos()

    df_base = df_raw.copy()
    n_inicial = len(df_base)

    sin_liga = df_base["league"].isna()
    if sin_liga.any():
        equipos = sorted(df_base.loc[sin_liga, "team"].dropna().unique())
        df_base.loc[sin_liga, "league"] = P.LIGA_FALTANTE
        if verbose:
            print(f"Etiquetados como {P.LIGA_FALTANTE}: {len(equipos)} equipos, "
                  f"{int(sin_liga.sum())} filas ({', '.join(equipos[:4])}...)")

    sin_datos = int(df_base["pos"].isna().sum())
    df_base = df_base[df_base["pos"].notna()].reset_index(drop=True)

    if verbose:
        print(f"Jugadores en el CSV:                    {n_inicial}")
        print(f"  - sin posición / sin minutos jugados: {sin_datos}")
        print(f"Jugadores con posición registrada:      {len(df_base)}")

    if guardar:
        R.asegurar_carpetas()
        df_base.to_csv(R.CSV_BASE, index=False, encoding=P.ENCODING_CSV)

    return df_base
