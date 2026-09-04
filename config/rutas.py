# -*- coding: utf-8 -*-
"""Rutas del proyecto, todas ancladas a la ubicación de este archivo.

Ninguna ruta del proyecto es relativa al directorio de trabajo: un notebook que
vive en `notebooks/` y un script que se lanza desde la raíz tienen cwd distintos,
y con rutas relativas cada uno escribiría en un sitio diferente. Anclando en
`__file__` la respuesta es la misma se ejecute desde donde se ejecute.
"""
from pathlib import Path

# config/rutas.py -> config/ -> raíz del proyecto
RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

REPORTS = RAIZ / "reports"
FIGURAS = REPORTS / "figuras"

MODELOS_ENTRENADOS = RAIZ / "models" / "entrenados"

# --- Archivos concretos ---------------------------------------------------------------
# El nombre lleva la temporada dentro. Con un nombre fijo, cambiar la temporada en
# config/parametros.py no habría disparado la descarga: `recolectar_datos_permitidos()`
# habría encontrado el archivo antiguo, lo habría dado por bueno y se habría seguido
# entrenando con la temporada anterior sin un solo aviso.
def _sufijo_temporada() -> str:
    from config import parametros
    return "_".join(parametros.FBREF_TEMPORADAS)


CSV_CRUDO = DATA_RAW / f"datos_jugadores_{_sufijo_temporada()}.csv"
CSV_BASE = DATA_PROCESSED / f"jugadores_base_{_sufijo_temporada()}.csv"


def csv_jugadores(pos: str) -> Path:
    """Dataset de una posición con su etiqueta de cluster."""
    return REPORTS / f"jugadores_{pos}_con_cluster.csv"


def csv_perfil(pos: str) -> Path:
    """Tabla de perfiles medios de los clusters de una posición."""
    return REPORTS / f"perfil_clusters_{pos}.csv"


def joblib_modelo(pos: str) -> Path:
    """Bundle serializado (imputer + scaler + kmeans + pca) de una posición."""
    return MODELOS_ENTRENADOS / f"modelo_{pos}.joblib"


def figura(nombre: str) -> Path:
    """Ruta de una figura dentro de reports/figuras/."""
    return FIGURAS / nombre


def asegurar_carpetas() -> None:
    """Crea las carpetas de salida si no existen. Idempotente."""
    for carpeta in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURAS, MODELOS_ENTRENADOS):
        carpeta.mkdir(parents=True, exist_ok=True)
