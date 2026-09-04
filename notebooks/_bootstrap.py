# -*- coding: utf-8 -*-
"""Añade la raíz del proyecto a sys.path para poder importar `config` y `src`.

Jupyter pone el directorio del notebook (`notebooks/`) en sys.path, no la raíz, así
que sin esto `from src import ...` falla. Se usa con un simple `import _bootstrap`
en la primera celda.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
