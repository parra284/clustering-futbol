# -*- coding: utf-8 -*-
"""Utilidades pequeñas compartidas por todo el proyecto."""


def en_notebook() -> bool:
    """True si estamos dentro de un kernel de IPython/Jupyter."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def mostrar(obj) -> None:
    """Muestra un DataFrame con `display()` en Jupyter y con `print()` fuera de él.

    En un notebook, IPython inyecta `display` como builtin, así que llamarlo sin
    importarlo funciona. En un `.py` ejecutado con `python -m models.modelo_gk` eso
    da NameError: por eso todas las funciones de src/ pasan por aquí.
    """
    if en_notebook():
        from IPython.display import display
        display(obj)
    else:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)
