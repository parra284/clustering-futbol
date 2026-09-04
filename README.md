# ⚽ Segmentación de jugadores de fútbol — 5 grandes ligas europeas, 2025-26

Clustering **por posición** de jugadores de FBref siguiendo el protocolo **CRISP-DM**.

- **Problema.** Comparar a un central con un extremo no tiene sentido: un clustering global separa
  porteros de jugadores de campo y poco más.
- **Enfoque.** Cuatro modelos K-Means independientes, uno por línea. Cada jugador se compara solo
  contra los de su propia posición, y con las métricas que definen *esa* posición.
- **Resultado.** **15 arquetipos** sobre 2 528 apariciones de 1 998 jugadores distintos.
- **Polivalentes.** Un "DF,MF" entra a los dos modelos y recibe dos etiquetas: qué rol cumple
  según desde qué línea se le mire.

---

## Estructura

| Carpeta | Contiene |
|---|---|
| `app/` | Dashboard de Streamlit (`dashboard.py`) y sus gráficos (`graficos.py`) |
| `config/` | Todos los parámetros: rutas, umbrales, variables de cada modelo, `k`, semillas, paleta |
| `data/raw/` | CSV descargado de FBref (3 536 jugadores × 75 columnas), un archivo por temporada |
| `data/processed/` | Base limpia: liga rellenada y filas sin posición eliminadas |
| `models/` | Los 4 modelos, uno por archivo, más `entrenar_todos.py` |
| `models/entrenados/` | Los 4 `.joblib` (imputer + scaler + kmeans + pca) |
| `notebooks/` | El informe CRISP-DM completo y un espacio de pruebas |
| `reports/` | 8 CSV generados + `figuras/` con las 15 gráficas |
| `src/` | Las funciones largas y reutilizables del pipeline |

Ningún archivo usa rutas relativas al directorio de trabajo: todo se ancla en `config/rutas.py`.

---

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  ·  source .venv/bin/activate en Linux/macOS
pip install -r requirements.txt
```

Python 3.13. No hace falta descargar nada de FBref: `data/raw/` ya viene en el repositorio.

**Para cambiar de temporada** basta con editar `FBREF_TEMPORADAS` en `config/parametros.py`
(`"2526"` = 2025-26) y reejecutar. El nombre del CSV lleva la temporada dentro, así que el cambio
dispara una descarga nueva en vez de reutilizar la anterior en silencio. Los 15 nombres de perfil,
en cambio, **hay que releerlos a mano** tras cada reentrenamiento.

---

## Cómo ejecutar

Siempre **desde la raíz del proyecto**.

| Qué | Comando |
|---|---|
| Entrenar los 4 modelos y exportar todo | `python -m models.entrenar_todos` |
| Igual, pero sin generar las figuras (más rápido) | `python -m models.entrenar_todos --sin-figuras` |
| Entrenar una sola posición | `python -m models.modelo_gk` |
| Ver el dashboard | `streamlit run app/dashboard.py` |
| Leer el informe completo | `jupyter lab notebooks/clustering_jugadores_crispdm.ipynb` |
| Volver a descargar de FBref | `python -c "from src import datos; datos.recolectar_datos_permitidos(forzar_descarga=True)"` |

El notebook y `entrenar_todos.py` producen **exactamente los mismos archivos**: ejecutan el mismo
código de `src/`.

---

## Los 4 modelos

Todos son `KMeans(n_init=10, random_state=42)` sobre variables llevadas a **por 90 minutos**,
imputadas por mediana y escaladas con **Yeo-Johnson**.

| Modelo | k | Variables | Etiquetados | Entrenan | Umbral entren. | Silueta (entren.) |
|---|---|---|---|---|---|---|
| **GK** Porteros | 3 | 5 | 139 | 105 | 15 partidos | 0.210 |
| **DF** Defensas | 3 | 8 | 735 | 605 | 10 partidos | 0.148 |
| **MF** Mediocampistas | 4 | 10 | 1 143 | 878 | 10 partidos | 0.133 |
| **FW** Delanteros | 5 | 10 | 511 | 380 | 10 partidos | 0.109 |

**Dos umbrales, no uno:**

- **Inclusión — 5 partidos (450 min).** Quién recibe etiqueta. Bajo a propósito: un juvenil con
  600 minutos es justo el caso que interesa a scouting.
- **Entrenamiento — 10 partidos (15 en porteros).** Quién *define* los perfiles. El imputer, el
  scaler y los centroides se ajustan solo con estos; el resto recibe etiqueta por predicción y
  queda marcado como `confianza = baja`.

Los porteros piden 15 porque sus variables son porcentajes, y un porcentaje sobre pocos partidos
es muy inestable.

---

## El dashboard

```powershell
streamlit run app/dashboard.py
```

| Pestaña | Qué permite |
|---|---|
| 🔍 **Jugador** | Buscar por nombre → ficha completa (liga, temporada, equipo, país, edad, minutos), su perfil, sus 5-10 variables en percentiles y por 90, y los 8 jugadores más parecidos |
| 👥 **Equipo** | La plantilla entera repartida por perfiles, con el reparto de cada línea |
| 🧭 **Explorar** | Filtrar las 2 528 apariciones por posición, perfil, edad, liga y minutos; **ordenar por minutos o por cualquier variable del modelo** (asc./desc.) y dispersión edad/minutos |
| ⚖️ **Comparar** | Dos jugadores de la misma posición, variable a variable, en percentiles y en cifras reales |

**Tres decisiones de diseño que conviene saber leer:**

- **Las comparaciones van en percentiles, no en unidades reales.** `Crs`/90 llega a 5 y
  `Ast`/90 a 0.5: en un eje común la segunda sería invisible. Las cifras reales están
  siempre en la tabla de al lado.
- **Un percentil es dentro de su propia posición.** El 90 de un defensa y el 90 de un
  delantero no significan lo mismo, y por eso solo se comparan jugadores de la misma línea.
- **Solo se puede ordenar por lo que todas las posiciones elegidas midieron.** Entre
  posiciones de campo quedan 7-9 variables comunes; si se incluyen porteros, solo
  `Fld/90`. Para ordenar por las 8 de defensas o las 5 de porteros, elige esa sola
  posición. Un filtro vacío significa «sin filtrar», no «ningún resultado».
- **«Parecido» es cercano en las variables del modelo**, no parecido en todo. El modelo
  no mide pases, conducción ni posicionamiento: Rodri y un extremo pueden salir cerca.

El tema (claro/oscuro) lo resuelve Streamlit; `.streamlit/config.toml` solo fija el color
de acento al azul del proyecto, porque el rojo por defecto hacía que un percentil 100 se
leyera como una alarma.

---

## Los 15 perfiles

| Posición | Cluster | Perfil | n |
|---|---|---|---|
| **Porteros** | 0 | Portero muy exigido | 36 |
| | 1 | Portero de bloque sólido | 33 |
| | 2 | Portero de área | 70 |
| **Defensas** | 0 | Central posicional | 235 |
| | 1 | Central de combate | 250 |
| | 2 | Lateral ofensivo | 250 |
| **Mediocampistas** | 0 | Interior de conducción | 271 |
| | 1 | Carrilero | 268 |
| | 2 | Interior ofensivo | 279 |
| | 3 | Pivote destructor | 325 |
| **Delanteros** | 0 | Nueve goleador | 101 |
| | 1 | Nueve de choque | 129 |
| | 2 | Extremo centrador | 87 |
| | 3 | Extremo de presión | 94 |
| | 4 | Extremo regateador | 100 |

Los nombres están en `config/perfiles.py` y se escriben en el CSV al exportar. K-Means **no numera
igual al reentrenar**, y el paso de 2023-24 a 2025-26 lo demostró: en porteros la numeración rotó
entera y en defensas cambió la estructura (de lateral/central/carrilero a dos tipos de central más
un lateral). Si se reentrena, hay que releerlos uno por uno.

---

## Las tres decisiones de la fase 3

Cada una está justificada con evidencia en el notebook, no por convención.

| # | Decisión | Cómo se comprobó | Figura |
|---|---|---|---|
| 1 | Umbral de minutos | ARI de la partición al mover el corte entre 6 y 18 partidos | `31_umbral_minutos.png` |
| 2 | Escalado Yeo-Johnson y no `StandardScaler` | máx \|z\| por variable: con `StandardScaler` un solo jugador queda a 4.8 σ y domina la distancia; con Yeo-Johnson baja a ~2.1 σ | `32_escalado.png` |
| 3 | Qué variables entran | Se quita cada una y se mide el ARI. Por encima de 0.90 no aporta: así salieron `G-PK`/90 en defensas y `SoT%` en mediocampistas | `33_ablacion_variables.png` |

---

## Salidas generadas

| Archivo | Qué es |
|---|---|
| `reports/jugadores_{POS}_con_cluster.csv` | Dataset completo + `cluster`, `perfil`, `confianza`, `dist_centroide` |
| `reports/perfil_clusters_{POS}.csv` | Media de cada variable por cluster, en unidades reales por 90 |
| `models/entrenados/modelo_{POS}.joblib` | Bundle reutilizable: imputer, scaler, kmeans, pca y qué columnas convertir a por-90 |
| `reports/figuras/*.png` | Las 15 gráficas del informe |
| `data/processed/jugadores_base.csv` | Base limpia, entrada común de los cuatro modelos |

`dist_centroide` mide cuánto encaja el jugador en su arquetipo: menor = más limpio. Es lo que
ordena la lista de candidatos de scouting.

---

## Notas

- **`confianza = baja`** significa que el jugador se parece a ese arquetipo *según los minutos que
  lleva*, no que sea su rol confirmado.
- Las siluetas son bajas (0.13–0.19) porque los perfiles de juego son un **continuo**, no grupos
  separados: no hay una frontera natural entre un interior creador y un mediapunta. El valor está
  en los ejes que se encuentran, no en la separación.
- **Trabajo futuro.** DBSCAN y clustering jerárquico sobre los mismos datos; `notebooks/exploracion.ipynb`
  ya viene preparado para ello.
