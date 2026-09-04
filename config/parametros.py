# -*- coding: utf-8 -*-
"""Todos los parámetros del proyecto en un solo sitio.

Cambiar un umbral, una variable o un `k` aquí lo cambia en el notebook, en los
cuatro modelos y en el dashboard a la vez. Antes de este archivo, los umbrales
estaban escritos tres veces (notebook, dashboard, y como texto literal en la
ayuda del dashboard) y se quedaban obsoletos en silencio.
"""

# ======================================================================================
# 1. Recolección (FBref vía soccerdata)
# ======================================================================================

# Tabla combinada de las 5 grandes ligas europeas: La Liga, Ligue 1, Premier League,
# Serie A y Bundesliga.
FBREF_LIGAS = ["Big 5 European Leagues Combined"]
FBREF_TEMPORADAS = ["2324"]

# Tablas que la librería permite actualmente. 'keeper' es imprescindible para el
# modelo de porteros (paradas, % de parada, goles encajados).
FBREF_TABLAS = ["standard", "shooting", "playing_time", "misc", "keeper"]

ENCODING_CSV = "utf-8-sig"


# ======================================================================================
# 2. Filtros de población
# ======================================================================================

# Dos umbrales de minutos, expresados en partidos completos (1 partido = 90 min):
#
#   ENTRENAMIENTO — quién define los perfiles. 10 partidos (media temporada de un titular
#   habitual). Los porteros piden 15 porque sus variables son porcentajes, y un porcentaje
#   sobre pocos partidos es muy inestable.
#
#   INCLUSIÓN — quién recibe una etiqueta. 5 partidos, igual para todos. Una vez fijados
#   los perfiles, asignar a un juvenil con 600 minutos no contamina nada, y es justo el
#   caso que a un departamento de scouting le interesa.
#
# Son cortes convencionales, no optimizados; la sección 3.1 del notebook comprueba que el
# resultado no depende del valor exacto.
PARTIDO = 90
MINUTOS_ENTRENAMIENTO = {
    "GK": 15 * PARTIDO,   # 1350 min
    "DF": 10 * PARTIDO,   #  900 min
    "MF": 10 * PARTIDO,   #  900 min
    "FW": 10 * PARTIDO,   #  900 min
}
MINUTOS_INCLUSION = 5 * PARTIDO   # 450 min, para las cuatro posiciones

POSICIONES = ["GK", "DF", "MF", "FW"]
NOMBRE_POS = {"GK": "Porteros", "DF": "Defensas",
              "MF": "Mediocampistas", "FW": "Delanteros"}

# La tabla combinada de las 5 grandes ligas llega con el bloque alemán sin etiquetar:
# 'league' viene vacío para los 18 equipos de la Bundesliga, aunque sus filas están
# completas. Sin este relleno esos jugadores entran igual a los modelos —el filtro es por
# posición y minutos, no por liga— pero aparecen sin liga en las tablas exportadas.
LIGA_FALTANTE = "GER-Bundesliga"

# Los jóvenes son el caso de uso que motiva el umbral de inclusión bajo, así que se
# cuenta explícitamente cuántos entran.
EDAD_JOVEN = 21


# ======================================================================================
# 3. Variables de cada modelo
# ======================================================================================
# Cada posición se segmenta con SUS propias métricas. Todos los totales de temporada se
# llevan a "por 90 minutos" antes de entrar al modelo (lo hace preparar_features).
#
# Columnas que ninguna posición usa como variable:
#   · identificadores y contexto ....... player, team, league, nation, pos, age, born
#   · exposición (cuánto juega, no cómo)  MP, Starts, Min, 90s, Min%, Mn/MP, Starts_*, Subs_*
#   · resultado del equipo ............. Team Success_* y, en porteros, W / D / L. Miden lo
#                                        bien que va el equipo con el jugador en el campo,
#                                        no cómo juega ese jugador.
#   · sucesos casi inexistentes ........ CrdR, 2CrdY y OG están en 0 para el 78-99 % de la
#                                        plantilla; al escalar, un solo suceso se convierte
#                                        en un valor extremo que arrastra al jugador entero.
#   · columnas 100 % vacías ............ Performance_PKwon, Performance_PKcon
#   · totales con tasa equivalente ..... Standard_Gls/Sh/SoT, Performance_Gls/Ast/G+A...
#   · variables sin poder discriminante en SU posición: se comprobó quitando cada una y
#     midiendo cuánto cambia la partición. G-PK/90 en defensas y SoT% en mediocampistas
#     dejaban la partición prácticamente idéntica (ARI 0.91) y bajaban la silueta, así que
#     solo añadían una dimensión de ruido a la distancia. Detalle en las secciones 5 y 6.

FEATURES = {
    # PORTEROS — carga de trabajo, resolución de esa carga y presencia en el área.
    # Sin 'Performance_GA90': los goles encajados por 90 correlacionan 0.80 con SoTA/90,
    # -0.71 con Save% y -0.75 con CS%, es decir que quedan casi determinados por las otras
    # tres. Incluirlos hace que la primera componente principal absorba el 53 % de la
    # varianza en lugar del 45 %, sin cambiar la partición resultante.
    "GK": [
        "Performance_SoTA",       # tiros a puerta recibidos -> /90 (cuánto le llega)
        "Performance_Save%",      # % de paradas (el eje de habilidad, independiente de la carga)
        "Performance_CS%",        # % de partidos con la portería a cero
        "Penalty Kicks_PKatt",    # penaltis enfrentados -> /90 (indisciplina del bloque)
        "Performance_Fld",        # faltas recibidas -> /90 (salidas y disputas en el área)
    ],
    # DEFENSAS — acción defensiva, roce/disciplina y aporte ofensivo. El eje dominante es
    # central vs. lateral, y los centros por 90 son su marcador más nítido.
    "DF": [
        "Performance_Int",        # intercepciones -> /90
        "Performance_TklW",       # entradas ganadas -> /90
        "Performance_Fls",        # faltas cometidas -> /90
        "Performance_Fld",        # faltas recibidas -> /90
        "Performance_CrdY",       # amarillas -> /90
        "Performance_Crs",        # centros -> /90 (lateral ofensivo)
        "Per 90 Minutes_Ast",     # asistencias por 90
        "Standard_Sh/90",         # tiros por 90
    ],
    # MEDIOCAMPISTAS — el abanico de roles más ancho: pivote destructor, interior creador,
    # carrilero y mediapunta llegador conviven en la misma etiqueta.
    "MF": [
        "Performance_Int",        # intercepciones -> /90
        "Performance_TklW",       # entradas ganadas -> /90
        "Performance_Fls",        # faltas cometidas -> /90
        "Performance_Fld",        # faltas recibidas -> /90 (proxy de conducción)
        "Performance_CrdY",       # amarillas -> /90
        "Performance_Crs",        # centros -> /90
        "Performance_Off",        # fueras de juego -> /90 (llegada al área)
        "Per 90 Minutes_Ast",     # asistencias por 90
        "Per 90 Minutes_G-PK",    # goles sin penalti por 90
        "Standard_Sh/90",         # volumen de remate por 90
    ],
    # DELANTEROS — finalización, movimiento de ruptura, juego de banda y trabajo sin balón.
    "FW": [
        "Per 90 Minutes_G-PK",    # goles sin penalti por 90
        "Per 90 Minutes_Ast",     # asistencias por 90
        "Standard_Sh/90",         # volumen de remate por 90
        "Standard_SoT%",          # puntería (% de tiros a puerta)
        "Performance_Off",        # fueras de juego -> /90 (ataque al espacio)
        "Performance_Crs",        # centros -> /90 (extremo)
        "Performance_Fld",        # faltas recibidas -> /90 (regateador)
        "Performance_Fls",        # faltas cometidas -> /90
        "Performance_Int",        # intercepciones -> /90 (presión alta)
        "Performance_TklW",       # entradas ganadas -> /90 (presión alta)
    ],
}

# Número de clusters de cada modelo. Se fija aquí, en un solo sitio, y las secciones 4
# a 7 lo leen: así la auditoría de variables de 3.3 usa exactamente el mismo `k` que
# el modelo final. El criterio para cada valor está en la sección correspondiente.
K_ELEGIDO = {"GK": 3, "DF": 3, "MF": 4, "FW": 5}

# Candidatas que se evaluaron y quedaron fuera de cada modelo. Las secciones 4 a 7
# las grafican con figura_descartadas() para mostrar la razón de cada exclusión.
DESCARTADAS = {
    "GK": {"Performance_Saves": "Saves/90", "Performance_GA90": "GA90",
           "Penalty Kicks_Save%": "PK Save%", "Performance_Int": "Int/90",
           "Performance_TklW": "TklW/90"},
    "DF": {"Standard_SoT/90": "SoT/90", "Performance_Off": "Off/90",
           "Standard_SoT%": "SoT%", "Performance_CrdR": "CrdR/90"},
    "MF": {"Standard_SoT/90": "SoT/90", "Standard_SoT%": "SoT%",
           "Performance_PKatt": "PKatt/90", "Performance_CrdR": "CrdR/90"},
    "FW": {"Standard_SoT/90": "SoT/90", "Standard_G/Sh": "G/Sh",
           "Performance_PKatt": "PKatt/90", "Performance_CrdY": "CrdY/90",
           "Performance_CrdR": "CrdR/90"},
}


# ======================================================================================
# 4. Hiperparámetros del pipeline
# ======================================================================================

RANDOM_STATE = 42
N_INIT = 10                       # reinicios de K-Means
COL_90S = "Playing Time_90s"      # divisor para llevar totales a "por 90"
ESCALADO = "yeo-johnson"          # alternativa: "standard"
N_REPRESENTATIVOS = 6             # jugadores más cercanos a cada centroide que se listan
N_HALLAZGOS = 8                   # candidatos de confianza baja que se listan

# Rango de búsqueda de k en el método del codo. GK usa K_MAX_GK porque solo tiene 103
# jugadores de entrenamiento y k=8 produce clusters de tamaño irrelevante.
K_MIN = 2
K_MAX = 8
K_MAX_GK = 6


# ======================================================================================
# 5. Parámetros de las auditorías de la fase 3
# ======================================================================================

# 3.1 — umbrales alternativos (en partidos) contra los que se mide la estabilidad (ARI).
RANGO_UMBRALES = [6, 8, 10, 12, 15, 18]

# 3.2 — posición sobre la que se demuestra el efecto del escalado.
POS_DEMO_ESCALADO = "DF"

# 3.3 — variables que se probaron y se retiraron por este mismo test. Se vuelven a añadir
# al conjunto candidato para que la figura muestre la decisión, no solo su resultado.
RETIRADAS = {"GK": [], "DF": ["Per 90 Minutes_G-PK"], "MF": ["Standard_SoT%"], "FW": []}

# Por encima de este ARI, quitar la variable deja la partición igual: no aporta.
UMBRAL_ARI = 0.90
