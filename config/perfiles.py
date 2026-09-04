# -*- coding: utf-8 -*-
"""Nombre futbolístico de cada uno de los 15 clusters.

Salen de leer `reports/perfil_clusters_{POS}.csv` —la desviación de cada cluster
respecto a la media de su posición— y de comprobar contra los jugadores más cercanos
a cada centroide.

**Estos nombres son de la temporada 2025-26.** K-Means no numera los clusters igual al
reentrenar: al pasar de 2023-24 a 2025-26 la numeración de porteros rotó entera (el
cluster 0 pasó a ser el 2) y en defensas cambió hasta la estructura, de
lateral/central/carrilero a dos tipos de central más un lateral. Si se vuelve a
reentrenar, hay que releerlos uno por uno.

Para que ese riesgo no llegue al dashboard, `src.modelado.exportar()` escribe el nombre
resuelto como columna `perfil` dentro del CSV en el momento de exportar: la etiqueta
viaja con los datos, y el dashboard no vuelve a mapear nada por su cuenta.
"""

PERFILES = {
    # PORTEROS ---------------------------------------------------------------------
    # El eje dominante es la calidad del bloque que tienen delante: CS% va de -53 % a
    # +64 % respecto a la media, y arrastra a SoTA/90 y a los penaltis enfrentados.
    "GK": {
        # SoTA +12 %, CS% -53 %, PKatt +37 %, Save% -9 %: le llega de todo, encaja y su
        # equipo comete muchos penaltis. Leo Román, Agirrezabala, Rønnow, Vlachodimos.
        0: "Portero muy exigido",
        # SoTA -20 %, CS% +64 %, Save% +7 %, PKatt -34 %: poco trabajo y lo resuelve.
        # Kobel, Oblak, Di Gregorio, Courtois, Sommer.
        1: "Portero de bloque sólido",
        # Todo en la media salvo Fld/90 +15 %: sale a disputar. Es el grupo más grande
        # (70 de 139). Falcone, Suzuki, Henderson, De Gea, Sels, Kelleher.
        2: "Portero de área",
    },
    # DEFENSAS ---------------------------------------------------------------------
    # Centros y asistencias por 90 separan al lateral (+113 % y +108 %) del central;
    # dentro de los centrales, el roce (faltas y amarillas) separa dos maneras.
    "DF": {
        # Por debajo de la media en TODO: no entra, no hace faltas, no ataca. Central
        # posicional. Joachim Andersen, Chalobah, Toti Gomes, Santiago Bueno.
        0: "Central posicional",
        # Int +22 %, Fls +25 %, CrdY +44 %, Crs -48 %, Ast -61 %: corta y se juega la
        # tarjeta, sin aporte ofensivo. Vivian, Schlotterbeck, Andrich, Unai Núñez.
        1: "Central de combate",
        # Crs +113 %, Ast +108 %, Fld +31 %, Sh +26 %: el lateral que ataca.
        # Bogle, Chavarría, Álex Moreno, Mojica, Mukiele, Truffert.
        2: "Lateral ofensivo",
    },
    # MEDIOCAMPISTAS ---------------------------------------------------------------
    # El abanico más ancho: el eje destrucción ↔ llegada explica casi todo.
    "MF": {
        # Fld +36 %, CrdY +42 %, Fls +26 %, G-PK +21 %: recibe faltas (proxy de
        # conducción) y llega. Vlašić, Rabiot, Summerville, Esposito.
        0: "Interior de conducción",
        # Crs +32 % y todo lo demás por debajo: son laterales que FBref lista como MF.
        # Sabitzer, Dalot, Cambiaso, Da Costa, Çelik.
        1: "Carrilero",
        # G-PK +104 %, Off +96 %, Sh +65 %, Ast +56 %, Int -46 %: ataca y no defiende.
        # Jaidon Anthony, Cho, Tsyhankov, Trossard, Nico Williams.
        2: "Interior ofensivo",
        # Int +47 %, TklW +30 %, con Off -73 %, G-PK -65 %, Ast -57 %: puro corte.
        # Es el grupo más grande (325). Aebischer, Ampadu, Bouaddi, Eric García.
        3: "Pivote destructor",
    },
    # DELANTEROS -------------------------------------------------------------------
    # Dos familias —nueves y extremos— y dentro de cada una, qué añaden sin balón.
    "FW": {
        # G-PK +93 %, Sh +33 %, SoT% +18 %, Crs -62 %: el que mete goles.
        # Watkins, Callum Wilson, Édouard, Schick, Balogun.
        0: "Nueve goleador",
        # Off +52 %, Fls +26 %, Crs -70 %, Ast -46 %: pelea, cae en fuera de juego y no
        # asocia. Giroud, Samatta, Cheddira, Diallo, Noslin.
        1: "Nueve de choque",
        # Crs +106 %, Ast +23 %, Fls -39 %: extremo de banda que centra.
        # Danjuma, De Ketelaere, Moses Simon, Jacob Murphy, Pedro Neto.
        2: "Extremo centrador",
        # Int +72 %, TklW +66 %, con G-PK -58 % y Sh -31 %: trabaja más de lo que remata.
        # Mitoma, Grüll, Vlašić, Tramoni.
        3: "Extremo de presión",
        # Ast +70 %, Fld +44 %, Crs +40 %, más trabajo defensivo: desborda y asiste.
        # Doku, Pépé, Akliouche, Bahoya, Rowe, Strefezza.
        4: "Extremo regateador",
    },
}

# Columnas que el dashboard lee de cada CSV exportado.
COLUMNAS_DASHBOARD = ["league", "team", "player", "pos", "age",
                      "Playing Time_Min", "cluster", "perfil",
                      "confianza", "dist_centroide"]

# Equipo que aparece seleccionado al abrir el dashboard.
EQUIPO_POR_DEFECTO = "Arsenal"
