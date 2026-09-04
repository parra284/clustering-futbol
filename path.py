import soccerdata as sd
import os

# Obtiene la ruta donde está instalado el paquete soccerdata
ruta_soccerdata = sd.__path__[0]

# Construye la ruta al archivo league_dict.json
ruta_json = os.path.join(ruta_soccerdata, 'config', 'league_dict.json')

print("Tu archivo JSON está en:")
print(ruta_json)