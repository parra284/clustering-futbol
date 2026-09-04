import soccerdata as sd
import pandas as pd

def recolectar_datos_permitidos():
    print("Inicializando conexión con FBref...")
    fbref = sd.FBref(leagues=['Big 5 European Leagues Combined'], seasons=["2324"])
    
    # Solo solicitamos las tablas que la librería permite actualmente
    categorias_stats = ['standard', 'shooting', 'playing_time', 'misc', 'keeper']
    
    lista_dfs = []
    
    for stat in categorias_stats:
        print(f"Descargando estadísticas de tipo: '{stat}'...")
        try:
            df_stat = fbref.read_player_season_stats(stat_type=stat)
            lista_dfs.append(df_stat)
        except Exception as e:
            print(f"⚠️ Error descargando {stat}: {e}")
            
    print("\nUniendo todas las métricas en una sola tabla...")
    # Concatenamos horizontalmente
    df_completo = pd.concat(lista_dfs, axis=1)
    
    # Eliminamos columnas duplicadas (como edad, minutos o equipo que vienen repetidas)
    df_completo = df_completo.loc[:, ~df_completo.columns.duplicated()]
    
    # Limpiamos los índices
    df_completo = df_completo.reset_index()
    
    # Aplanamos los nombres de las columnas para evitar problemas en pandas
    df_completo.columns = ['_'.join([str(c) for c in col if c]).strip() if isinstance(col, tuple) else col for col in df_completo.columns.values]
    
    print("\n--- VOLUMEN DE DATOS ---")
    print(f"Jugadores procesados: {len(df_completo)}")
    print(f"Características (features) obtenidas: {len(df_completo.columns)}")
    
    # 4. GUARDAR LOS DATOS EN UN CSV
    nombre_archivo = "datos_jugadores_permitidos.csv"
    df_completo.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')
    print(f"\n✅ ¡Los datos avanzados se han guardado exitosamente en '{nombre_archivo}'!")
    
    # Mostrar un pequeño resumen de las columnas generadas
    print("\n🔍 Algunas de las métricas que logramos obtener:")
    metricas_ejemplo = [c for c in df_completo.columns if 'Prg' in c or 'xG' in c or 'Aerial' in c or 'Ast' in c]
    print(metricas_ejemplo[:10])

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    recolectar_datos_permitidos()