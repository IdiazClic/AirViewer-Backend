import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# =======================================================
# 1. CONFIGURACIÓN Y CÁLCULO DE AQI (Debe existir)
# =======================================================

# Configuración de los datos
START_DATE = datetime(2024, 1, 1, 0, 0)
END_DATE = datetime(2024, 12, 31, 23, 0)
HOURS = int((END_DATE - START_DATE).total_seconds() / 3600)

# Coordenadas simuladas para Trujillo (una sola estación)
LATITUD = -8.1098
LONGITUD = -79.0238

# Función placeholder para calcular AQI (ASUMIMOS que existe en ml_model.py o la definimos aquí)
def calculate_aqi(pm25, pm10, no2, co):
    """Calcula un AQI simple basado en PM2.5 (Simulación)."""
    # Usaremos una conversión simplificada solo de PM2.5 para mantener el script autocontenido
    # La EPA usa PM2.5 * 2.5 como un buen proxy para el AQI general en rangos bajos/medios.
    if pm25 > 50:
        return round(pm25 * 3) # Valor más agresivo si está alto
    return round(pm25 * 2.5) + random.randint(-5, 5)


# =======================================================
# 2. FUNCIÓN DE GENERACIÓN PRINCIPAL (Corregida)
# =======================================================

def generate_simulated_data(num_hours):
    """Genera datos simulados para 1 año completo usando ciclos de Numpy."""
    
    # 1. Generar la secuencia de tiempo
    timestamps = [START_DATE + timedelta(hours=i) for i in range(num_hours)]
    
    # 2. Generar datos base con tendencia diaria/anual
    time_index = np.arange(num_hours)
    
    # Simulación de PM2.5 (Tendencia anual + ruido diario)
    annual_cycle = 20 * np.sin(time_index * 2 * np.pi / (365 * 24)) 
    daily_cycle = 10 * np.sin(time_index * 2 * np.pi / 24) 
    
    # PM2.5 base + ciclos + ruido + ALEATORIEDAD FINA PARA SIMULACIÓN DE LECTURA
    PM2_5 = 33 + annual_cycle + daily_cycle + np.random.normal(0, 5, num_hours)
    PM2_5 = np.clip(PM2_5, 15, 80) 
    
    # PM10 (Generalmente relacionado con PM2.5)
    PM10 = PM2_5 * np.random.uniform(1.5, 2.5, num_hours)
    PM10 = np.clip(PM10, 25, 120)
    
    # NO2 y CO (Añadido y corregido para que la API tenga todos los datos)
    # Usando valores realistas y aplicando la misma lógica de ciclos
    NO2 = 25 + 5 * np.sin(time_index * 2 * np.pi / 24) + np.random.normal(0, 2, num_hours)
    CO = 55 + 10 * np.sin(time_index * 2 * np.pi / 24) + np.random.normal(0, 5, num_hours)

    # Crear el DataFrame y calcular AQI y otros datos requeridos por la tesis
    df = pd.DataFrame({
        'timestamp': timestamps,
        'PM2_5': PM2_5.round(2),
        'PM10': PM10.round(2),
        'NO2': NO2.round(2),
        'CO': CO.round(2),
        'Latitud': LATITUD,
        'Longitud': LONGITUD
    })
    
    # Calcular el AQI para cada fila (usando la función calculate_aqi)
    df['AQI'] = df.apply(lambda row: calculate_aqi(row['PM2_5'], row['PM10'], row['NO2'], row['CO']), axis=1)
    
    # Renombrar columnas para la salida final de la API
    df = df.rename(columns={'PM2_5': 'pm25', 'PM10': 'pm10', 'NO2': 'no2', 'CO': 'co', 'AQI': 'aqi'})
    
    return df

# =======================================================
# 3. EJECUCIÓN DEL SCRIPT
# =======================================================

if __name__ == '__main__':
    # Asegúrate de que el directorio 'data' exista antes de escribir (Render lo hará con mkdir)
    
    # 🛑 Se llama a la función corregida y principal 🛑
    simulated_df = generate_simulated_data(HOURS) 
    
    # Guardar el archivo
    simulated_df.to_csv('data/historical_data.csv', index=False)

    print(f"Dataset de simulación creado exitosamente en: AirViewer/backend/data/historical_data.csv")
    print(f"Dimensiones del dataset: {simulated_df.shape} ({HOURS} registros)")
