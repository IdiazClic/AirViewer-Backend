import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración de los datos
START_DATE = datetime(2024, 1, 1, 0, 0)
END_DATE = datetime(2024, 12, 31, 23, 0)
HOURS = int((END_DATE - START_DATE).total_seconds() / 3600)

# Coordenadas simuladas para Trujillo (una sola estación)
LATITUD = -8.1098
LONGITUD = -79.0238

def generate_simulated_data(hours):
    """Genera datos de calidad del aire y meteorológicos simulados con tendencias."""
    
    # 1. Generar la secuencia de tiempo
    timestamps = [START_DATE + timedelta(hours=i) for i in range(hours)]
    
    # 2. Generar datos base con tendencia diaria/anual
    time_index = np.arange(hours)
    
    # Simulación de PM2.5 (Tendencia anual + ruido diario)
    # Patrón anual (más alto en invierno o en meses secos, p. ej., mitad de año)
    annual_cycle = 20 * np.sin(time_index * 2 * np.pi / (365 * 24)) 
    # Patrón diario (más alto en horas pico 7-9 y 18-20)
    daily_cycle = 10 * np.sin(time_index * 2 * np.pi / 24) 
    
    # PM2.5 base + ciclos + ruido
    PM2_5 = 35 + annual_cycle + daily_cycle + np.random.normal(0, 5, hours)
    PM2_5 = np.clip(PM2_5, 15, 80) # Limitar valores
    
    # PM10 (Generalmente relacionado con PM2.5, pero más alto)
    PM10 = PM2_5 * np.random.uniform(1.5, 2.5, hours)
    PM10 = np.clip(PM10, 25, 120)
    
    # CO2 (Alto en horas pico de tráfico)
    CO2 = 500 + 150 * np.sin(time_index * 2 * np.pi / 24) + np.random.normal(0, 30, hours)
    CO2 = np.clip(CO2, 450, 800).astype(int)
    
    # Variables Meteorológicas (Temperatura más alta en verano)
    Temperatura = 22 + 5 * np.sin(time_index * 2 * np.pi / (365 * 24)) + np.random.normal(0, 2, hours)
    Humedad = 80 + 15 * np.sin(time_index * 2 * np.pi / (365 * 24)) + np.random.normal(0, 5, hours)
    Presion = 1008 + np.random.normal(0, 5, hours)
    
    # Crear el DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'PM2_5': PM2_5.round(2),
        'PM10': PM10.round(2),
        'Temperatura': Temperatura.round(1),
        'Humedad': Humedad.round(0).astype(int),
        'Presion': Presion.round(2),
        'CO2': CO2,
        'Latitud': LATITUD,
        'Longitud': LONGITUD
    })
    
    return data

# Generar y guardar el archivo
simulated_df = generate_simulated_data(HOURS)
simulated_df.to_csv('data/historical_data.csv', index=False)

print(f"Dataset de simulación creado exitosamente en: AirViewer/backend/data/historical_data.csv")
print(f"Dimensiones del dataset: {simulated_df.shape} ({HOURS} registros)")
