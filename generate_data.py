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

ESTACIONES = [
    {'name': 'Estación Centro Histórico', 'lat': -8.1100, 'lng': -79.0238},
    {'name': 'Estación Industrial', 'lat': -8.0850, 'lng': -79.0050},
    {'name': 'Estación Residencial', 'lat': -8.1250, 'lng': -79.0400}
]

# Luego, en tu función de generación de datos, iteras sobre esta lista:
def generate_simulated_data(num_hours):
    all_data = []
    for station in ESTACIONES:
        # Generar datos para esta estación específica (con tendencias ligeramente diferentes)
        # ...
        df_station['Estacion'] = station['name']
        df_station['Latitud'] = station['lat']
        df_station['Longitud'] = station['lng']
        all_data.append(df_station)
        
    return pd.concat(all_data)

# Función placeholder para calcular AQI (ASUMIMOS que existe en ml_model.py o la definimos aquí)
def calculate_aqi(pm25, pm10, no2, co):
    """Calcula un AQI simple basado en PM2.5 (Simulación)."""
    # Usaremos una conversión simplificada solo de PM2.5 para mantener el script autocontenido
    # La EPA usa PM2.5 * 2.5 como un buen proxy para el AQI general en rangos bajos/medios.
    if pm25 > 50:
        return round(pm25 * 3) # Valor más agresivo si está alto
    return round(pm25 * 2.5) + random.randint(-5, 5)
    # Calcular AQI y añadir ruido para dinamismo
    aqi_base = int(predicted_pm25_std * 2.5) 
    aqi_final = aqi_base + random.uniform(-2.0, 2.0) 
    
    predictions.append({
        "time_h": i + 1,
        # 🛑 CORRECCIÓN: Asegurar que el AQI sea un entero (sin milésimas)
        "pred_aqi": int(round(aqi_final)), 
        "pred_pm25": round(predicted_pm25_std, 2)
    })

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
    PM10 = PM2_5 * np.random.uniform(1.5, 1.8, num_hours)
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
#3. FUNCIONES DE GESTIÓN Y CARGA DE TABLA
# =======================================================

function renderHistoryTable(records) {
    historyTableBody.innerHTML = ''; 
    
    if (records.length === 0) {
        historyTableBody.innerHTML = '<tr><td colspan="6" class="text-center">No hay registros históricos en el rango seleccionado.</td></tr>';
        return;
    }
    
    records.forEach(record => {
        const row = historyTableBody.insertRow();
        # Columna 1: Fecha/Hora
        const dateObj = new Date(record.timestamp);
        const formattedTime = dateObj.toLocaleDateString() + ' ' + dateObj.toLocaleTimeString();

        row.insertCell().textContent = formattedTime;
        # Columna 2: AQI
        row.insertCell().textContent = record.aqi.toFixed(0);
        # Columna 3: PM2.5
        row.insertCell().textContent = record.pm25.toFixed(1);
        # Columna 4: PM10
        row.insertCell().textContent = record.pm10.toFixed(1);
        # Columna 5: NO2
        row.insertCell().textContent = record.no2.toFixed(1);
        # Columna 6: CO
        row.insertCell().textContent = record.co.toFixed(1);
    });
}

async function fetchHistoryData(startDate = null, endDate = null) {
    let url = `${API_BASE_URL}/history`;
    
    # Si tienes lógica de filtro por fecha en el Backend, modifícala aquí
    if (startDate && endDate) {
        url += `?start_date=${startDate}&end_date=${endDate}`;
    }

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('API response was not ok.');
        
        const data = await response.json();
        renderHistoryTable(data);
    } catch (error) {
        console.error('Error al cargar datos históricos:', error);
        historyTableBody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error de comunicación con el Backend.</td></tr>';
    }
}

function handleSearchClick() {
    # Implementa la lógica de filtro (por ahora, solo llama a la carga sin filtro)
    fetchHistoryData(); 
}

async function addRecord() {
    const timestamp = document.getElementById('input-timestamp').value;
    const pm25 = document.getElementById('input-pm25').value;
    const pm10 = document.getElementById('input-pm10').value;
    
    if (!timestamp || !pm25 || !pm10) {
        alert("Por favor, complete Fecha/Hora, PM2.5 y PM10.");
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/history/record`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                timestamp: timestamp,
                pm25: parseFloat(pm25),
                pm10: parseFloat(pm10)
            })
        });

        if (response.ok) {
            alert('Registro añadido con éxito.');
            fetchHistoryData(); // Recarga la tabla
            # Limpiar campos después de añadir
            document.getElementById('input-timestamp').value = '';
            document.getElementById('input-pm25').value = '';
            document.getElementById('input-pm10').value = '';
        } else {
            throw new Error('Fallo al añadir registro');
        }
    } catch (error) {
        alert('Error al agregar el registro. Verifique el servidor.');
        console.error(error);
    }
}

async function deleteLastRecord() {
    if (!confirm("¿Está seguro de eliminar el último registro?")) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/history/record/last`, {
            method: 'DELETE'
        });

        if (response.ok) {
            alert('Último registro eliminado.');
            fetchHistoryData(); // Recarga la tabla
        } else {
            throw new Error('Fallo al eliminar registro');
        }
    } catch (error) {
        alert('Error al eliminar el registro. Verifique el servidor.');
        console.error(error);
    }
}

// Función de descarga (asumiendo que el Backend maneja la descarga)
function handleDownload() {
    # Lógica para descargar el CSV
    const url = `${API_BASE_URL}/history/download?start_date=${startDateInput.value}&end_date=${endDateInput.value}`;
    window.open(url, '_blank');
}

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






