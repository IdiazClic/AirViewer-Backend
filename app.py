# AirViewer/backend/app.py (CÓDIGO FINAL Y COMPLETO)

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import io
import random
from datetime import datetime, timedelta

# Importa las funciones del módulo de ML (ml_model.py debe estar en la misma carpeta)
import ml_model 

# =======================================================
# 1. VARIABLES GLOBALES DE ML Y GESTIÓN DE DATOS (DB Simulada)
# =======================================================
AIR_QUALITY_MODEL = None
AIR_QUALITY_SCALER = None

# Simulación de la base de datos histórica en memoria
DB_RECORDS = [
    {"id": 1, "timestamp": (datetime.now() - timedelta(hours=2)).isoformat() + "Z", "aqi": 75, "pm25": 28.1, "pm10": 45.0, "no2": 40.5, "co": 3.1},
    {"id": 2, "timestamp": (datetime.now() - timedelta(hours=1)).isoformat() + "Z", "aqi": 90, "pm25": 35.0, "pm10": 55.0, "no2": 55.2, "co": 4.5},
    {"id": 3, "timestamp": datetime.now().isoformat() + "Z", "aqi": 82, "pm25": 30.2, "pm10": 48.0, "no2": 45.1, "co": 3.8}
]
NEXT_ID = 4 # Variable para asignar el próximo ID

app = Flask(__name__)
CORS(app) 

# AirViewer/backend/app.py (Añadir cerca de DB_RECORDS)
import requests # Necesario para hacer solicitudes HTTP

# --- CONFIGURACIÓN DE THINGSPEAK ---
THINGSPEAK_CHANNEL_ID = '2989972' # REEMPLAZA ESTE VALOR CON TU ID DE CANAL
THINGSPEAK_READ_KEY = 'DW1VFS3QXOJRWSIK' # REEMPLAZA CON TU CLAVE (Si es privado)

# Mapeo de Campos: ThingSpeak usa 'field1', 'field2', etc.
# Debes saber qué número de campo corresponde a PM2.5, PM10, etc.
FIELD_MAP = {
    'PM2.5': 'field1', 
    'PM10': 'field2',
    'NO2': 'field3',
    'CO': 'field4',
    # Asume que Field5 es la temperatura para cálculos Cstd (aunque no la usemos aquí)
    'TEMP': 'field5' 
}

# =======================================================
# 2. LÓGICA DE CARGA DE ML
# =======================================================
def initialize_ml_components():
    """Carga los artefactos ML si aún no están cargados. Intentará entrenar si no existen."""
    global AIR_QUALITY_MODEL, AIR_QUALITY_SCALER
    
    try:
        AIR_QUALITY_MODEL, AIR_QUALITY_SCALER = ml_model.load_artefacts()
        print("Modelos ML inicializados con éxito.")
    
    except Exception as e:
        print(f"ADVERTENCIA: Falló la carga/entrenamiento de artefactos: {e}. Usando simulación de datos.")


@app.before_request
def check_ml_init():
    """Ejecuta la inicialización de ML si no ha ocurrido."""
    if AIR_QUALITY_MODEL is None:
        initialize_ml_components()

# =======================================================
# 3. ENDPOINTS DE MONITOREO Y PREDICCIÓN
# =======================================================

@app.route('/api/v1/data/current', methods=['GET'])
# AirViewer/backend/app.py (Función get_current_data)

def get_current_data():
    """Retorna la última lectura de sensores, priorizando ThingSpeak."""
    
    # 1. Intento de Conexión a ThingSpeak
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_KEY}"
        
        # Hacemos la solicitud a la API de ThingSpeak
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Lanza un error si el status es 4xx o 5xx
        ts_data = response.json()

        # 2. Extracción y Normalización de Datos
        pm25 = float(ts_data.get(FIELD_MAP['PM2.5'], 0))
        pm10 = float(ts_data.get(FIELD_MAP['PM10'], 0))
        no2 = float(ts_data.get(FIELD_MAP['NO2'], 0))
        co = float(ts_data.get(FIELD_MAP['CO'], 0))
        
        # Simulación de AQI basado en el valor de PM2.5 (Simple para el Dashboard)
        aqi_val = int(pm25 * 2.5)
        
        if aqi_val <= 50: estado = "Buena"
        elif aqi_val <= 150: estado = "Moderada"
        else: estado = "No saludable"
        
        data = {
            "timestamp": datetime.now().isoformat() + "Z",
            "aqi": aqi_val,
            "estado": estado,
            "pm25": round(pm25, 1), 
            "pm10": round(pm10, 1),
            "no2": round(no2, 1),
            "co": round(co, 1)
        }
        return jsonify(data)
    
    except Exception as e:
        # 3. Fallback a Simulación si falla la API (tu código de simulación anterior)
        print(f"ADVERTENCIA: Fallo al conectar con ThingSpeak ({e}). Usando simulación.")
        
        aqi_val = random.randint(50, 250) 
        if aqi_val <= 50: estado = "Buena"
        elif aqi_val <= 150: estado = "Moderada"
        else: estado = "No saludable"
        
        data = {
            "timestamp": datetime.now().isoformat() + "Z",
            "aqi": aqi_val,
            "estado": estado,
            "pm25": round(random.uniform(15, 70), 1), 
            "pm10": round(random.uniform(30, 100), 1),
            "no2": round(random.uniform(30, 80), 1),
            "co": round(random.uniform(2, 8), 1)
        }
        return jsonify(data)

@app.route('/api/v1/data/last_24h', methods=['GET'])
def get_last_24h():
    """Retorna la tendencia del AQI promediada por hora para las últimas 24h (para gráficas)."""
    now = datetime.now()
    hourly_data = []
    for i in range(24):
        hour_time = now - timedelta(hours=23 - i)
        hourly_data.append({"time": hour_time.strftime("%H:%M"), "aqi": random.randint(60, 110)})
    return jsonify(hourly_data)


@app.route('/api/v1/prediction/next_24h', methods=['GET'])
def get_prediction():
    """Retorna el vector de predicciones ML."""
    
    simulated_input_data = pd.DataFrame({
        'PM2_5': [random.randint(50, 120) for _ in range(24)],
        'PM10': [random.randint(80, 150) for _ in range(24)],
        'Temperatura': [random.uniform(20, 25) for _ in range(24)],
        'Humedad': [random.uniform(70, 90) for _ in range(24)],
        'Presion': [random.uniform(1005, 1015) for _ in range(24)],
        'CO2': [random.randint(450, 700) for _ in range(24)],
    })
    
    try:
        predictions = ml_model.make_prediction(AIR_QUALITY_MODEL, AIR_QUALITY_SCALER, simulated_input_data) 
    except Exception:
        # Fallback de simulación si el ML falla
        predictions = [{"time_h": h + 1, "pred_aqi": random.randint(100, 150), "pred_pm25": random.uniform(35, 55)} for h in range(24)]

    return jsonify(predictions)


@app.route('/api/v1/model/metrics', methods=['GET'])
def get_model_metrics():
    """Retorna las métricas de rendimiento (RMSE, R2)."""
    try:
        metrics = ml_model.get_evaluation_metrics()
    except Exception:
        metrics = {"rmse": 0.0, "r_squared": 0.0, "mae": 0.0, "model_name": "FALLO DE CARGA", "last_trained": "N/A"}
    return jsonify(metrics)

@app.route('/api/v1/prediction/sources', methods=['GET'])
def get_sources_contribution():
    """Retorna el porcentaje de contribución de las fuentes de contaminación."""
    sources = {
        "pred_period": "2025-11-18",
        "sources": [
            {"name": "Tráfico Vehicular", "weight_percent": 65},
            {"name": "Actividad Industrial", "weight_percent": 20},
            {"name": "Eventos Naturales (Polvo)", "weight_percent": 10},
            {"name": "Otros", "weight_percent": 5}
        ]
    }
    return jsonify(sources)

# =======================================================
# 4. ENDPOINTS DE GESTIÓN Y CONSULTA HISTÓRICA
# =======================================================

@app.route('/api/v1/thesis/indicators', methods=['GET'])
def get_thesis_indicators():
    """Retorna los resultados simulados de los 4 indicadores operacionales de la tesis."""
    indicators = {
        "TPA_Alcance_Hrs": 18.44, # Promedio Final de GE
        "TPA_Respuesta_Seg": 2.40,  # Promedio Final de GE
        "PPE_Precision_Pct": 245.00, # Promedio Final de GE
        "PSC_Superacion_Pct": 48.65 # Promedio Final de GE
    }
    return jsonify(indicators)


@app.route('/api/v1/history', methods=['GET'])
def get_history():
    """Retorna todos los datos históricos (de la DB simulada) para la tabla del Front-end."""
    # Nota: El Front-end llama a esta ruta para llenar la tabla de gestión
    return jsonify(DB_RECORDS) 


@app.route('/api/v1/history/record', methods=['POST'])
def add_new_record():
    """Añade un nuevo registro a la DB simulada (Función 'Agregar')."""
    global NEXT_ID
    global DB_RECORDS
    
    try:
        data = request.get_json()
        if not all(k in data for k in ('timestamp', 'pm25', 'pm10')):
            return jsonify({"error": "Faltan campos esenciales (timestamp, pm25, pm10)"}), 400

        pm25 = float(data['pm25'])
        pm10 = float(data['pm10'])
        aqi_sim = int(pm25 * 2.5) 
        
        new_record = {
            "id": NEXT_ID,
            "timestamp": data['timestamp'],
            "aqi": aqi_sim,
            "pm25": pm25,
            "pm10": pm10,
            "no2": round(random.uniform(30, 60), 1),
            "co": round(random.uniform(2, 5), 1)
        }
        
        DB_RECORDS.append(new_record)
        NEXT_ID += 1
        
        return jsonify({"message": "Registro añadido con éxito", "id": new_record['id']}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/v1/history/record/last', methods=['DELETE'])
def delete_last_record():
    """Elimina el último registro de la DB simulada (Función 'Eliminar Último')."""
    global DB_RECORDS
    
    if not DB_RECORDS:
        return jsonify({"message": "La base de datos está vacía"}), 404
        
    last_record = DB_RECORDS.pop()
    
    return jsonify({"message": "Último registro eliminado con éxito", "id_eliminado": last_record['id']}), 200

@app.route('/api/v1/history/download', methods=['GET'])
def download_history():
    """Genera y retorna un archivo CSV con datos históricos filtrados."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Simulación de datos para CSV de descarga
    csv_data = "timestamp,AQI,PM2.5,PM10,NO2,CO\n"
    csv_data += "2025-11-15 10:00,75,28.1,45.0,40.5,3.1\n"
    csv_data += "2025-11-15 11:00,90,35.0,55.0,55.2,4.5\n"

    buffer = io.BytesIO(csv_data.encode())
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'AirViewer_Historical_Data_{start_date}_to_{end_date}.csv'
    )


if __name__ == '__main__':
    # El servidor intentará cargar/entrenar el modelo al inicio (antes de app.run)
    if AIR_QUALITY_MODEL is None:
        initialize_ml_components()
        
    print("--- Servidor AirViewer Flask iniciado en http://localhost:5000 ---")
    app.run(debug=True, port=5000)