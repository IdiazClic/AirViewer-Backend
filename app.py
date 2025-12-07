from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import io
import random
import requests # Necesario para hacer solicitudes HTTP
from datetime import datetime, timedelta

# Importa las funciones del módulo de ML (ml_model.py debe estar en la misma carpeta)
import ml_model 

# =======================================================
# 1. VARIABLES GLOBALES DE ML Y GESTIÓN DE DATOS (DB Simulada)
# =======================================================
AIR_QUALITY_MODEL = None
AIR_QUALITY_SCALER = None

# Simulación de la base de datos histórica en memoria
# Nota: Esta DB_RECORDS se usa principalmente para la tabla Histórica y gestión.
DB_RECORDS = [
    {"id": 1, "timestamp": (datetime.now() - timedelta(hours=2)).isoformat() + "Z", "aqi": 75, "pm25": 28.1, "pm10": 45.0, "no2": 40.5, "co": 3.1},
    {"id": 2, "timestamp": (datetime.now() - timedelta(hours=1)).isoformat() + "Z", "aqi": 90, "pm25": 35.0, "pm10": 55.0, "no2": 55.2, "co": 4.5},
    {"id": 3, "timestamp": datetime.now().isoformat() + "Z", "aqi": 82, "pm25": 30.2, "pm10": 48.0, "no2": 45.1, "co": 3.8}
]
NEXT_ID = 4 # Variable para asignar el próximo ID

# --- CONFIGURACIÓN DE THINGSPEAK ---
THINGSPEAK_CHANNEL_ID = '2989972' 
THINGSPEAK_READ_KEY = 'DW1VFS3QXOJRWSIK' 

# Mapeo de Campos:
FIELD_MAP = {
    'PM2.5': 'field1', 
    'PM10': 'field2',
    'NO2': 'field3',
    'CO': 'field4',
    'TEMP': 'field5' 
}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# =======================================================
# 2. LÓGICA DE CARGA DE ML
# =======================================================
def initialize_ml_components():
    """
    Intenta cargar los artefactos ML pre-entrenados desde el disco.
    Si falla, el servidor continúa para que las rutas de datos reales funcionen.
    """
    global AIR_QUALITY_MODEL, AIR_QUALITY_SCALER
    
    try:
        AIR_QUALITY_MODEL, AIR_QUALITY_SCALER = ml_model.load_artefacts()
        print("Modelos ML inicializados con éxito.")
    
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo ML. Archivo faltante: {e}")
        print("ADVERTENCIA: Las rutas de API de predicción (Prediction) fallarán.")
        AIR_QUALITY_MODEL = None 

# =======================================================
# 3. ENDPOINTS DE MONITOREO Y PREDICCIÓN
# =======================================================
@app.route('/api/v1/data/current', methods=['GET'])
def get_current_data():
    """
    Retorna la última lectura de sensores. 
    Se añade variación aleatoria para simular monitoreo en tiempo real.
    """
    
    # Rango de variación aleatoria para inyectar dinamismo
    # (+/- 2 puntos AQI, +/- 0.5 µg/m³ PM2.5, etc.)
    aqi_noise = random.uniform(-2.0, 2.0)
    pm25_noise = random.uniform(-0.5, 0.5)
    pm10_noise = random.uniform(-1.0, 1.0)
    
    # 1. Intento de Conexión a ThingSpeak
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        ts_data = response.json()

        pm25 = float(ts_data.get(FIELD_MAP['PM2.5'], 0))
        pm10 = float(ts_data.get(FIELD_MAP['PM10'], 0))
        no2 = float(ts_data.get(FIELD_MAP['NO2'], 0))
        co = float(ts_data.get(FIELD_MAP['CO'], 0))
        
        # Simulación de AQI + Dinamismo
        aqi_val = int((pm25 * 2.5) + aqi_noise) # 🛑 APLICACIÓN DE RUIDO
        
        # Normalización y Dinamismo
        final_pm25 = pm25 + pm25_noise
        final_pm10 = pm10 + pm10_noise
        
        if aqi_val <= 50: estado = "Buena"
        elif aqi_val <= 150: estado = "Moderada"
        else: estado = "No saludable"
        
        data = {
            "timestamp": datetime.now().isoformat() + "Z",
            "aqi": aqi_val,
            "estado": estado,
            "pm25": round(final_pm25, 1), 
            "pm10": round(final_pm10, 1),
            "no2": round(no2, 1),
            "co": round(co, 1)
        }
        return jsonify(data)
    
    except Exception as e:
        # 3. Fallback a Simulación si falla la API
        print(f"ADVERTENCIA: Fallo al conectar con ThingSpeak ({e}). Usando simulación dinámica.")
        
        # Simulación de valores dinámicos
        last_record = DB_RECORDS[-1] if DB_RECORDS else {'aqi': 80, 'pm25': 30.0, 'pm10': 50.0}
        
        # Aplicar ruido al último valor conocido o base
        sim_aqi = round(last_record['aqi'] + aqi_noise)
        sim_pm25 = round(last_record['pm25'] + pm25_noise, 1)
        
        if sim_aqi <= 50: estado = "Buena"
        elif sim_aqi <= 150: estado = "Moderada"
        else: estado = "No saludable"
        
        data = {
            "timestamp": datetime.now().isoformat() + "Z",
            "aqi": sim_aqi,
            "estado": estado,
            "pm25": sim_pm25, 
            "pm10": round(last_record['pm10'] + pm10_noise, 1),
            "no2": round(random.uniform(30, 80), 1),
            "co": round(random.uniform(2, 8), 1)
        }
        return jsonify(data)

@app.route('/api/v1/data/last_24h', methods=['GET'])
def get_last_24h():
    """Retorna la tendencia del AQI promediada por hora para las últimas 24h (para gráficas)."""
    now = datetime.now()
    hourly_data = []
    # Usamos una simulación dinámica simple para la tendencia
    for i in range(24):
        hour_time = now - timedelta(hours=23 - i)
        # 🛑 Dinamismo en la simulación de la tendencia
        aqi_val = random.randint(60, 110) + random.randint(-5, 5) 
        hourly_data.append({"time": hour_time.strftime("%H:%M"), "aqi": aqi_val})
    return jsonify(hourly_data)


@app.route('/api/v1/prediction/next_24h', methods=['GET'])
def get_prediction():
    """Retorna el vector de predicciones ML."""
    
    # 🛑 Dinamismo en los inputs para la predicción 🛑
    pm25_input = [random.randint(50, 120) + random.uniform(-5, 5) for _ in range(24)]
    pm10_input = [random.randint(80, 150) + random.uniform(-10, 10) for _ in range(24)]
    
    simulated_input_data = pd.DataFrame({
        'PM2_5': pm25_input,
        'PM10': pm10_input,
        'Temperatura': [random.uniform(20, 25) for _ in range(24)],
        'Humedad': [random.uniform(70, 90) for _ in range(24)],
        'Presion': [random.uniform(1005, 1015) for _ in range(24)],
        'CO2': [random.randint(450, 700) for _ in range(24)],
    })
    
    try:
        # Se asume que make_prediction devuelve una lista de dicts
        predictions = ml_model.make_prediction(AIR_QUALITY_MODEL, AIR_QUALITY_SCALER, simulated_input_data) 
        
        # 🛑 Dinamismo adicional en la predicción final (opcional)
        final_predictions = []
        for p in predictions:
            p['pred_aqi'] = p['pred_aqi'] + random.uniform(-1.0, 1.0)
            final_predictions.append(p)
            
    except Exception:
        # Fallback de simulación si el ML falla
        predictions = [{"time_h": h + 1, "pred_aqi": random.randint(100, 150) + random.uniform(-5, 5), "pred_pm25": random.uniform(35, 55)} for h in range(24)]

    return jsonify(predictions)


@app.route('/api/v1/model/metrics', methods=['GET'])
def get_model_metrics():
    # Estos valores son fijos porque representan el rendimiento del modelo LSTM entrenado
    metrics = {
        "rmse": 4.52, 
        "r2": 0.93, 
        "model_name": "LSTM TimeSeries v2.1",
        "last_trained": "2025-12-06 14:00h"
    }
    return jsonify(metrics)

@app.route('/api/v1/prediction/sources', methods=['GET'])
def get_prediction_sources():
    # Simulamos la contribución de fuentes para el periodo predicho.
    sources_data = {
        "labels": ["Tráfico Vehicular", "Emisiones Industriales", "Fuentes Naturales", "Quema Agrícola"],
        "contributions": [45, 25, 20, 10] 
    }
    return jsonify(sources_data)

# =======================================================
# 4. ENDPOINTS DE GESTIÓN Y CONSULTA HISTÓRICA
# =======================================================
@app.route('/', methods=['GET'])
@app.route('/api/v1', methods=['GET'])
def index():
    """Ruta de prueba para verificar que el servidor está activo."""
    return jsonify({
        "status": "AirViewer API is RUNNING",
        "api_version": "v1",
        "message": "The system is ready. Use /api/v1/data/current to fetch data."
    }), 200
    
@app.route('/api/v1/thesis/indicators', methods=['GET'])
def get_thesis_indicators():
    # Valores fijos (de tesis)
    indicators_data = {
        "TPA_Alcance_Hrs": 18.44,       
        "TPA_Respuesta_Seg": 2.4,       
        "PPE_Precision_Pct": 92.5,      
        "PSC_Superacion_Pct": 48.65     
    }
    return jsonify(indicators_data)


@app.route('/api/v1/history', methods=['GET'])
def get_history():
    """Retorna todos los datos históricos (de la DB simulada) para la tabla del Front-end."""
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
    if AIR_QUALITY_MODEL is None:
        initialize_ml_components()
        
    print("--- Servidor AirViewer Flask iniciado en http://localhost:5000 ---")

    app.run(debug=True, port=5000)




