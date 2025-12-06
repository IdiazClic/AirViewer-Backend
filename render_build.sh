#!/usr/bin/env bash

# 🛑 NUEVO: CREAR LAS CARPETAS NECESARIAS
mkdir -p data
mkdir -p model

# 1. Generar la data histórica necesaria (Ahora la carpeta 'data' existe)
python generate_data.py

# 2. Entrenar el modelo LSTM y guardar el .h5 y .pkl (Ahora la carpeta 'model' existe)
python ml_model.py
