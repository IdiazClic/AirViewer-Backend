#!/usr/bin/env bash

# 1. Generar la data histórica necesaria
python generate_data.py

# 2. Entrenar el modelo LSTM y guardar el .h5 y .pkl
python ml_model.py
