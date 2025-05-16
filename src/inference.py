import os
import json
import joblib
import pandas as pd
import requests
import xgboost as xgb
from datetime import datetime

def model_fn(model_dir):
    """Carga el modelo y artefactos"""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'modelo_xgb.bin'))
    
    le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    
    with open(os.path.join(model_dir, 'productos.json')) as f:
        productos_conocidos = json.load(f)
    
    return {
        'model': model,
        'label_encoder': le,
        'productos_conocidos': productos_conocidos
    }

def input_fn(request_body, request_content_type):
    """Procesa la entrada de la API"""
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Content type no soportado: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    model = model_artifacts['model']
    le = model_artifacts['label_encoder']
    productos_validos = model_artifacts['productos_conocidos']

    estaciones = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    mes = datetime.now().month
    reporte = []

    features = ['producto', 'stock', 'temp', 'lluvia', 'mes', 'estacion']
    
    for item in input_data['productos']:
        producto = str(item['producto']).strip()
        if producto not in productos_validos:
            continue
        
        try:
            prod_cod = le.transform([producto])[0]
            estacion = estaciones.get(item['estacion'], 0)

            data = {
                'producto': [prod_cod],
                'stock': [item['stock']],
                'temp': [item['temp']],
                'lluvia': [item['lluvia']],
                'mes': [mes],
                'estacion': [estacion]
            }

            dmatrix = xgb.DMatrix(pd.DataFrame(data)[features])
            pred = model.predict(dmatrix)[0]

            reporte.append({
                'producto': producto,
                'stock_actual': item['stock'],
                'demanda_esperada': round(pred),
                'comprar': max(0, round(pred - item['stock']))
            })

        except Exception as e:
            print(f"Error procesando {producto}: {str(e)}")
            continue

    return reporte


def output_fn(prediction, content_type):
    """Formatea la salida"""
    return json.dumps(prediction, ensure_ascii=False)