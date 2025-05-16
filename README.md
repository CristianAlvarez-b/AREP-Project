# Proyecto de Predicción de Demanda con XGBoost y SageMaker
# Desarrollado por: Cristian Alvarez, Juan David Contreras
## 📌 Descripción
Sistema de predicción de demanda de productos que integra datos históricos de ventas con información meteorológica en tiempo real, desplegado en AWS SageMaker.

## 📂 Estructura del Proyecto
```
  ├── dataset/                   # Datos históricos
  │   └── historico.csv          # Dataset principal
  ├── img/                       # Gráficos y visualizaciones
  │   ├── correlacion.png        # Matriz de correlación
  │   ├── errores.png            # Análisis de errores
  │   └── ventas_comparacion.png # Predicciones vs reales
  ├── src/                       # Código fuente
  │   ├── inference.py           # Script para despliegue
  │   ├── train.py               # Entrenamiento del modelo
  │   └── weather_api.py         # Integración con API meteorológica
  ├── despliegue.ipynb           # Notebook para despliegue
  ├── label_encoder.pkl          # Encoder para categorías
  ├── modelo_xgb.bin             # Modelo entrenado (formato binario)
  ├── productos.json             # Catálogo de productos
  ├── requirements.txt           # Dependencias
  └── sagemaker_notebook.ipynb   # Notebook de SageMaker
```
## ⚙️ Requisitos
- Python 3.8+
- AWS CLI configurado
- Cuenta de AWS con permisos para SageMaker

## 🚀 Instalación

Clonar el repositorio:

```bash
git clone [URL_DEL_REPOSITORIO]
cd nombre-del-repositorio
```
Instalar dependencias:

```bash
pip install -r requirements.txt
Configurar credenciales AWS:
```
```bash
aws configure
```
## 🧠 Entrenamiento del Modelo
Ejecutar el script de entrenamiento:

```bash
python src/train.py --data dataset/historico.csv --output modelo_xgb.bin
```
Argumentos opcionales:

--epochs: Número de iteraciones (default: 100)

--test-size: Porcentaje para test (default: 0.2)

## ☁️ Despliegue en SageMaker
Preparar el paquete:

```bash
tar -czvf model.tar.gz modelo_xgb.bin label_encoder.pkl productos.json src/inference.py
```
Subir a S3:

```bash
aws s3 cp model.tar.gz s3://tu-bucket-s3/model.tar.gz
```
Desplegar usando el notebook despliegue.ipynb o ejecutar:

```python
from sagemaker import Model
from sagemaker.image_uris import retrieve

image_uri = retrieve("xgboost", region="us-east-2", version="1.7-1")
model = Model(model_data="s3://tu-bucket-s3/model.tar.gz",
              image_uri=image_uri,
              role="AmazonSageMaker-ExecutionRole",
              entry_point="inference.py")

predictor = model.deploy(initial_instance_count=1, 
                        instance_type="ml.m5.large",
                        endpoint_name="demanda-productos")
```
🌦️ Uso del Endpoint
Ejemplo de petición:

```python
import boto3
import json

client = boto3.client("sagemaker-runtime", region_name="us-east-2")

response = client.invoke_endpoint(
    EndpointName="demanda-productos",
    ContentType="application/json",
    Body=json.dumps({
        "api_key": "TU_API_KEY_OPENWEATHER",
        "ciudad": "Madrid",
        "productos_stock": {
            "Camiseta": 10,
            "Pantalon": 5
        }
    })
)

print(json.loads(response["Body"].read().decode()))
```
## 💾 Datos

historico.csv: Contiene:

- fecha: Fecha de registro

- producto: Nombre del producto

- ventas: Unidades vendidas

- stock: Unidades en inventario

- Datos meteorológicos (temp, lluvia)

## 📊 Visualizaciones
Ver directorio img/ para análisis de:

- Correlación entre variables

- Error de predicción

- Comparación ventas reales vs predicciones

## 🔄 Mantenimiento
Apagar endpoint cuando no se use:

```python
client = boto3.client("sagemaker")
client.delete_endpoint(EndpointName="demanda-productos")
```
