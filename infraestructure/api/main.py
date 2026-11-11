"""Actividad 2: API de inferencia en tiempo real con FastAPI.

Carga el modelo con el alias 'champion' de MLflow al iniciar.
Ejecución:
    poetry run uvicorn infraestructure.api.main:app --port 8000 --reload
Documentación Swagger:
    http://localhost:8000/docs
"""
import os
import sys
from pathlib import Path

# Asegurar la raíz del proyecto en el path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# Lectura de configuración
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(TRACKING_URI)

# Cargar el modelo champion al arrancar la API
try:
    client = MlflowClient()
    champion_info = client.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    VERSION = champion_info.version
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    print(f"API cargó exitosamente el modelo '{NOMBRE_MODELO}' (Versión {VERSION})")
except Exception as e:
    modelo = None
    VERSION = "No registrado"
    print(f"Advertencia: No se pudo cargar el champion: {e}")

app = FastAPI(
    title="API de Detección de Fraude",
    description="Servicio en tiempo real que consume el modelo champion de MLflow.",
    version="1.0.0",
)


class Transaccion(BaseModel):
    """Esquema de datos para una transacción financiera individual."""
    Num_Tarjetas: int = Field(..., example=1)
    Balance_Tj: float = Field(..., example=2000.0)
    Total_Transac: int = Field(..., example=31)
    Trans_Int: int = Field(..., example=9)
    Estado: str = Field(..., example="Alabama")
    Genero: str = Field(..., example="Male")
    Tipo_Tarjeta: str = Field(..., example="American Express")


@app.get("/health")
def health():
    """Endpoint de monitoreo y estado del servicio."""
    return {
        "estado": "operativo",
        "modelo": NOMBRE_MODELO,
        "version_activa": VERSION,
    }


@app.post("/predict")
def predict(transaccion: Transaccion):
    """Endpoint para evaluar la probabilidad de fraude de una transacción."""
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="No hay un modelo champion disponible en el servidor de MLflow."
        )

    # Convertir JSON a DataFrame con las columnas del modelo
    datos = pd.DataFrame([transaccion.model_dump()])
    datos_alineados = datos[list(modelo.feature_names_in_)]

    proba = float(modelo.predict_proba(datos_alineados)[0, 1])
    umbral = cfg["metricas"].get("umbral", 0.5)

    return {
        "probabilidad_fraude": round(proba, 4),
        "es_fraude": int(proba >= umbral),
        "version_modelo_usada": VERSION,
    }