"""Actividad 3: Despliegue de los pipelines de entrenamiento, inferencia y monitoreo en Prefect."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prefect import serve
from infraestructure.flows.drift_evento_flow import drift_evento_flow
from infraestructure.flows.drift_features_flow import drift_features_flow
from infraestructure.flows.entrenamiento_flow import entrenamiento_flow
from infraestructure.flows.inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(
        name="entrenamiento-fraude-deployment"
    )

    despliegue_inferencia = inferencia_flow.to_deployment(
        name="inferencia-diaria-deployment",
        cron="0 6 * * *",
    )

    despliegue_drift_features = drift_features_flow.to_deployment(
        name="drift-features-deployment"
    )

    despliegue_drift_evento = drift_evento_flow.to_deployment(
        name="drift-evento-deployment"
    )

    print("Publicando y sirviendo despliegues en Prefect (http://localhost:4200)...")
    serve(
        despliegue_entrenamiento,
        despliegue_inferencia,
        despliegue_drift_features,
        despliegue_drift_evento,
    )