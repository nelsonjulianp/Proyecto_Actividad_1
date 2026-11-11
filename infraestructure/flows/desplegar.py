"""Actividad 2: Despliegue de los pipelines en Prefect con serve.

- entrenamiento-fraude-deployment: Sin horario cron, lanzado a mano desde la UI.
- inferencia-diaria-deployment: Horario cron diario a las 06:00 UTC ("0 6 * * *").
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prefect import serve
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

    print("Publicando y sirviendo despliegues en Prefect (http://localhost:4200)...")
    serve(despliegue_entrenamiento, despliegue_inferencia)