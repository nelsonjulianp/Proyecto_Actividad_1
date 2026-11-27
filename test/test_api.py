"""Tests de la API FastAPI de fraude usando un modelo falso (sin MLflow)."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain.modelo import VARIABLES

TRANSACCION_EJEMPLO = {
    "Num_Tarjetas": 1,
    "Balance_Tj": 2000.0,
    "Total_Transac": 31,
    "Trans_Int": 9,
    "Estado": "Alabama",
    "Genero": "Male",
    "Tipo_Tarjeta": "American Express",
}


class ModeloFalso:
    feature_names_in_ = np.array(VARIABLES)

    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: SimpleNamespace(
            get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(version="7")
        ),
    )
    sys.modules.pop("infraestructure.api.main", None)

    import infraestructure.api.main as main

    return TestClient(main.app)


def test_health_y_predict(client):
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["version_activa"] == "7"

    resp_predict = client.post("/predict", json=TRANSACCION_EJEMPLO)
    assert resp_predict.status_code == 200
    assert resp_predict.json() == {
        "probabilidad_fraude": 0.8,
        "es_fraude": 1,
        "version_modelo_usada": "7",
    }