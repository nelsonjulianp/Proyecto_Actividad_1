"""Tests unitarios del pipeline de inferencia por lotes (sin Prefect ni MLflow)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from domain.modelo import OBJETIVO, VARIABLES
from domain.monitoreo import aplicar_drift
from infraestructure.flows.inferencia_flow import predecir_y_guardar


class ModeloFalsoBatch:
    feature_names_in_ = np.array(VARIABLES)

    def predict_proba(self, X):
        # Devuelve probabilidad de fraude = 0.75 para todas las filas
        return np.tile([0.25, 0.75], (len(X), 1))


def test_inferencia_con_drift_y_guardado_csv(tmp_path):
    rng = np.random.default_rng(42)
    base = pd.DataFrame({
        "Num_Tarjetas": rng.integers(1, 5, 20),
        "Balance_Tj": rng.uniform(100, 5000, 20),
        "Total_Transac": rng.integers(5, 80, 20),
        "Trans_Int": rng.integers(0, 15, 20),
        "Estado": rng.choice(["Alabama", "Texas"], 20),
        "Genero": rng.choice(["Male", "Female"], 20),
        "Tipo_Tarjeta": rng.choice(["Visa", "Mastercard"], 20),
        OBJETIVO: [0] * 18 + [1] * 2,
    })

    lote_drift = aplicar_drift(base, nivel=0.6, semilla=42)
    cfg_prueba = {
        "metricas": {"umbral": 0.5},
        "inferencia": {"carpeta_salida": str(tmp_path)},
    }

    # .fn ejecuta la función pura subyacente de la tarea de Prefect sin levantar servidor
    ruta_csv, total_filas, proba_media = predecir_y_guardar.fn(
        ModeloFalsoBatch(), lote_drift, cfg_prueba
    )

    archivo_salida = Path(ruta_csv)
    assert archivo_salida.exists()
    assert total_filas == 20
    assert proba_media == 0.75

    df_resultado = pd.read_csv(archivo_salida)
    assert "probabilidad_fraude" in df_resultado.columns
    assert "prediccion_fraude" in df_resultado.columns
    assert (df_resultado["prediccion_fraude"] == 1).all()