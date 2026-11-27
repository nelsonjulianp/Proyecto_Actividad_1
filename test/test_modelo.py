"""Tests unitarios de la lógica del modelo de fraude (sin Prefect ni MLflow)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import pytest

from domain.modelo import CATEGORICAS, NUMERICAS, VARIABLES, calcular_metricas, construir_modelo, resolver_features


def test_resolver_features_all():
    num, cat, modo = resolver_features(modo="ALL")
    assert num == NUMERICAS
    assert cat == CATEGORICAS
    assert modo == "ALL"


def test_resolver_features_invalida_lanza_error():
    with pytest.raises(ValueError, match="desconocidas"):
        resolver_features(modo="CUSTOM", lista_custom=["Columna_Inexistente"])


def test_construir_y_entrenar_pipeline():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Num_Tarjetas": rng.integers(1, 5, 40),
        "Balance_Tj": rng.uniform(0, 10000, 40),
        "Total_Transac": rng.integers(1, 100, 40),
        "Trans_Int": rng.integers(0, 20, 40),
        "Estado": rng.choice(["Alabama", "Texas"], 40),
        "Genero": rng.choice(["Male", "Female"], 40),
        "Tipo_Tarjeta": rng.choice(["Visa", "Mastercard"], 40),
    })
    y = np.array([0] * 30 + [1] * 10)

    pipe = construir_modelo("logistic_regression", {"max_iter": 200, "random_state": 42}, NUMERICAS, CATEGORICAS)
    pipe.fit(df[VARIABLES], y)
    proba = pipe.predict_proba(df[VARIABLES])[:, 1]

    assert proba.shape == (40,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    metricas = calcular_metricas(y, proba, umbral=0.5)
    assert "auc" in metricas
    assert 0.0 <= metricas["auc"] <= 1.0