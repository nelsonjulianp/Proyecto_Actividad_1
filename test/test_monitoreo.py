"""Tests unitarios de emulación y detección de drift (sin servicios externos)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import pytest

from domain.modelo import NUMERICAS, OBJETIVO
from domain.monitoreo import (
    SUBIDA_TASA_MAXIMA,
    aplicar_drift,
    drift_de_target,
    fraccion_desde_evidently,
    validar_lote,
)


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({columna: rng.normal(10, 2, 500) for columna in NUMERICAS})
    df[OBJETIVO] = rng.choice([0, 1], size=500, p=[0.94, 0.06])
    return df


def test_drift_cero_no_cambia_los_datos(datos):
    assert aplicar_drift(datos, 0.0).equals(datos)


def test_drift_nivel_invalido_falla(datos):
    with pytest.raises(ValueError):
        aplicar_drift(datos, 1.5)


def test_drift_sube_la_tasa_del_objetivo_en_proporcion_al_nivel(datos):
    tasa = datos[OBJETIVO].mean()
    lote = aplicar_drift(datos, 0.6)
    assert lote[OBJETIVO].mean() == pytest.approx(tasa + 0.6 * SUBIDA_TASA_MAXIMA, abs=0.01)


def test_drift_cambia_la_fraccion_indicada_de_variables(datos):
    lote = aplicar_drift(datos, 0.6)
    cambiadas = [c for c in NUMERICAS if not np.allclose(lote[c], datos[c])]
    assert len(cambiadas) == round(0.6 * len(NUMERICAS))


def test_fraccion_desde_evidently():
    resultado = {
        "metrics": [
            {"metric_name": "ValueDrift(column=Balance_Tj)", "value": 0.01},
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 2, "share": 0.2857}},
        ]
    }
    assert fraccion_desde_evidently(resultado) == pytest.approx(0.2857)


def test_validar_lote_exige_objetivo(datos):
    with pytest.raises(ValueError, match=OBJETIVO):
        validar_lote(datos.drop(columns=[OBJETIVO]))


def test_validar_lote_rechaza_nulos_en_objetivo(datos):
    lote_nulo = datos.copy()
    lote_nulo.loc[0, OBJETIVO] = np.nan
    with pytest.raises(ValueError, match="nulos"):
        validar_lote(lote_nulo)


def test_drift_de_target_detecta_cambio_de_tasa():
    referencia = pd.Series([0] * 940 + [1] * 60)
    lote = pd.Series([0] * 760 + [1] * 240)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is True


def test_drift_de_target_no_alerta_con_la_misma_tasa():
    referencia = pd.Series([0] * 940 + [1] * 60)
    lote = pd.Series([0] * 470 + [1] * 30)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is False