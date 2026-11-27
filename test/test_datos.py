"""Tests de datos: esquema, nulos y rangos del dataset de fraude."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from domain.modelo import NUMERICAS, OBJETIVO
from scripts.load_data import COLUMNAS_MODELO, cargar_datos


@pytest.fixture(scope="module")
def df():
    return cargar_datos(str(ROOT_DIR / "data" / "train.csv"))


def test_tiene_todas_las_columnas_esperadas(df):
    assert list(df.columns) == COLUMNAS_MODELO


def test_no_hay_nulos(df):
    assert int(df.isna().sum().sum()) == 0


def test_objetivo_es_binario(df):
    assert set(df[OBJETIVO].unique()) <= {0, 1}


def test_rangos_numericos_no_negativos(df):
    for col in NUMERICAS:
        assert (df[col] >= 0).all(), f"Valores negativos encontrados en {col}"