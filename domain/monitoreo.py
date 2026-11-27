"""Lógica pura de monitoreo y detección de drift (sin Prefect ni MLflow)."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from domain.modelo import NUMERICAS, OBJETIVO

# Subida máxima de la tasa de fraude (0.3 = +30 puntos porcentuales cuando nivel = 1.0)
SUBIDA_TASA_MAXIMA = 0.3


def dividir(df, division):
    """Parte los datos de forma estratificada igual que el entrenamiento. Devuelve (entrenamiento, prueba)."""
    return train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )


def aplicar_drift(df, nivel, semilla=42):
    """Emula drift controlado en variables numéricas y en la tasa del objetivo."""
    if not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe estar entre 0 y 1, recibido: {nivel}")

    out = df.copy()
    rng = np.random.default_rng(semilla)

    # 1. Drift en features numéricas (desplaza la media en 2 desviaciones estándar)
    n = round(nivel * len(NUMERICAS))
    if n > 0:
        columnas_afectadas = rng.choice(NUMERICAS, size=n, replace=False)
        for columna in columnas_afectadas:
            std_val = out[columna].std()
            if pd.notna(std_val) and std_val > 0:
                out[columna] = out[columna] + 2 * std_val

    # 2. Drift en el target (incrementa la proporción de fraudes)
    out[OBJETIVO] = _subir_tasa(out[OBJETIVO], nivel * SUBIDA_TASA_MAXIMA, rng)
    return out


def _subir_tasa(objetivo, subida, rng):
    """Convierte registros negativos (0) en positivos (1) al azar hasta subir la tasa."""
    objetivo_int = objetivo.astype(int)
    tasa_actual = objetivo_int.mean()
    nueva_tasa = min(tasa_actual + subida, 1.0)

    pos_objetivo = round(nueva_tasa * len(objetivo_int))
    faltan = pos_objetivo - int(objetivo_int.sum())

    if faltan <= 0:
        return objetivo

    subidos = objetivo.copy()
    indices_ceros = objetivo_int.index[objetivo_int == 0]
    faltan = min(faltan, len(indices_ceros))

    if faltan > 0:
        a_cambiar = rng.choice(indices_ceros, size=faltan, replace=False)
        subidos.loc[a_cambiar] = 1

    return subidos


def fraccion_desde_evidently(resultado):
    """Extrae la fracción de columnas con drift desde el diccionario de Evidently (`snapshot.dict()`)."""
    for metrica in resultado["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            return float(metrica["value"]["share"])
    raise ValueError("El resultado de Evidently no incluye la métrica DriftedColumnsCount")


def validar_lote(lote):
    """Valida que el lote incluya la columna objetivo 'Fraude' sin valores nulos."""
    if OBJETIVO not in lote.columns:
        raise ValueError(f"El lote no tiene la columna '{OBJETIVO}': no se puede calcular el drift de evento")
    if lote[OBJETIVO].isna().any():
        raise ValueError(f"El lote tiene valores nulos en '{OBJETIVO}': no se puede calcular el drift de evento")


def drift_de_target(referencia, lote, alfa):
    """Compara la proporción de Fraude entre entrenamiento y lote mediante test chi-cuadrado."""
    ref_int = referencia.astype(int)
    lote_int = lote.astype(int)

    pos_ref = int(ref_int.sum())
    neg_ref = len(ref_int) - pos_ref
    pos_lote = int(lote_int.sum())
    neg_lote = len(lote_int) - pos_lote

    _, p_valor, _, _ = chi2_contingency([[pos_ref, neg_ref], [pos_lote, neg_lote]])
    return {
        "tasa_referencia": pos_ref / len(ref_int),
        "tasa_lote": pos_lote / len(lote_int),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }