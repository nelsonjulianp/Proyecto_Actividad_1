"""Actividad 3: Monitoreo de drift de features con Evidently."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task

from domain.modelo import VARIABLES
from domain.monitoreo import dividir, fraccion_desde_evidently
from scripts.load_data import cargar_datos

CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


@task(name="obtener_referencia_features")
def obtener_referencia(cfg):
    """Carga el conjunto de entrenamiento como línea base de comparación."""
    ruta_datos = ROOT_DIR / cfg["datos"]["ruta"]
    entrenamiento, _ = dividir(cargar_datos(str(ruta_datos)), cfg["division"])
    return entrenamiento[VARIABLES]


@task(name="calcular_drift_features")
def calcular_drift(datos_referencia, cfg):
    """Ejecuta DataDriftPreset de Evidently, guarda el reporte HTML y devuelve la fracción con drift."""
    ruta_lote = ROOT_DIR / cfg["monitoreo"]["lote_actual"]
    lote = pd.read_csv(ruta_lote)

    snapshot = Report([DataDriftPreset()]).run(
        reference_data=datos_referencia,
        current_data=lote[VARIABLES],
    )

    carpeta = ROOT_DIR / cfg["monitoreo"]["carpeta_reportes"]
    carpeta.mkdir(parents=True, exist_ok=True)
    nivel = cfg["inferencia"]["nivel_drift"]
    ruta_html = carpeta / f"drift_features_{ruta_lote.stem}_nivel_{nivel}.html"
    snapshot.save_html(str(ruta_html))
    print(f"Reporte de drift de features guardado en: {ruta_html}")

    return fraccion_desde_evidently(snapshot.dict())


@task(name="evaluar_alerta_features")
def alerta(fraccion, umbral):
    """Emite alerta si la fracción de variables con drift supera el umbral configurado."""
    hay_alerta = fraccion > umbral
    if hay_alerta:
        print(f"ALERTA: drift de features en {fraccion:.1%} de las variables (umbral {umbral:.1%})")
    else:
        print(f"Sin alerta: drift de features en {fraccion:.1%} de las variables (umbral {umbral:.1%})")
    return hay_alerta


@flow(name="drift-features-fraude", log_prints=True)
def drift_features_flow():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ref = obtener_referencia(cfg)
    fraccion = calcular_drift(ref, cfg)
    return alerta(fraccion, cfg["monitoreo"]["umbral_drift"])


if __name__ == "__main__":
    drift_features_flow()