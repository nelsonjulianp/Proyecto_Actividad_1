"""Actividad 3: Monitoreo de drift de evento (variable objetivo 'Fraude')."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import yaml
from prefect import flow, task

from domain.modelo import OBJETIVO
from domain.monitoreo import dividir, drift_de_target, validar_lote
from scripts.load_data import cargar_datos

CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


@task(name="obtener_referencia_target")
def obtener_referencia(cfg):
    """Extrae la variable objetivo 'Fraude' de la partición de entrenamiento."""
    ruta_datos = ROOT_DIR / cfg["datos"]["ruta"]
    entrenamiento, _ = dividir(cargar_datos(str(ruta_datos)), cfg["division"])
    return entrenamiento[OBJETIVO]


@task(name="evaluar_drift_evento")
def evaluar_evento(referencia_objetivo, cfg):
    """Compara la tasa de fraude del lote contra entrenamiento. Rechaza lotes sin etiquetas."""
    ruta_lote = ROOT_DIR / cfg["monitoreo"]["lote_actual"]
    lote = pd.read_csv(ruta_lote)
    validar_lote(lote)

    resultado = drift_de_target(
        referencia_objetivo,
        lote[OBJETIVO],
        cfg["monitoreo"]["alfa_evento"],
    )
    print(
        f"Tasa de {OBJETIVO}: entrenamiento {resultado['tasa_referencia']:.2%}, "
        f"lote {resultado['tasa_lote']:.2%} (p-valor {resultado['p_valor']:.4f})"
    )
    return resultado


@task(name="evaluar_alerta_evento")
def alerta(resultado, alfa):
    """Emite alerta si el p-valor es menor al nivel de significación alfa."""
    hay_alerta = resultado["drift"]
    if hay_alerta:
        print(f"ALERTA: drift del target '{OBJETIVO}' detectado (p-valor {resultado['p_valor']:.4f} < {alfa})")
    else:
        print(f"Sin alerta: drift del target '{OBJETIVO}' estable (p-valor {resultado['p_valor']:.4f} >= {alfa})")
    return hay_alerta


@flow(name="drift-evento-fraude", log_prints=True)
def drift_evento_flow():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ref = obtener_referencia(cfg)
    resultado = evaluar_evento(ref, cfg)
    return alerta(resultado, cfg["monitoreo"]["alfa_evento"])


if __name__ == "__main__":
    drift_evento_flow()