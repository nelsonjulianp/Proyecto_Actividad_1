"""Pipeline de inferencia por lotes en Prefect que consume el modelo champion."""
import os
import sys
from datetime import date
from pathlib import Path

# Asegurar la raíz del proyecto en el path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task

from scripts.load_data import cargar_datos as leer_datos

CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


@task(name="cargar_champion")
def cargar_champion(cfg):
    """Carga el modelo activo mediante el alias champion desde MLflow."""
    nombre = cfg["registro"]["nombre_modelo"]
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print(f"Buscando modelo con alias 'champion' para '{nombre}' en MLflow...")
    version_info = client.get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado con éxito: {nombre} (Versión {version_info.version}) | Run origen: {version_info.run_id}")
    return modelo, version_info.version, version_info.run_id


@task(name="ingestar_lote")
def ingestar_lote(cfg):
    """Extrae un lote de transacciones para evaluar la inferencia del día."""
    ruta_datos = ROOT_DIR / cfg["datos"]["ruta"]
    df = leer_datos(str(ruta_datos))
    tamano = cfg["inferencia"]["tamano_lote"]
    print(f"Ingestando lote de {tamano} transacciones simuladas...")
    return df.sample(n=tamano, random_state=int(date.today().strftime("%Y%m%d")))


@task(name="predecir_y_guardar")
def predecir_y_guardar(modelo, lote, cfg):
    """Calcula probabilidades de fraude y almacena el resultado en un CSV fechado."""
    cols_modelo = list(modelo.feature_names_in_)
    proba = modelo.predict_proba(lote[cols_modelo])[:, 1]

    resultado = lote.copy()
    resultado["probabilidad_fraude"] = proba.round(4)
    resultado["prediccion_fraude"] = (proba >= cfg["metricas"]["umbral"]).astype(int)

    carpeta = ROOT_DIR / cfg["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_salida = carpeta / f"prediccion_{date.today().isoformat()}.csv"

    resultado.to_csv(ruta_salida, index=False)
    print(f"Predicciones guardadas exitosamente en: {ruta_salida}")
    return str(ruta_salida), len(proba), float(proba.mean())


@task(name="registrar_inferencia_mlflow")
def registrar_inferencia_mlflow(version, run_id_train, ruta_csv, total_filas, proba_media, cfg):
    """Crea un run en MLflow que enlaza la inferencia con la versión del modelo champion."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id_origen", run_id_train)
        mlflow.log_metric("transacciones_evaluadas", total_filas)
        mlflow.log_metric("probabilidad_media_fraude", proba_media)
        mlflow.log_artifact(ruta_csv)
    print("Inferencia registrada en MLflow con éxito.")


@flow(name="inferencia-diaria-fraude", log_prints=True)
def inferencia_flow():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    modelo, version, run_id_train = cargar_champion(cfg)
    lote = ingestar_lote(cfg)
    ruta_csv, filas, proba_media = predecir_y_guardar(modelo, lote, cfg)
    registrar_inferencia_mlflow(version, run_id_train, ruta_csv, filas, proba_media, cfg)


if __name__ == "__main__":
    inferencia_flow()