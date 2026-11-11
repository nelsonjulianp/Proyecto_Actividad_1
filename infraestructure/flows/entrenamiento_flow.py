"""Pipeline de entrenamiento orquestado con Prefect y registrado en MLflow.

Flujo: Ingesta -> Entrenamiento -> Evaluación -> Registro y Promoción a Champion.
"""
import os
import sys
import tempfile
from pathlib import Path

# 1. Asegurar que la raíz del proyecto esté en el path de Python
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split

from domain.modelo import OBJETIVO, calcular_metricas, construir_modelo, resolver_features
from scripts.load_data import cargar_datos as leer_datos

CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
FEATURES_PATH = ROOT_DIR / "configs" / "features.yaml"


@task(name="cargar_datos")
def cargar_datos(ruta_relativa):
    """Carga y valida el archivo CSV mediante el script de carga."""
    ruta_absoluta = ROOT_DIR / ruta_relativa
    print(f"Cargando dataset desde {ruta_absoluta}...")
    return leer_datos(str(ruta_absoluta))


@task(name="entrenar_y_registrar")
def entrenar_y_registrar(df, cfg, candidato):
    """Entrena un modelo candidato, calcula métricas y registra el run en MLflow."""
    tipo = candidato["tipo"]
    parametros = candidato.get("parametros") or {}
    nombre_run = candidato["nombre_run"]

    # Cargar selección de variables desde features.yaml
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, encoding="utf-8") as f:
            feat_cfg = yaml.safe_load(f) or {}
            modo_feat = feat_cfg.get("seleccion", "ALL")
            custom_feat = feat_cfg.get("custom", [])
    else:
        modo_feat = "ALL"
        custom_feat = []

    numericas, categoricas, alias_feat = resolver_features(modo_feat, custom_feat)
    variables = numericas + categoricas

    division = cfg["division"]
    train_df, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )

    X_train, y_train = train_df[variables], train_df[OBJETIVO]
    X_test, y_test = test_df[variables], test_df[OBJETIVO]

    print(f"Entrenando candidato: {nombre_run} ({tipo}) con {len(variables)} features...")
    modelo = construir_modelo(tipo, parametros, numericas, categoricas)
    modelo.fit(X_train, y_train)

    proba = modelo.predict_proba(X_test)[:, 1]
    umbral = cfg["metricas"].get("umbral", 0.5)
    metricas = calcular_metricas(y_test, proba, umbral)
    print(f"Resultados {nombre_run} -> AUC: {metricas['auc']:.4f} | Recall: {metricas['recall']:.4f} | F1: {metricas['f1']:.4f}")

    # Configurar MLflow Tracking
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params({
            "modelo": tipo,
            **parametros,
            "semilla": division["random_state"],
            "features": alias_feat,
            "n_features": len(variables),
            "umbral": umbral,
        })
        mlflow.log_metrics(metricas)

        # Guardar con cloudpickle para serializar LightGBM y XGBoost sin bloqueos de skops
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_local_path = os.path.join(tmp_dir, "model")
            mlflow.sklearn.save_model(
                sk_model=modelo,
                path=model_local_path,
                signature=infer_signature(X_test.head(5), proba[:5]),
                input_example=X_test.head(3),
                serialization_format="cloudpickle",
            )
            mlflow.log_artifacts(model_local_path, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"Run registrado en MLflow: {run_id} | URI del modelo: {model_uri}")

    return run_id, model_uri, metricas


@task(name="promover_si_mejora")
def promover_si_mejora(model_uri, metricas, cfg):
    """Compara el AUC contra el umbral mínimo y contra el champion actual en MLflow."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    nombre_modelo = cfg["registro"]["nombre_modelo"]
    umbral_minimo = cfg["registro"]["umbral_auc"]
    auc_nuevo = metricas["auc"]

    print(f"Evaluando promoción para '{nombre_modelo}' (AUC nuevo: {auc_nuevo:.4f} | Umbral mínimo: {umbral_minimo})...")

    if auc_nuevo < umbral_minimo:
        print(f"AUC {auc_nuevo:.4f} por debajo del mínimo ({umbral_minimo}): no califica para champion.")
        return None

    try:
        champion = client.get_model_version_by_alias(nombre_modelo, "champion")
    except (MlflowException, Exception):
        champion = None

    if champion is not None:
        run_champion = client.get_run(champion.run_id)
        auc_champion = run_champion.data.metrics.get("auc", 0.0)
        print(f"Champion actual: versión {champion.version} con AUC = {auc_champion:.4f}")

        if auc_nuevo <= auc_champion:
            print(f"AUC nuevo ({auc_nuevo:.4f}) no supera al champion actual ({auc_champion:.4f}): el alias no cambia.")
            return None
        print(f"AUC nuevo supera al champion vigente ({auc_nuevo:.4f} > {auc_champion:.4f}). Promoviendo...")
    else:
        print("No existe ningún champion previo. Promoviendo primera versión...")

    version = mlflow.register_model(model_uri, nombre_modelo)
    client.set_registered_model_alias(nombre_modelo, "champion", version.version)
    client.set_model_version_tag(nombre_modelo, version.version, "estado", "aprobado")
    print(f"Versión {version.version} registrada exitosamente como nuevo CHAMPION (AUC: {auc_nuevo:.4f}).")
    return version.version


@flow(name="entrenamiento-fraude", log_prints=True)
def entrenamiento_flow():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = cargar_datos(cfg["datos"]["ruta"])

    resultados = []
    print(f"Iniciando entrenamiento de {len(cfg['candidatos'])} candidatos...")
    for candidato in cfg["candidatos"]:
        run_id, model_uri, metricas = entrenar_y_registrar(df, cfg, candidato)
        resultados.append((run_id, model_uri, metricas))

    # Seleccionar el candidato con mayor AUC de toda la corrida
    mejor_run_id, mejor_uri, mejores_metricas = max(resultados, key=lambda r: r[2]["auc"])
    print(f"Mejor candidato de la ejecución: Run {mejor_run_id} con AUC = {mejores_metricas['auc']:.4f}")

    promover_si_mejora(mejor_uri, mejores_metricas, cfg)
    return mejor_run_id


if __name__ == "__main__":
    entrenamiento_flow()