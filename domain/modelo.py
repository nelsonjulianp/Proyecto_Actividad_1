"""Lógica de negocio del modelo de fraude: preprocesamiento, pipeline y métricas.

No utiliza Prefect ni MLflow: se puede probar de forma aislada.
"""
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Variables del dataset (se descarta 'ID')
NUMERICAS = ["Num_Tarjetas", "Balance_Tj", "Total_Transac", "Trans_Int"]
CATEGORICAS = ["Estado", "Genero", "Tipo_Tarjeta"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "Fraude"


def resolver_features(modo=None, lista_custom=None):
    """Resuelve cuáles variables usar según la configuración de configs/features.yaml."""
    if modo is None:
        with open("configs/features.yaml", encoding="utf-8") as f:
            cfg_feat = yaml.safe_load(f)
            modo = cfg_feat.get("seleccion", "ALL")
            lista_custom = cfg_feat.get("custom", [])

    if modo == "ALL":
        elegidas = VARIABLES
    elif modo == "NUMERIC":
        elegidas = NUMERICAS
    elif modo == "CUSTOM":
        desconocidas = [v for v in lista_custom if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"Variables desconocidas en features.yaml: {desconocidas}")
        elegidas = lista_custom
    else:
        raise ValueError(f"Modo de selección no soportado: {modo}")

    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return numericas, categoricas, modo


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Construye un Pipeline de scikit-learn: preprocesador (escalado + one-hot) + estimador."""
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "lightgbm":
        estimador = LGBMClassifier(**parametros)
    elif tipo == "xgboost":
        estimador = XGBClassifier(**parametros)
    else:
        raise ValueError(f"Modelo '{tipo}' no soportado.")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))

    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def calcular_metricas(y_true, proba, umbral=0.5):
    """Calcula las métricas de clasificación binaria."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }