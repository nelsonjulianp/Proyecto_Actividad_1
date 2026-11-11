# Detección de Fraude - MLOps en Producción (Actividad 2)

Este proyecto implementa un sistema automatizado de Machine Learning (MLOps) para la detección de transacciones fraudulentas. La arquitectura está diseñada para entrenar modelos candidatos, evaluar su rendimiento (AUC), gestionar su ciclo de vida y exponer el mejor modelo en tiempo real.

## 🛠️ Tecnologías y Arquitectura

* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost.

* **Orquestación:** Prefect (Pipelines de entrenamiento e inferencia batch).

* **Tracking y Model Registry:** MLflow (Registro de experimentos, métricas, artefactos y promoción del modelo `champion`).

* **Servicio Web (API):** FastAPI y Uvicorn (Inferencia en tiempo real).

* **Gestión de Entorno:** Poetry.

## 📂 Estructura del Repositorio

```
mi-proyecto-mlops/
├── configs/
│   ├── config.yaml          # Hiperparámetros, umbrales y modelos candidatos
│   └── features.yaml        # Selección dinámica de variables predictoras
├── data/
│   ├── train.csv            # Dataset principal de transacciones
│   └── predicciones/        # Salida del pipeline de inferencia diaria
├── domain/
│   ├── __init__.py
│   └── modelo.py            # Lógica de ML (Preprocesamiento y Estimador)
├── infraestructure/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          # API en tiempo real con FastAPI
│   └── flows/
│       ├── __init__.py
│       ├── entrenamiento_flow.py # Pipeline: Entrena y corona al champion en MLflow
│       ├── inferencia_flow.py    # Pipeline: Predice lote diario con el champion
│       └── desplegar.py          # Despliegue de los flujos en Prefect (Cron)
├── scripts/
│   ├── __init__.py
│   └── load_data.py         # Ingesta y validación de columnas
├── pyproject.toml           # Dependencias del proyecto
└── README.md                # Este documento

```

## 🚀 Requisitos Previos

1. Python 3.12 o superior.

2. Poetry instalado (`pip install poetry`).

3. Docker Desktop (para levantar los contenedores de MLflow y Prefect).

## ⚙️ Instalación y Configuración

1. **Clonar el repositorio y preparar el entorno:**

   ```
   poetry install
   
   ```

2. **Levantar los servicios base (Prefect y MLflow):**
   Navega a la carpeta `services/` (provista en la Actividad 1) y ejecuta:

   ```
   docker compose up -d
   
   ```

   * UI de MLflow: `http://localhost:5000`

   * UI de Prefect: `http://localhost:4200`

## 🏃‍♂️ Guía de Ejecución

Abre una terminal en la raíz del proyecto y configura las variables de entorno (ejemplo en PowerShell):

```
$env:PYTHONPATH="."
$env:PREFECT_API_URL="http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI="http://localhost:5000"

```

### 1. Entrenamiento y Promoción del Modelo

Este flujo entrena todos los candidatos definidos en `configs/config.yaml` (Regresión Logística, LightGBM, XGBoost) y promueve al mejor como **`champion`** en MLflow si supera el umbral mínimo de AUC.

```
poetry run python infraestructure/flows/entrenamiento_flow.py

```

### 2. Inferencia por Lotes Diaria

Carga automáticamente la versión activa del modelo `champion` de MLflow, ingesta un lote de 100 transacciones simuladas, genera las predicciones y guarda un archivo CSV fechado en `data/predicciones/`.

```
poetry run python infraestructure/flows/inferencia_flow.py

```

### 3. Programación de Despliegues (Cron)

Deja este script corriendo para registrar el despliegue manual del entrenamiento y el despliegue automático diario de la inferencia (a las 06:00 UTC).

```
poetry run python infraestructure/flows/desplegar.py

```

### 4. API en Tiempo Real (FastAPI)

En una nueva terminal (exportando `$env:MLFLOW_TRACKING_URI`), levanta el servidor web. La API descargará la última versión del modelo `champion` para atender transacciones individuales.

```
poetry run uvicorn infraestructure.api.main:app --port 8000 --reload

```

Ingresa a `http://localhost:8000/docs` para interactuar con la API de forma gráfica y evaluar el nivel de fraude de una transacción.