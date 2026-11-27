# Detección de Fraude - MLOps en Producción (Actividad 3: Monitoreo, Drift y CI)

Sistema automatizado de Machine Learning (MLOps) para la detección de transacciones fraudulentas. Incluye entrenamiento y promoción de modelos (`champion`), inferencia por lotes con emulación controlada de *drift*, monitoreo estadístico de *feature drift* (Evidently) y *target drift* (test chi-cuadrado), servicio REST en tiempo real y suite de pruebas automatizadas.

## Tecnologías y Arquitectura

* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost.
* **Monitoreo y Drift:** Evidently AI y SciPy (`chi2_contingency`).
* **Orquestación:** Prefect (Pipelines de entrenamiento, inferencia y detección de drift).
* **Tracking y Model Registry:** MLflow (Registro de experimentos, artefactos y alias `champion`).
* **Servicio Web (API):** FastAPI y Uvicorn (Inferencia en tiempo real).
* **Testing y CI:** Pytest, HTTPX y GitHub Actions.
* **Gestión de Entorno:** Poetry.

## Estructura del Repositorio

```text
mi-proyecto-mlops/
├── .github/workflows/
│   └── ci.yml                     # Pipeline de Integración Continua (GitHub Actions)
├── configs/
│   ├── config.yaml                # Hiperparámetros, MLflow, nivel de drift y umbrales de alerta
│   └── features.yaml              # Selección dinámica de variables predictoras
├── data/
│   ├── train.csv                  # Dataset principal de transacciones (sep=';')
│   └── predicciones/              # Salida del pipeline de inferencia por lotes
├── domain/
│   ├── __init__.py
│   ├── modelo.py                  # Lógica pura de ML (preprocesamiento, estimador, métricas)
│   └── monitoreo.py               # Lógica pura de emulación y detección de drift
├── infraestructure/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                # API en tiempo real con FastAPI
│   └── flows/
│       ├── __init__.py
│       ├── entrenamiento_flow.py  # Entrena candidatos y corona al champion en MLflow
│       ├── inferencia_flow.py     # Puntúa lote de prueba con drift emulado y guarda copia
│       ├── drift_features_flow.py # Evalúa drift de variables con Evidently y genera HTML
│       ├── drift_evento_flow.py   # Evalúa drift de target (Fraude) con test chi-cuadrado
│       └── desplegar.py           # Publica todos los despliegues en Prefect
├── lotes/                         # Lote actual puntuado para monitoreo (ignorado en Git)
├── reports/                       # Reportes HTML de Evidently por lote (ignorado en Git)
├── scripts/
│   ├── __init__.py
│   └── load_data.py               # Ingesta y validación de esquema del dataset
├── tests/
│   ├── __init__.py
│   ├── test_datos.py              # Tests de esquema, nulos y rangos
│   ├── test_modelo.py             # Tests de selección de features, entrenamiento y AUC
│   ├── test_inferencia.py         # Tests de predicción por lotes y guardado
│   ├── test_monitoreo.py          # Tests de emulación de drift, Evidently y chi-cuadrado
│   └── test_api.py                # Tests de FastAPI con mock de MLflow
└── pyproject.toml                 # Dependencias gestionadas con Poetry
```

## Requisitos Previos

1. Python 3.12 o superior.
2. Poetry instalado (`pip install poetry`).
3. Docker Desktop (para levantar los contenedores locales de MLflow y Prefect).

## Guía de Ejecución (PowerShell)

### 1. Instalar dependencias y ejecutar tests (Sin Docker)

La suite de pruebas está desacoplada de servicios externos (usa modelos simulados / *mocks* para MLflow y funciones puras para Prefect):

```powershell
poetry install
poetry run pytest -v
```

### 2. Levantar servicios base (Prefect y MLflow)

Desde la carpeta `services/` levanta los contenedores y luego configura las variables de entorno en la raíz del proyecto:

```powershell
$env:PYTHONPATH="."
$env:PREFECT_API_URL="http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
```

* UI de MLflow: `http://localhost:5000`
* UI de Prefect: `http://localhost:4200`

### 3. Entrenamiento y Promoción del Modelo (`champion`)

Entrena los candidatos configurados en `configs/config.yaml` y promueve al mejor como `champion` en MLflow si supera el umbral mínimo de AUC:

```powershell
poetry run python infraestructure/flows/entrenamiento_flow.py
```

### 4. Inferencia con Drift Emulado y Monitoreo

El nivel de drift emulado se controla mediante el parámetro `inferencia.nivel_drift` en `configs/config.yaml` (valores de prueba: `0.0`, `0.3` y `0.6`).

1. **Puntuar lote de prueba y guardar copia en `lotes/lote_actual.csv`:**
   ```powershell
   poetry run python infraestructure/flows/inferencia_flow.py
   ```

2. **Detectar drift en variables de entrada (Evidently -> Reporte en `reports/`):**
   ```powershell
   poetry run python infraestructure/flows/drift_features_flow.py
   ```

3. **Detectar drift en la variable objetivo `Fraude` (Test Chi-cuadrado):**
   ```powershell
   poetry run python infraestructure/flows/drift_evento_flow.py
   ```

### 5. Despliegue de Pipelines en Prefect

Publica y sirve los 4 despliegues (entrenamiento, inferencia diaria por cron, drift de features y drift de evento):

```powershell
poetry run python infraestructure/flows/desplegar.py
```

### 6. API en Tiempo Real (FastAPI)

Levanta el servidor web consumiendo automáticamente el alias `@champion` registrado en MLflow:

```powershell
poetry run uvicorn infraestructure.api.main:app --port 8000 --reload
```

Documentación interactiva (Swagger UI): `http://localhost:8000/docs`