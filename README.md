# Proyecto Actividad 1 - MLOps

**Asignatura:** MLOps  
**Estudiante:** Nelson Julián Portillo Fonseca  

---

## 1. Descripción del Proyecto
Estructura inicial e implementación del entorno base para el taller de MLOps, integrando control de versiones con Git, gestión de dependencias con Poetry, orquestación de flujos con Prefect y seguimiento de experimentos con MLflow mediante Docker.

---

## 2. Requisitos Previos
* **Python:** 3.12 a 3.14
* **Poetry:** 2.x
* **Git:** Versión reciente
* **Docker Desktop:** Reciente (para levantar los servicios de MLflow y Prefect)

---

## 3. Clonación y Preparación del Entorno (Para Colaboradores / Reproducción)

1. **Clonar el repositorio:**
   git clone https://github.com/nelsonjulianp/Proyecto_Actividad_1.git
   cd Proyecto_Actividad_1

2. **Instalar el entorno virtual y dependencias desde el lock:**
   poetry install

3. **Verificar que el entorno funcione correctamente:**
   poetry run python -c "import prefect, mlflow, sklearn; print('ok')"

---

## 4. Levantamiento de Servicios (Infraestructura)
Los servicios de orquestación y tracking se ejecutan mediante contenedores Docker.

1. Abre **Docker Desktop**.
2. Dirígete a la carpeta services/ y levanta los contenedores:
   cd services
   docker compose up -d
3. Comprueba que las interfaces respondan en el navegador:
   * **MLflow UI:** http://localhost:5000
   * **Prefect UI:** http://localhost:4200

---

## 5. Ejecución de la Prueba de Validación
Para verificar la comunicación con Prefect y el registro de experimentos en MLflow:

1. Asegúrate de estar en la raíz del proyecto y define las variables de entorno en tu terminal de PowerShell:
   $env:PREFECT_API_URL="http://localhost:4200/api"
   $env:MLFLOW_TRACKING_URI="http://localhost:5000"

2. Ejecuta el script de prueba de flujo:
   poetry run python flows/prueba_flow.py

3. **Verificación de resultados:**
   * En **Prefect** (http://localhost:4200/runs), el flujo prueba aparecerá en estado **Completed** mostrando en los logs la suma de 1 a 5 = 15.
   * En **MLflow** (http://localhost:5000), el experimento prueba-validacion registrará el run con sus respectivos parámetros y métricas.