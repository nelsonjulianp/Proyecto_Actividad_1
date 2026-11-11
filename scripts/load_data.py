"""Carga el dataset de fraude, valida sus columnas y muestra su tamaño.

Por defecto lee `datos.ruta` de configs/config.yaml. Las rutas se resuelven desde la
raíz del proyecto.

Ejecución (desde la raíz):
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""
from pathlib import Path
import pandas as pd
import yaml

RUTA_CONFIG = "configs/config.yaml"

# Columnas requeridas del dataset de fraude.
# El identificador 'ID' se valida para asegurar la estructura, pero se descarta en el retorno.
COLUMNAS_ESPERADAS = [
    "ID",
    "Num_Tarjetas",
    "Balance_Tj",
    "Total_Transac",
    "Trans_Int",
    "Estado",
    "Genero",
    "Tipo_Tarjeta",
    "Fraude",
]

# Columnas que realmente entran al modelado (sin identificador)
COLUMNAS_MODELO = [c for c in COLUMNAS_ESPERADAS if c != "ID"]


def cargar_datos(ruta=None):
    """Devuelve el DataFrame de fraude validado. Falla si falta alguna columna esperada."""
    if ruta is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            ruta = cfg["datos"]["ruta"]

    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_path.resolve()}")

    # Se utiliza sep=';' acorde a la estructura de train.csv
    df = pd.read_csv(ruta_path, sep=";")
    
    # Si por alguna razón viniera separado por comas, se hace fallback
    if df.shape[1] == 1:
        df = pd.read_csv(ruta_path, sep=",")

    faltan = [col for col in COLUMNAS_ESPERADAS if col not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas obligatorias: {faltan}")

    # Retorna únicamente las variables predictoras y el objetivo 'Fraude'
    return df[COLUMNAS_MODELO].copy()


def main():
    # Prueba rápida de lectura directa
    df = cargar_datos("data/train.csv")
    print("=" * 60)
    print("✓ Carga de datos exitosa")
    print(f"Filas: {len(df):,} | Columnas: {df.shape[1]} | Nulos detectados: {int(df.isna().sum().sum())}")
    print(f"Variables cargadas: {list(df.columns)}")
    print(f"Tasa de fraude observada: {df['Fraude'].mean():.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()