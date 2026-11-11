import yaml
from pathlib import Path

def load_parameters(config_path: str = "configs/parameters.yaml") -> dict:
    """Carga el archivo de configuración YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo de configuración no existe en: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Prueba rápida de carga
    cfg = load_parameters()
    print("Configuración cargada correctamente:")
    print("Ruta de datos de entrenamiento:", cfg["paths"]["train_data"])
    print("Modelos configurados:", list(cfg["models"].keys()))