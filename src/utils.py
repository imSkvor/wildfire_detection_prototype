import json
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Кастомный кодировщик для сериализации numpy объектов."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)

def save_json_with_numpy(data, filepath):
    """Сохранение данных с numpy объектами в JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

def load_json(filepath):
    """Загрузка JSON файла."""
    with open(filepath, 'r') as f:
        return json.load(f)
