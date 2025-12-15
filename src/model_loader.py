import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Класс для загрузки и управления моделями ViT.

    Поддерживает:
    1. vit-base-patch16-224-in21k-fire-detection
    2. vit-fire-detection (EdBianchi, finetuning предыдущей)
    """

    model_configs = {
        "base": {
            "model_name": "Gurveer05/vit-base-patch16-224-in21k-fire-detection",
            "description": "Базовая ViT модель, дообученная на датасете пожаров"

        },
        "finetuned": {
            "model_name": "EdBianchi/vit-fire-detection",
            "description": "Дообученная версия ViT с высокими метриками"
        }
    }

    def __init__(self, device: Optional[str] = None):
        """
        Инициализация загрузчика.
        device: str = "cuda"/"cpu", None для автоопределния.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")

        self.models = {}
        self.processors = {}
        self.model_info = {}

    
    def load_model(self, model_key: str, cache_dir: Optional[str] = None) -> tuple[ViTForImageClassification, ViTImageProcessor]:
        """
        Загрзука выбранной модели и её процессора.
        model_key: str = "base"/"finetuned"
        cache_dir: Optional[str] папка для кеширования моделей

        Возвращает tuple(model, processor)
        """
        if model_key not in self.model_configs:
            raise ValueError(f"Неизвестный ключ модели: {model_key}. Доступны: {list(self.model_configs.keys())}")

        model_config = self.model_configs[model_key]
        model_name = model_config["model_name"]

        logger.info(f"Загрузка модели: {model_name}")

        try:
            processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir = cache_dir
            )
        
            model = ViTForImageClassification.from_pretrained(
                model_name,
                cache_dir = cache_dir,
                ignore_mismatched_sizes = True
            )

            model = model.to(self.device)
            model.eval()

            self.models[model_key] = model
            self.processors[model_key] = processor
            self.model_info[model_key] = {
                "name": model_name,
                "description": model_config["description"],
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "device": str(self.device)
            }

            logger.info(f"Модель {model_key} успешно загружена. Параметров: {self.model_info[model_key]["num_parameters"]:,}")

            return model, processor
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise

    
    def load_all_models(self, cache_dir: Optional[str] = "../models") -> dict:
        """
        Загрузка всех доступных моделей.
        cache_dir: str -- папка для кеширования.

        Возвращает словарь со всеми моделями и процессорами
        """
        logger.info("Загрузка всех моделей...")

        for model_key in self.model_configs:
            try:
                self.load_model(model_key, cache_dir)
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {model_key}: {e}")
        
        logger.info(f"Загружено {len(self.models)} из {len(self.model_configs)}")

        return {
            "models": self.models,
            "processors": self.processors,
            "info": self.model_info
        }

    
    def get_model_info(self, model_key: Optional[str] = None) -> dict:
        """
        Возвращает информацию о загруженной модели.
        Если model_key не указан, то возвращает информацию о всех загруженных.
        """
        if model_key:
            if model_key not in self.model_info:
                raise ValueError(f"Модель {model_key} не загружена")
            return self.model_info[model_key]

        return self.model_info


    def preprocess_batch(self, images, model_key: Optional[str] = "base"):
        """
        Предобработка батча изображений для данной модели

        images: Список путей к изображениям
        model_key: ключ модели

        Возвращает тензоры для модели
        """

        if model_key not in self.processors:
            raise ValueError(f"Процессор для модели {model_key} не загружен")

        return self.processors[model_key](images, return_tensors = "pt")
