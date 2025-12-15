import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WildFireDataLoader:
    """
    Класс для разгрузки и организации данынх Wildfire Predition Dataset.

    Выполняет:

    """
    class_mapping: dict[str, int] = {"Fire": 0, "Non_Fire": 1}
    reversed_class_mapping: dict[int, str] = {0: "Fire", 1: "Non_Fire"}

    def __init__(self, base_path: str = "../data/raw_orig"):
        """
        Инициализация загрузчика данных.
        base_path: str -- пусть к папке с исходными данными
        """
        self.base_path: Path = Path(base_path)
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.metadata = {}


    def discover_structure(self) -> dict:
        """
        Определяет структуру датасета.
        Возвращает словарь с инфморацией
        о папках train, test, их подпаках
        и найденных типах данных изображений.
        """
        logger.info(f"Поиск данных в: {self.base_path}")

        structure = {
            "train_exists": False,
            "test_exists": False,
            "train_subfolders": [],
            "test_subfolders": [],
            "image_formats": set()
        }

        train_path: Path = self.base_path / "train"
        test_path: Path = self.base_path / "test"

        if train_path.exists():
            structure["train_exists"] = True
            structure["train_subfolders"] = [
                f.name for f in train_path.iterdir()
                if f.is_dir() and f.name in self.class_mapping
            ]

        if test_path.exists():
            structure["test_exists"] = True
            structure["test_subfolders"] = [
                f.name for f in test_path.iterdir()
                if f.is_dir() and f.name in self.class_mapping
            ]

        all_images = list(self.base_path.rglob("*.jpg")) + \
                     list(self.base_path.rglob("*.png")) + \
                     list(self.base_path.rglob("*.jpeg"))

        for img in all_images[:100]:
            structure["image_formats"].add(img.suffix.lower)

        return structure


    def load_from_subfolder(self, split: str = "train") -> pd.DataFrame:
        """
        Загрузка данных из структуры.
        Возвращает pd.DataFrame с данными
        """
        data: list = []
        split_path: Path = self.base_path / split

        if not split_path.exists():
            logger.warning(f"Папка {split} не найдена в {split_path}")
            return pd.DataFrame()

        for class_name, label in self.class_mapping.items():
            class_path: Path = split_path / class_name

            if not class_path.exists():
                logger.warning(f"Папка класса {class_name} не найдена в {class_path}")

            image_formats: list[str] = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            image_files: list = []
            for format in image_formats:
                image_files.extend(class_path.glob(format))

            logger.info(f"Найдено {len(image_files)} изображений для класса {class_name}")

            for img_path in image_files:
                data.append({
                    "image_path": str(img_path.absolute()),
                    "label": label,
                    "class_name": class_name,
                    "split": split,
                    "filename": img_path.name
                })

        return pd.DataFrame(data)


    def load_all_data(self, cache: bool = True) -> dict[str, pd.DataFrame]:
        """
        Загружает все доступные даныне.
        cache: кэшировать ли загруженные в памяти
        Возвращает dict{"train": DataFrame, "test": DataFrame}
        """
        logger.info("Загрузка данных...")
        structure = self.discover_structure()
        logger.info(f"Структура: {structure}")

        if structure["test_exists"] and structure["test_subfolders"]:
            self.train_df = self.load_from_subfolder("train")
            self.test_df = self.load_from_subfolder("test")
        else:
            logger.warning("Найдена нестандартная структура даннх")
            raise RuntimeError("Implement me?..")

        self.metadata = {
            "train_samples": len(self.train_df) if self.train_df is not None else 0,
            "test_samples": len(self.test_df) if self.test_df is not None else 0,
            "train_classes": self.train_df['label'].value_counts().to_dict() if self.train_df is not None else {},
            "test_classes": self.test_df['label'].value_counts().to_dict() if self.test_df is not None else {},
            "structure": structure
        }

        logger.info(f"Загрузка завершена. Тренировочных -- {self.metadata["train_samples"]}, тестовых -- {self.metadata["test_samples"]}")

        return {
            "train": self.train_df,
            "test": self.test_df,
            "metadata": self.metadata
        }
