from typing import Optional
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Класс для оценки и сравения производительности моделей
    """

    def __init__(self, data_loader, model_loader, batch_size: int = 16):
        """
        Инициализирует "оценщик"

        Принимает загрузчики данных и моделей, размер одного батча
        """
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.batch_size: int = batch_size
        self.results: dict = {}


    def evaluate_model(self, model_key: str, df: pd.DataFrame) -> dict:
        """
        Оценивает одну модель на данном датасете.
        
        Принимает:
        model_key: str = "base"/"finetuned"
        df: DataFrame с данными

        Возвращает dict с результатами оценки
        """
        logger.info(f"Оценка модели {model_key} на {len(df)} изображениях...")

        model, processor = self.model_loader.models[model_key], self.model_loader.processors[model_key]
        model.eval()

        all_preds: list = []
        all_labels: list = []
        all_probs: list = []

        # проход одним батчом
        for i in tqdm(range(0, len(df), self.batch_size)):
            batch_df: pd.DataFrame = df.iloc[i : i + self.batch_size]
            images: list = []

            for path in batch_df["image_path"]:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {path}: {e}")
                    continue
            
            if not images:
                continue

            inputs = processor(images, return_tensors = "pt")
            inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim = -1)
                if model_key == "finetuned":
                    fire_probs = probs[:, 0] # Fire
                    no_fire_probs = probs[:, 1] + probs[:, 2] # Normal + Smoke
                    probs = torch.stack([fire_probs, no_fire_probs], dim=1)
                    probs = probs / probs.sum(dim=1, keepdim=True)

                preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_df["label"].values[:len(images)])

        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics["model_key"] = model_key
        metrics["samples"] = len(all_labels)

        self.results[model_key] = {
            "metrics": metrics,
            "predictions": all_preds,
            "probabilities": all_probs,
            "true_labels": all_labels
        }

        return metrics


    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """
        Расчёт метрик классификации
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        }

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

        return metrics


    def compare_models(self, split: str = "test") -> pd.DataFrame:
        """
        Сравнение всех загруженных моделей

        split: str = "test"/"train"

        Возвращает сравнительную таблицу в виде DataFrame 
        """

        logger.info(f"Сравнение моделей на {split}...")

        if split == "train":
            df = self.data_loader.train_df
        else:
            df = self.data_loader.test_df

        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных для split: {split}")

        comparison_data: list = []
        for model_key in self.model_loader.models.keys():
            metrics = self.evaluate_model(model_key, df)
            comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)

        if len(comparison_df) > 1:
            base_metrics = comparison_df.iloc[0]
            all_metrics = ["accuracy", "precision", "recall", "f1"]
            for metric in all_metrics:
                comparison_df[f"{metric}_diff"] = comparison_df[metric] - base_metrics[metric]

        return comparison_df


    def plot_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Визуализация сравнения моделей
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        metrics = ["accuracy", "precision", "recall", "f1"]
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(comparison_df["model_key"], comparison_df[metric], color=["blue", "orange"])
            ax.set_title(f"{metric.upper()}")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f"{height:.3f}", ha="center", va="bottom")

        ax = axes[0, 2]
        for i, model_key in enumerate(comparison_df["model_key"]):
            cm = self.results[model_key]["metrics"]["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", ax=axes[1, i], 
                       cmap="Blues", cbar=False,
                       xticklabels=["No Fire", "Fire"],
                       yticklabels=["No Fire", "Fire"])
            axes[1, i].set_title(f"Confusion Matrix: {model_key}")
            axes[1, i].set_xlabel("Predicted")
            axes[1, i].set_ylabel("Actual")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


    def get_misclassified_examples(self, model_key: str, n_examples: int = 10) -> list[dict]:
        """
        Получение примеров неправильной классификации
        """
        if model_key not in self.results:
            raise ValueError(f"Модель {model_key} не оценена")

        df = self.data_loader.test_df.copy()
        preds = self.results[model_key]["predictions"]
        labels = self.results[model_key]["true_labels"]

        misclassified_idx = np.where(np.array(preds) != np.array(labels))[0]
        misclassified: list = []

        for idx in misclassified_idx[:n_examples]:
            misclassified.append({
                "image_path": df.iloc[idx]["image_path"],
                "true_label": labels[idx],
                "predicted_label": preds[idx],
                "true_class": "fire" if labels[idx] == 0 else "no_fire",
                "predicted_class": "fire" if preds[idx] == 0 else "no_fire",
                "probability": self.results[model_key]["probabilities"][idx][preds[idx]]
            })

        return misclassified
