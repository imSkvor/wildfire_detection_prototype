# Wildfire Detection using Vision Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)

Прототип системы обнаружения лесных пожаров на основе предобученных моделей Vision Tranformer (ViT). Проект разработан в рамках учебной практики.

В проекте использовались две модели:
Base: https://huggingface.co/Gurveer05/vit-base-patch16-224-in21k-fire-detection
Fine-tuned: https://huggingface.co/EdBianchi/vit-fire-detection

## Результаты

### Сравнение моделей на тестовых данных

| Модель | Accuracy | Precision | **Recall** | F1-Score |
|--------|----------|-----------|------------|----------|
| `vit-base-patch16-224-in21k-fire-detection` | 0.979 | 0.972 | **0.992** | 0.982 |
| `vit-fire-detection` (fine-tuned) | 0.942 | 0.933 | 0.965 | 0.949 |

На этом датасете базовая модель демонстрирует куда болле высокую полноту (Recall = 0.992), что для задачи обнаружения пожаров критически важно.

## Быстрый старт:

### Установка зависимостей:

```bash
git clone https://github.com/imSkvor/wildfire_detection_prototype.git
cd wildfire-detection
pip install -r requirements.txt
```

### Загрузка данных:

Данные взяты из https://www.kaggle.com/datasets/gurveersinghvirk/wildfire-images?resource=download-directory&select=train

```bash
unzip wildfire-prediction-dataset.zip -d data/raw_orig/
```

### Запуск анализа
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_model_evaluation.ipynb
```

## Архитектура проекта:

```text
wildfire-detection/
- data/                          # Данные (в .gitignore)
  - raw_orig/                    # Исходные изображения
  - processed/                   # Обработанные метаданные
- models/                        # Локальные копии моделей
  - notebooks/                   # Jupyter Notebooks
    - 01_eda.ipynb               # Разведочный анализ данных
    - 02_model_evaluation.ipynb  # Оценка моделей
- outputs/                       # Результаты и графики
  - figures/                     # Визуализации
  - reports/                     # Метрики в JSON/CSV
- src/                           # Исходный код Python
  - data_loader.py               # Загрузка и организация данных
    - model_loader.py            # Загрузка предобученных моделей
    - evaluator.py               # Расчет и сравнение метрик
    - utils.py                   # Вспомогательные функции
- requirements.txt               # Зависимости Python
- README.md                      # Документация
```

## Использование

### Загрузка данных:

```python
from src.data_loader import WildFireDataLoader

# Автоматическое обнаружение структуры данных
loader = WildFireDataLoader('data/raw_org')
data = loader.load_all_data()

train_df = data["train"]  # DataFrame с путями и метками
test_df = data["test"]    # DataFrame с путями и метками
```

### Загрузка моделей:

```python
from src.model_loader import ModelLoader

# Автоматическое определение устройства (CUDA/CPU)
model_loader = ModelLoader()

# Загрузка обеих моделей
models = model_loader.load_all_models()

# Доступ к конкретной модели
base_model, base_processor = models["models"]["base"], models["processors"]["base"]
```

### Оценка моделей:

```python
from src.evaluator import ModelEvaluator

# Инициализация оценщика
evaluator = ModelEvaluator(data_loader, model_loader, batch_size=32)

# Оценка конкретной модели
metrics = evaluator.evaluate_model("base", test_df)

# Сравнение всех моделей
comparison_df = evaluator.compare_models("test")
```

# Метрики
Проект фокусируется на Recall, как критически важной метрике для обнаружения пожаров.

- Recall = 0.992 означает, что модель обнаруживает больше 99% реальных пожаров

# Модели

1. Базовая модель - `Gurveer05/vit-base-patch16-224-in21k-fire-detection`
   - ViT, дообученная на датасете пожаров
   - 85.8 миллионов параметров
   - Размер входного изображения -- 224 x 224
2. Fine-tuned модель - `EdBianchi/vit-fire-detection`
   - Дальнейшее дообучение предыдущей модели
   - Более высокие заявленные метрики (precision & recall > 0.99)
   - Модель обучена разделять на три класса -- пожар, дым, нет пожара, что могло повлиять на точность в этой бинарной классификации

# Замечания
Модели обучены на наземных изображениях, в идеале смотерть бы на спутниковые (их куда проще достать, такую модель несравнимо проще применять на практике), но я не справился найти хороших обученных моделей.
Для реального спутникого мониторинга потребуется дообчение на соотвествующих данынх.

# Планы по улучшению:
1. Интеграция Grad-CAM для анализа интерпритируемости
2. Дообучение на спутниковых данных

# Зависимости

Основные зависимости перечислены в `requirements.txt`
