# ДЗ 1 по рекомендательным системам

**Студент:** Михайлов Владислав Вадимович

**Датасет:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), категория Video_Games (McAuley Lab, UCSD).
Брал файлы с Hugging Face ([McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)):
- оценки, 5-core: [`benchmark/5core/rating_only/Video_Games.csv`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/rating_only/Video_Games.csv)
- метаданные товаров: [`raw/meta_categories/meta_Video_Games.jsonl`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl)

Оставил взаимодействия начиная с 2015 года и отфильтровал товары с < 10 отзывами и пользователей с < 5.
Вышло ~346 тыс. взаимодействий, 43.9 тыс. пользователей и 8.2 тыс. товаров. Train — до конца 2020, validation — 2021,
test — 2022 и 2023 до сентября.

## Что где лежит
- `experiments.ipynb` — весь эксперимент: данные, EDA, протокол, подбор гиперпараметров, сравнение, выводы
- `models/item_knn.py` — item-based CF
- `models/ials.py` — iALS
- `models/ease.py` — EASE
- `models/fm.py` — факторизационная машина с метапризнаками
- `models/ials_custom.py` — iALS со своим целевым сигналом (оценка × давность покупки)
- `models/utils.py` — загрузка данных, разбиение, признаки, метрики

## Как запустить
Поставить зависимости из `requirements.txt` (у меня Python 3.12), открыть `experiments.ipynb` в Jupyter или VS Code и
нажать Run All. Первая ячейка сама скачает данные (~480 МБ) в папку `data/`. Всё считается примерно за час на CPU.
