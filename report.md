# Отчет по лабораторной работе №4

## Что реализовано

В работе реализованы без использования DL-фреймворков:

1. Однослойная сеть для бинарной классификации `Linear -> BReLU -> Sigmoid`.
2. Перцептрон с одним скрытым слоем `Linear -> BReLU -> Linear -> Sigmoid`.
3. Функция потерь `Binary Cross-Entropy`.
4. Оптимизатор `SGD` с настраиваемым размером батча, momentum и Nesterov.
5. Модификация `Adam`, в которой объединены momentum/Nesterov и RMSProp-подобное масштабирование градиента.

## Датасеты

Используются два набора данных по постановке:

1. `make_moons(n_samples=400, noise=0.15, random_state=ISU_ID)`
2. `make_classification(n_samples=200, n_features=5, n_redundant=2, n_informative=2, n_clusters_per_class=2, n_classes=2, random_state=ISU_ID)`

Разбиение выполнено в пропорции:

1. train: 60%
2. validation: 20%
3. test: 20%

В коде `ISU_ID` вынесен в [config.py](/Users/vitamija/online-courses/metopt-lab4/src/config.py) и может быть переопределен через переменную окружения `ISU_ID`.

## Структура проекта

1. `src/activations.py` - BReLU и Sigmoid.
2. `src/losses.py` - binary cross-entropy.
3. `src/models.py` - однослойная модель и MLP с одним скрытым слоем.
4. `src/optimizers.py` - SGD и Adam-like.
5. `src/training.py` - mini-batch обучение с early stopping.
6. `src/experiments.py` - подбор гиперпараметров, запуск экспериментов и построение графиков.

## План исследования

1. Для каждого датасета обучаются обе модели.
2. Для каждой модели перебираются два оптимизатора.
3. На validation-части выбирается лучшая конфигурация.
4. После этого считаются метрики на test-части и сохраняются графики обучения.

## Результаты экспериментов

Итоговая сводка сохраняется в `img/summary.csv`. Лучшие конфигурации получились такими.

### Moons

1. Лучшая модель: `hidden_layer_perceptron`
2. Лучший оптимизатор: `SGD`
3. `learning_rate = 0.05`
4. `batch_size = 16`
5. Лучшая эпоха: `55`
6. Test accuracy: `0.975`
7. Test F1: `0.9773`

### Classification5D

1. Лучшая модель: `hidden_layer_perceptron`
2. Лучший оптимизатор: `SGD`
3. `learning_rate = 0.03`
4. `batch_size = 16`
5. Лучшая эпоха: `24`
6. Test accuracy: `0.90`
7. Test F1: `0.90`

## Интерпретация результатов

1. На `make_moons` однослойная модель уступает перцептрону с одним скрытым слоем, потому что граница между классами нелинейна.
2. Перцептрон с одним скрытым слоем заметно лучше использует BReLU и строит более гибкую разделяющую поверхность.
3. На `make_classification` скрытый слой тоже оказался полезен, хотя задача проще, чем `moons`.
4. В подобранных режимах лучший результат показал именно `SGD` с momentum/Nesterov. Adam-like тоже работал, но не дал лучшей validation-метрики.

## Артефакты

После запуска `python3 -m src.experiments` создаются:

1. `img/summary.csv`
2. `img/summary.json`
3. `img/learning_curve_moons.png`
4. `img/learning_curve_classification5d.png`
5. `img/decision_boundary_moons.png`

Ниже приведены основные иллюстрации.

![Moons decision boundary](./img/decision_boundary_moons.png)

![Moons learning curve](./img/learning_curve_moons.png)

![Classification learning curve](./img/learning_curve_classification5d.png)
