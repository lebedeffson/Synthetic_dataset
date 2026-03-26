# Материалы для презентации по генерации synthetic dataset

## 1. Данные о приборе

- В репозитории отсутствуют паспорт прибора, модель облучателя и паспорт гипертермической установки.
- Поэтому в презентации этот раздел нужно формулировать аккуратно: доступны только режимные параметры установки и клеточная выживаемость.
- Наблюдаемые параметры эксперимента: `Радиация 0-8 Gy`, `Температура 42/43/44 C`, `Время 30/45 мин`, производная thermal dose `CEM43`.

## 2. Выборка и биология данных

- Это **cell-level / in vitro** survival matrix, а не пациентский датасет.
- Дизайн эксперимента: `5 уровней RT x 3 терморежима = 15 design points`.
- Три терморежима: `42C 45 мин`, `43C 45 мин`, `44C 30 мин`.
- Биологический выход: доля выживших клеток после комбинированного воздействия гипертермии и радиации.
- Главные биологические ожидания: при росте RT выживаемость не должна расти; усиление thermal condition в наблюдаемом окне тоже не должно повышать выживаемость.

## 3. Как собирали правила

- Правила брали из уже собранной knowledge base проекта, а не придумывали вручную под benchmark.
- Активировали только те правила, которые можно честно применить к 4 наблюдаемым колонкам: `Радиация / Температура / Время / Выживаемость`.
- Клинические и in vivo правила сознательно исключались, чтобы не смешивать уровни биологии.
- Активные правила: `CL1-CL5`.

## 4. Откуда брали литературу

- Полная таблица: `tables/literature_table_ru.csv`.
- Ключевые статьи: `E03-E06` для DNA repair / sensitizing window, `E01/E14` для температурного порога, `E06/E07/E08` для интервалов и ограничений обобщения.

## 5. Общая таблица правил

- Полная таблица: `tables/rules_table_ru.csv`.
- `CL1`: monotonicity по радиации.
- `CL2`: sensitizing window около `41-43 C` и `30-60 мин`.
- `CL3`: thermal ordering в наблюдаемом домене.
- `CL4`: при температуре выше `43 C` усиливается direct cytotoxicity.
- `CL5`: высокая комбинированная доза должна вести к very low survival.

## 6. Как получали synthetic выборку

- Базовый support всегда фиксирован на исходных 15 design points.
- Наш основной метод `Matrix` строит rule-guided 5x3 поверхность лог-выживаемости, затем семплирует шум, применяет isotonic projection и calibration.
- Новая версия `Residual VAE` моделирует только остаточную вариативность вокруг matrix prior и получает штраф за нарушения правил уже на этапе обучения.
- Нейросетевые методы `VAE`, `GAN`, `Diffusion` обучались на rule-guided teacher blocks, потому что реальных наблюдений всего `n=15`.
- Любой метод после raw generation проходил одинаковый post-processing: projection, cap и calibration.

## 7. Архитектуры

- Таблица архитектур: `tables/architecture_table_ru.csv`.
- `Matrix`: rule-guided stochastic matrix.
- `Residual VAE`: гибридный residual generator поверх matrix prior, лучший кандидат на следующую финальную версию.
- `Diffusion`: DDPM-style MLP denoiser, лучший чистый neural candidate по full multi-seed fidelity.
- `VAE`: сильный альтернативный neural baseline с очень близкими метриками и чуть меньшим pressure.
- `GAN`: adversarial baseline, но худший по надежности.

## 8. Метрики и итог

- Single-run таблица: `tables/metrics_single_run_ru.csv`.
- Multi-seed таблица: `tables/metrics_multiseed_ru.csv`.
- Scorecard: `tables/method_scorecard_ru.csv`.
- Главный научно защищаемый вывод: **наш основной production-метод = Matrix**, потому что он первичный, прозрачный и напрямую опирается на правила и литературу.
- Главный вывод по следующей версии: **лучший гибридный метод = Residual VAE**.
- Среди чистых student-neural методов: **лучший neural method = Diffusion** по полному multi-seed benchmark.
- Самый низкий post-processing burden среди baseline-family теперь тоже показывает **Residual VAE**.
- `GAN` не брать как основной: есть провалы по monotonicity/compliance на части seed.

## 9. Что вставлять в слайды

- `figures/01_pipeline_overview.png` — схема пайплайна.
- `figures/02_fidelity_comparison.png` — качество на основном прогоне.
- `figures/03_explainability_and_compliance.png` — explainability и compliance.
- `figures/04_multiseed_stability.png` — устойчивость по 5 seed.
- `figures/05_method_positioning.png` — positioning chart.
- `figures/06_design_point_error_heatmaps.png` — ошибки по 15 design points.
- `figures/07_monotonicity_by_seed.png` — why GAN is risky.
- `figures/08_method_scorecard.png` — normalized multi-metric scorecard.
- `figures/09_executive_summary.png` — executive summary slide.
- `figures/10_rule_usage.png` — coverage and role of CL1-CL5.
- `figures/11_benchmark_podium.png` — three main presentation positions.

## 10. Итоговая рекомендация

- Для презентации говорить так: `Matrix` — основной и объяснимый метод, `Residual VAE` — лучшая следующая гибридная версия, `Diffusion` — лучший чистый нейросетевой вариант, `VAE` — сильная альтернативная neural baseline, `GAN` — нестабилен.
- Готовый текст для слайдов: `text/slides_outline_ru.md`.
- Короткие notes для выступления: `text/speaker_notes_ru.md`.
- Математика и формулы: `text/mathematical_foundation_ru.md`.
