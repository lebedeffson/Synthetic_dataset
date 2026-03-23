# Итог проекта

## Что сделано

В проекте построен финальный синтетический датасет по 15 исходным наблюдениям для клеточного эксперимента по комбинированному воздействию радиации и гипертермии:

- `Радиация`
- `Температура`
- `Время`
- `Выживаемость`

Главная идея проекта: генерировать данные не просто статистически похожими на исходные, а согласованными с правилами, извлеченными из научных статей по радиобиологии и гипертермии.

## Финальный результат

Основной итоговый датасет:

- `synthetic_data_cell_level_final/final_synthetic_dataset.csv`

Главные сопровождающие файлы:

- `synthetic_data_cell_level_final/final_dataset_report.md`
- `synthetic_data_cell_level_final/final_analysis_report.md`
- `synthetic_data_cell_level_final/independent_cell_level_validation.json`
- `synthetic_data_cell_level_final/design_point_rule_explanations.md`
- `knowledge_base/cell_level_rule_traceability.md`

## Какой подход выбран

В качестве финального решения оставлен не старый `CVAE`, а `cell-level article-guided design-preserving` генератор.

Почему:

- он сохраняет точный экспериментальный дизайн из 15 наблюдаемых точек;
- не генерирует неподтвержденные режимы вне исходной матрицы;
- использует только cell-level правила, релевантные именно клеточному survival dataset;
- дает более сильную объяснимость и более защищаемую научную интерпретацию.

## Какие правила активны в финальной версии

В генераторе используются только правила клеточного уровня:

1. При фиксированном терморежиме рост дозы радиации не должен увеличивать выживаемость.
2. Окно `41-43 C` и `30-60 мин` рассматривается как sensitizing window для подавления репарации ДНК и радиосенсибилизации.
3. В наблюдаемом дизайне порядок по терморежиму должен быть не мягче:
   `42C_45min >= 43C_45min >= 44C_30min` по выживаемости.
4. При температурах выше примерно `43 C` усиливается вклад прямой тепловой цитотоксичности.
5. При высокой комбинированной дозе (`Radiation >= 6 Gy` и `CEM43 >= 45`) выживаемость должна оставаться очень низкой.

Правила про гипоксию, перфузию, интервалы HT-RT, глубину опухоли, repeated sessions и клинический local control не используются в финальном генераторе, потому что они относятся к in vivo или clinical context, а не к данной 15-точечной клеточной матрице.

## Качество финального датасета

Ключевые метрики:

- `mean_wasserstein_normalized = 0.0050`
- `mean_ks_statistic = 0.0203`
- `tstr_mae = 0.0022`
- `tstr_r2 = 0.9991`
- `support_violation_rate_mean = 0.0000`
- `exact_design_support_rate = 1.0000`
- `local_mean_abs_error = 0.0023`
- `local_max_abs_error = 0.0122`
- `radiation_monotonicity_mean_rate = 1.0000`
- `thermal_monotonicity_mean_rate = 1.0000`

Итог: синтетический датасет очень близок к исходной матрице, не выходит за допустимую область и согласован с активными cell-level правилами.

## Устойчивость результата

Проведен дополнительный анализ устойчивости:

- bootstrap analysis по design points;
- multi-seed robustness analysis по 8 различным seed.

Итог:

- структура набора стабильна;
- монотонность сохраняется во всех прогонах;
- ошибка по design points меняется мало;
- наиболее чувствительной точкой остается `0 Gy, 42 C, 45 min`.

## Что важно честно указать

Ограничения проекта:

- реальных наблюдений всего `15`;
- финальный набор лучше трактовать как `design-preserving synthetic augmentation`, а не как симулятор произвольных новых режимов;
- часть правил является не прямым жестким законом из одной статьи, а проектным ограничением, совместимым со статьями и с исходной матрицей;
- самые трудные точки для точного попадания — верхняя граница выживаемости (`0.53`) и нулевая граница (`0.0`).

## Что является финальной рекомендуемой версией

Для работы, отчетов и дальнейшего ML/AI рекомендуется использовать:

- `synthetic_data_cell_level_final/final_synthetic_dataset.csv`

Для обоснования и защиты рекомендуется ссылаться на:

- `PROJECT_SUMMARY.md`
- `synthetic_data_cell_level_final/final_dataset_report.md`
- `synthetic_data_cell_level_final/final_analysis_report.md`
- `knowledge_base/cell_level_rule_traceability.md`
- `synthetic_data_cell_level_final/design_point_rule_explanations.md`

## Команда для воспроизведения

```bash
python scripts/build_cell_level_final_pipeline.py
python scripts/analyze_cell_level_final_dataset.py
```
