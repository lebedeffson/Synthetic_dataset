# Слайды для презентации

## Слайд 1. Постановка задачи
- У нас только 15 реальных design points в cell-level domain.
- Нужен explainable synthetic generator, который уважает биологические правила.

## Слайд 2. Данные и правила
- Дизайн: 5 уровней RT x 3 терморежима.
- Активные правила: CL1-CL5.
- Hard rules: CL1, CL3, CL5. Soft rules: CL2, CL4.

## Слайд 3. Наш базовый метод
- Matrix: rule-guided stochastic 5x3 generator.
- Использует projection, cap, calibration и explainability logging.

## Слайд 4. Что сравнивали
- Matrix, Residual VAE, Diffusion, VAE, GAN.
- Все neural methods тестировались в одном rule-guided контуре.

## Слайд 5. Ключевой benchmark
- Показываем full multi-seed table и scorecard.
- Отдельно выделяем fidelity и explainability burden.

## Слайд 6. Главный вывод
- Matrix = основной production / защита.
- Residual VAE = лучшая следующая гибридная версия.
- Diffusion = лучший чистый neural baseline.

## Слайд 7. Почему не GAN
- Есть провалы monotonicity/compliance на части seed.
- Надежность ниже остальных.

## Слайд 8. Что продаем
- Explainable rule-guided synthetic generator for small constrained biomedical data.
- С roadmap: Matrix core -> Residual VAE upgrade.
