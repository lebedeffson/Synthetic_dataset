# Сравнение 5 генераторов для cell-level synthetic dataset

## Данные о приборе

- В репозитории нет паспорта прибора, производителя или модели облучателя/гипертермической установки, поэтому эти данные нельзя добросовестно восстановить.
- Что точно есть в данных: номинальная доза радиации (`0-8 Gy`), температура (`42/43/44 C`), длительность нагрева (`30/45 мин`) и вычисляемая thermal dose `CEM43`.
- Для презентации это лучше формулировать как: `экспериментальная установка задавала фиксированные режимы RT + HT, а в анализе доступны только режимные параметры и клеточная выживаемость`.

## Выборка и биология данных

- Экспериментальный дизайн: 5 уровней радиации (`0/2/4/6/8 Gy`) x 3 терморежима (`42C 45 мин`, `43C 45 мин`, `44C 30 мин`) = 15 наблюдаемых design points.
- Биологический выход: доля выживших клеток (`Выживаемость`) после комбинированного воздействия ионизирующего излучения и гипертермии.
- В проекте это cell-level / in vitro survival matrix, а не клинический датасет пациентов.
- Производные признаки для интерпретации: `thermal_rank`, `thermal_label`, `CEM43`.
- Наблюдаемая биология: при усилении радиации выживаемость не должна расти; усиление терморежима внутри наблюдаемого окна тоже не должно повышать выживаемость.

## Как собирали правила

- Правила брались не `из головы`, а из уже собранной локальной knowledge base проекта: `knowledge_base/cell_level_rules.json`, `knowledge_base/cell_level_rules_ru.md`, `literature/literature_evidence.csv`.
- В активную cell-level часть вошли только правила, которые можно честно применить к 4 колонкам `Радиация / Температура / Время / Выживаемость`.
- Клинические и in vivo закономерности (перфузия, глубина очага, repeated sessions, patient toxicity) сознательно исключались, чтобы не смешивать уровни биологии.
- Итоговые активные правила: `CL1-CL5`. Они задают монотонность по RT, термический порядок, sensitizing window и very-low-survival regime для high combined dose.

## Откуда брали литературу

| ID | Year | Type | Context | Main finding | Link |
|---|---:|---|---|---|---|
| E01 | 1982 | animal_experiment | malignant tumors | Around 40 C tumor oxygenation improved, while higher temperatures reduced oxygenation because of vascular restriction | https://pubmed.ncbi.nlm.nih.gov/7146320/ |
| E02 | 2001 | review | solid tumors | Mild hyperthermia improves oxygenation during and up to 1-2 days after heating and can enhance radiation response | https://pubmed.ncbi.nlm.nih.gov/11260653/ |
| E03 | 2015 | mechanistic_review | cancer cells | Hyperthermia perturbs multiple DNA repair pathways including HR and NHEJ-related components | https://pmc.ncbi.nlm.nih.gov/articles/PMC4554295/ |
| E04 | 2011 | in_vitro_mechanistic | mixed cancer cells | Mild hyperthermia inhibits homologous recombination and induces BRCA2 degradation | https://pubmed.ncbi.nlm.nih.gov/21555554/ |
| E05 | 2017 | in_vitro_mechanistic | mixed cancer cells | HR inhibition and radiosensitization depend on thermal dose; 41-43 C for about 30-60 min is a key sensitizing range | https://pubmed.ncbi.nlm.nih.gov/28574821/ |
| E06 | 2020 | in_vitro | cervical cancer | Shorter interval caused more residual DNA damage, more apoptosis, and more cell kill; sequence had little effect | https://pubmed.ncbi.nlm.nih.gov/32138173/ |
| E07 | 2024 | translational_multimodel | cervical cancer | Radiosensitization critically depended on the HT-RT interval and shorter intervals aligned with better survival | https://pubmed.ncbi.nlm.nih.gov/37820768/ |
| E08 | 2019 | retrospective_clinical | locally advanced cervical cancer | Interval alone was not the dominant predictor once thermal dose and treatment quality were considered | https://pubmed.ncbi.nlm.nih.gov/30906734/ |
| E09 | 2005 | randomized_trial | superficial tumors <=3 cm | Adequate thermal dose improved complete response; superficial lesions responded better when sufficient heating was achieved | https://pubmed.ncbi.nlm.nih.gov/15860867/ |
| E10 | 2019 | in_vitro_pilot | hepatocellular carcinoma | The strongest cytotoxic pattern was 40 C + 4 Gy + 48 h with high apoptosis and reduced VEGF/PDGF | https://pubmed.ncbi.nlm.nih.gov/31450899/ |
| E11 | 2022 | clinical_review | multiple tumors | Temperature, duration, thermal dose, interval, and sequence all matter, but thermal dose is among the most reproducible predictors | https://pubmed.ncbi.nlm.nih.gov/35158893/ |
| E12 | 2016 | systematic_review | locally advanced cervical cancer | HTRT improved complete response and loco-regional control over RT without clear excess of severe toxicity | https://pubmed.ncbi.nlm.nih.gov/27411568/ |
| E13 | 2025 | phase_ii_clinical_trial | recurrent head and neck cancer after prior RT | Repeated HT within 2 h after RT produced high response with manageable toxicity | https://pubmed.ncbi.nlm.nih.gov/39920700/ |
| E14 | 2002 | review_chapter | tumor cells and tumors | The cytotoxic threshold for hyperthermia was described around 42.5 C and the critical temperature for cell-killing curves was between 42.5 C and 43 C | https://www.ncbi.nlm.nih.gov/books/NBK6245/ |

## Общая таблица правил

| Rule | Тип | IF | THEN | Evidence |
|---|---|---|---|---|
| CL1 | hard_process_rule | thermal_condition fixed; radiation_dose_gy increases | survival_fraction non_increasing | E03; E05; E06 |
| CL2 | soft_mechanistic_rule | temperature_c between [41.0, 43.0]; duration_min between [30.0, 60.0] | dna_repair_inhibition high; radiosensitization high | E03; E04; E05 |
| CL3 | hard_article_guided_rule | radiation_dose_gy fixed | survival_order non_increasing ['42C_45min', '43C_45min', '44C_30min'] | E03; E05 |
| CL4 | soft_to_hard_rule_in_domain | temperature_c greater_equal 43.0 | direct_heat_kill increases | E01; E05; E14 |
| CL5 | hard_plausibility_rule | radiation_dose_gy greater_equal 6.0; cem43 greater_equal 45.0 | survival_fraction very_low | E05; E06 |

## Каким образом получали синтетическую выборку

- Сначала фиксировался исходный 15-точечный экспериментальный support: синтетика не изобретает новые режимы лечения вне наблюдаемого дизайна.
- Затем строилась rule-guided матрица среднего лог-выживания и локальных шумов на базе наблюдаемой survival-матрицы.
- Для нейросетевых семейств (`VAE`, `GAN`, `diffusion`) обучение шло не на произвольных новых условиях, а на rule-guided bootstrap blocks, дистиллированных из исходной матрицы. Это нужно из-за крайне малого объема реальных наблюдений (`n=15`).
- Новая гибридная версия `Residual VAE` не генерирует весь блок с нуля: она моделирует только остаточную вариативность вокруг matrix prior и получает штраф за raw-rule violations уже в обучении.
- После семплирования любой нейросетевой блок проходил одинаковый post-processing: isotonic projection по радиации и термическому порядку, cap для высокой комбинированной дозы и глобальную калибровку назад к реальной матрице.
- Поэтому benchmark сравнивает не только качество генерации, но и то, насколько сильно каждую модель приходится `дотягивать правилами` до биологически объяснимого результата.

## Архитектуры

### Matrix

- Архитектура: Rule-guided stochastic 5x3 matrix with isotonic projection and calibration
- Обучение/получение: No neural training. Direct rule-guided generator on the exact 15-point design support.
- Параметров: 0
- Эпох: 0

### VAE

- Архитектура: MLP VAE, latent=8, hidden=96
- Обучение/получение: Trained on rule-guided bootstrap blocks distilled from the observed 15-point matrix.
- Параметров: 24031
- Эпох: 260

### GAN

- Архитектура: MLP GAN (LSGAN), latent=8, hidden=96
- Обучение/получение: Trained on rule-guided bootstrap blocks distilled from the observed 15-point matrix.
- Параметров: 11631
- Эпох: 420

### Diffusion

- Архитектура: DDPM-style MLP denoiser, steps=64, hidden=96
- Обучение/получение: Trained on rule-guided bootstrap blocks distilled from the observed 15-point matrix.
- Параметров: 15375
- Эпох: 280

### Residual VAE

- Архитектура: Residual rule-aware block VAE, latent=6, hidden=128, matrix prior + rule loss
- Обучение/получение: Learns residual stochasticity around the rule-guided matrix prior; projection/cap/calibration stay as a safety layer.
- Параметров: 70171
- Эпох: 320

## Метрики

- `Final compliance`: независимая article/rule compliance по наблюдаемым колонкам.
- `Local MAE`: средняя ошибка между synthetic mean и реальным значением на каждом design point.
- `TSTR R2`: utility-метрика train-on-synthetic test-on-real.
- `Mean Wasserstein(norm)`: сходство распределений.
- `Explainability pressure`: насколько сильно post-processing правил должен был сдвигать raw samples модели.

| Generator | Final compliance | Local MAE | TSTR R2 | Mean Wasserstein(norm) | Explainability pressure | Raw compliance |
|---|---:|---:|---:|---:|---:|---:|
| VAE | 1.0000 | 0.0007 | 0.9999 | 0.0038 | 0.0042 | 1.0000 |
| Diffusion | 1.0000 | 0.0008 | 0.9999 | 0.0040 | 0.0053 | 1.0000 |
| GAN | 1.0000 | 0.0010 | 0.9998 | 0.0045 | 0.0083 | 0.9583 |
| Residual VAE | 1.0000 | 0.0011 | 0.9998 | 0.0046 | 0.0027 | 1.0000 |
| Matrix | 1.0000 | 0.0016 | 0.9995 | 0.0045 | 0.0348 | 1.0000 |

## Ванино объяснение

- Лучший итоговый баланс по fidelity и article-compliance в этом прогоне показал `VAE`.
- Наиболее объяснимым по минимальному rule-correction pressure оказался `Residual VAE`.
- Среди нейросетевых семейств самым практичным компромиссом вышел `Residual VAE`.
- `Residual VAE` стоит рассматривать как next-version кандидата: он сохраняет matrix prior, но снижает explainability pressure до `0.0027` при `Local MAE = 0.0011`.
- Для этого проекта матричный генератор остается основным кандидатом на демонстрацию, потому что его логика напрямую читается через правила CL1-CL5 и литературу.
- Нейросетевые варианты полезны как sanity-check и как демонстрация, что even modern generative families на сверхмалой биомедицинской матрице все равно нуждаются в жестком rule-guided post-processing.
