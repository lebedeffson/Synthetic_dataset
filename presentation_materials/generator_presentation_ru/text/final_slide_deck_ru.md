# Финальная структура презентации

## Слайд 1. Название и главный тезис

**Заголовок слайда**

Explainable synthetic generator для малого биомедицинского эксперимента

**Что ставим на слайд**

Короткий подзаголовок: `15 design points, hard biological rules, explainable synthetic generation`.

Справа можно поставить [09_executive_summary.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/09_executive_summary.png) или использовать его как визуальный мотив.

Внизу одна фраза: `Основной метод: Matrix. Следующая версия: Residual VAE. Лучший pure neural benchmark: Diffusion.`

**Что говорить**

Мы решаем задачу синтетической генерации не для большого датасета, а для очень маленького и биологически ограниченного эксперимента. Поэтому главная цель проекта была не просто сделать красивую синтетику, а получить synthetic dataset, который сохраняет экспериментальный дизайн, уважает биологические правила и остается объяснимым.

---

## Слайд 2. Данные о приборе и эксперименте

**Заголовок слайда**

Экспериментальный дизайн и доступные данные

**Что ставим на слайд**

Крупно:

`5 уровней RT x 3 терморежима = 15 design points`

Под этим короткий блок текста:

`Радиация: 0, 2, 4, 6, 8 Gy`

`Терморежимы: 42C 45 мин, 43C 45 мин, 44C 30 мин`

`Целевая переменная: survival fraction`

Справа можно поставить [01_pipeline_overview.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/01_pipeline_overview.png), но обрезать так, чтобы визуально остался только блок с данными и правилами.

**Что говорить**

В репозитории нет паспорта прибора и формальной документации по установкам, поэтому этот раздел надо подавать аккуратно. Мы можем уверенно говорить о режимных параметрах эксперимента и о клеточной выживаемости. Это не клинический датасет, а cell-level in vitro survival matrix, и именно это определяет весь класс допустимых выводов.

---

## Слайд 3. Биология данных и зачем здесь правила

**Заголовок слайда**

Какая биология зашита в задаче

**Что ставим на слайд**

Три коротких утверждения крупным текстом:

`При росте RT survival не должен расти`

`Более жесткий thermal condition не должен повышать survival`

`Высокая combined dose должна вести к very low survival`

Внизу подпись: `Это не гипотезы модели, а domain constraints из литературы и observed biology.`

**Что говорить**

В нашей задаче правила появляются не как украшение сверху, а как способ формализовать базовую биологию эксперимента. Если модель нарушает эти условия, она может быть статистически красивой, но биологически незащищаемой. Поэтому генератор здесь должен минимизировать не только ошибку по данным, но и цену последующей rule-based коррекции.

---

## Слайд 4. Как собирали правила и литературу

**Заголовок слайда**

Откуда взялись правила CL1-CL5

**Что ставим на слайд**

Слева таблицу [rules_table_ru.csv](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/tables/rules_table_ru.csv) в сокращенном виде.

Справа или снизу график [10_rule_usage.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/10_rule_usage.png).

Дополнительно короткая подпись:

`Hard rules: CL1, CL3, CL5`

`Soft mechanistic priors: CL2, CL4`

**Что говорить**

Правила мы не придумывали вручную под benchmark. Мы брали их из knowledge base проекта и активировали только те, которые действительно применимы к наблюдаемым переменным. Это важный момент, потому что так мы не смешиваем cell-level эксперимент с клиническими или in vivo выводами. В активном наборе осталось пять правил, из которых три работают как hard constraints, а два как mechanistic priors.

---

## Слайд 5. Почему мы не начали с нейросети

**Заголовок слайда**

Проблема свободной генерации при n = 15

**Что ставим на слайд**

Одна большая схема или формула:

`D = {(x_i, y_i)}_{i=1}^{15}`

`Y* in R^(5x3)`

`Свободный генератор без prior => высокая вероятность переучивания и ложной биологии`

Можно взять фрагменты из [mathematical_foundation_ru.md](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/text/mathematical_foundation_ru.md).

**Что говорить**

Если у нас всего пятнадцать реальных design points, то обучать чистый генератор с нуля математически опасно. Модель очень быстро начинает интерполировать или галлюцинировать структуру там, где у нас нет данных. Поэтому сначала мы построили объяснимый rule-guided core, а уже потом стали проверять нейросетевые семейства как benchmark и как возможную надстройку.

---

## Слайд 6. Первая версия Matrix

**Заголовок слайда**

Наш базовый метод: initial Matrix

**Что ставим на слайд**

Формула:

`Y_raw^(0) = Y* + E`

`Y_final = Cal(Cap(Pi_rule(Y_raw^(0))))`

Рядом короткая таблица с initial metrics из [matrix_evolution_ru.csv](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/tables/matrix_evolution_ru.csv):

`Local MAE = 0.0023`

`Wasserstein = 0.0050`

`TSTR R2 = 0.9991`

`Pressure = 0.0338`

`Projection burden = 0.0134`

**Что говорить**

Первая версия уже была рабочей и design-preserving. Но у нее была важная слабость: слишком большая часть нагрузки приходилась на projection stage. Это значило, что raw blocks еще недостаточно хорошо ложились в biologically valid форму и мы слишком много чинили после генерации.

---

## Слайд 7. Как мы усилили Matrix

**Заголовок слайда**

Улучшенная Matrix: что изменили математически

**Что ставим на слайд**

Слева три пункта:

`1. Smoothed prior: Mu_prior`

`2. Heteroscedastic shrinkage sigma`

`3. Structured block noise вместо независимого шума`

Справа график [12_matrix_evolution.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/12_matrix_evolution.png).

**Что говорить**

Мы усилили Matrix не косметически, а на уровне самой генеративной конструкции. Вместо генерации вокруг observed matrix ввели сглаженный prior, вместо одной общей сигмы сделали cell-specific sigma, а вместо независимого шума добавили block structure. Это снизило ошибку и заметно уменьшило projection burden: модель стала сама генерировать более биологичную surface.

---

## Слайд 8. Какие альтернативы мы проверили

**Заголовок слайда**

Сравнение с топовыми генеративными семействами

**Что ставим на слайд**

Основной визуал: [02_fidelity_comparison.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/02_fidelity_comparison.png)

И рядом или снизу: [08_method_scorecard.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/08_method_scorecard.png)

Короткая подпись:

`Сравнивали Matrix, VAE, Diffusion, GAN и затем Residual VAE`

**Что говорить**

Мы не замкнулись в собственной модели и честно прогнали сильные альтернативы для synthetic generation. В pure neural benchmark лучшим оказался Diffusion, VAE показал очень близкий результат, а GAN оказался менее надежным. Это сравнение было нужно не для того, чтобы отказаться от своей модели, а чтобы понять, в какую сторону ее развивать дальше.

---

## Слайд 9. Почему гибрид именно с VAE

**Заголовок слайда**

Логика перехода к Residual VAE

**Что ставим на слайд**

Формулы:

`R = Y - Mu_prior`

`q_phi(z | R, F)`

`p_theta(R | z, F)`

`Y_raw = Mu_prior + R_hat(z, F)`

`Y_final = SafetyPostprocess(Y_raw)`

И короткий текст:

`VAE выбрали не потому, что он лучший standalone, а потому что он лучший residual-надстройщик над Matrix prior`

**Что говорить**

Хотя лучший pure neural baseline у нас Diffusion, для гибрида мы выбрали VAE. Причина в том, что нам нужен был не самый сильный самостоятельный генератор, а лучшая residual-модель поверх Matrix prior. VAE дает удобное latent space, естественную residual-постановку и хорошо сочетается с rule-aware loss.

---

## Слайд 10. Финальная гибридная архитектура

**Заголовок слайда**

Residual VAE как следующая версия системы

**Что ставим на слайд**

Крупно:

`Matrix prior + Residual VAE + rule-aware loss + safety layer`

Сбоку ключевые числа:

`Local MAE = 0.0010`

`Pressure = 0.0027`

`Raw compliance = 1.0`

`Min final compliance across seeds = 0.9375`

**Что говорить**

Гибрид не заменяет Matrix, а усиливает его. Он учит только ту часть вариативности, которую разумно отдавать нейросети. За счет этого мы получаем гораздо меньший explainability burden и хорошую fidelity. Но важно говорить честно: у гибрида пока остается один хвостовой seed с просадкой final compliance, хотя raw compliance держится идеально.

---

## Слайд 11. Explainability и доверие

**Заголовок слайда**

Почему этот генератор можно защищать

**Что ставим на слайд**

Основной визуал: [03_explainability_and_compliance.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/03_explainability_and_compliance.png)

Дополнительно можно вставить небольшой блок:

`Rule traceability`

`Design-point explanations`

`Block-level correction logs`

`Counterfactual analysis`

`Pressure decomposition`

**Что говорить**

Explainability у нас встроена в сам процесс генерации. Мы знаем, какие правила активировались, какой блок сколько потребовал correction burden, какие design points самые чувствительные и что произойдет, если изменить правила или пороги. Поэтому проект можно защищать не как black-box модель, а как систему с трассируемой логикой решений.

---

## Слайд 12. Итог и что продаем

**Заголовок слайда**

Финальная архитектурная позиция проекта

**Что ставим на слайд**

Лучше всего использовать [11_benchmark_podium.png](/home/lebedeffson/Synthetic_dataset/presentation_materials/generator_presentation_ru/figures/11_benchmark_podium.png)

И под ним одну строку:

`Matrix = core`

`Residual VAE = upgrade`

`Diffusion = best pure neural benchmark`

`GAN = reject`

**Что говорить**

Финальный вывод такой. Основной production-метод проекта — Matrix, потому что он самый объяснимый и научно защищаемый. Лучшая следующая версия системы — Residual VAE, потому что он усиливает Matrix и резко снижает explainability burden. Diffusion нужен как честный сильный benchmark среди современных neural methods. И именно в таком виде эту работу и надо показывать: как explainable rule-guided synthetic generator для малого и биологически ограниченного домена.
