# Математическая часть проекта

## 1. Matrix generator

Работаем в пространстве `log10(survival + eps)`.

Наблюдаемая матрица:
`Y* in R^(5x3)`

Где строки соответствуют дозам RT, а столбцы — трем observed thermal conditions.

Усиленная матричная модель строится как:
`Y_raw = Mu_prior + E`

где:
- `Mu_prior` — сглаженная monotone prior-surface, построенная из observed matrix;
- `E` — structured block noise;
- затем применяется safety layer: `projection + cap + calibration`.

Structured noise:
`E_ij = Sigma_ij * T * (a_g * z_g + a_r * z_row_i + a_c * z_col_j + a_l * z_local_ij)`

где `Sigma_ij` — heteroscedastic shrinkage sigma, а шум усечен по `truncation_z`.

## 2. Hard и soft rules

Hard rules:
- `CL1`: survival non-increasing по radiation dose
- `CL3`: thermal ordering в observed domain
- `CL5`: при high combined dose survival должен быть very low

Soft rules:
- `CL2`: sensitizing window 41-43 C / 30-60 min
- `CL4`: high temperature direct cytotoxicity

Hard rules встраиваются в projection/cap. Soft rules используются как mechanistic priors и интерпретация.

## 3. Residual VAE

Гибридная модель не генерирует блок с нуля. Она учит только остаток вокруг matrix prior:

`R = Y - Mu_prior`

Encoder/decoder:
- `q_phi(z | R, F)`
- `p_theta(R | z, F)`

где `F` — rule-aware feature tensor (`RT`, `temperature`, `time`, `CEM43`, `thermal_rank`, `CL2/CL4/CL5 indicators`).

Итоговая генерация:
`Y_raw = Mu_prior + R_hat(z, F)`
`Y_final = SafetyPostprocess(Y_raw)`

## 4. Loss function for Residual VAE

`L = L_recon + beta * L_KL + lambda_rule * L_rule + lambda_center * L_center + lambda_var * L_var + lambda_smooth * L_smooth`

Где:
- `L_recon`: reconstruction loss в log-survival space
- `L_KL`: regularization of latent space
- `L_rule`: penalty for raw rule violations before projection
- `L_center`: alignment to teacher-block mean / target center
- `L_var`: variance matching
- `L_smooth`: residual budget regularization

## 5. Метрики

- `Local MAE`: средняя абсолютная ошибка по 15 design points
- `Local Max Error`: максимальная ошибка среди 15 design points
- `Wasserstein(norm)`: близость распределений
- `TSTR R2`: utility, train on synthetic test on real
- `Explainability pressure`: средняя величина пост-коррекции после projection/cap/calibration

## 6. Интерпретация результатов

Если метод хорош только по fidelity, но требует большого rule-correction burden, он не подходит как explainable production method.
Поэтому в проекте мы разделяем:
- `Matrix` как scientific core
- `Residual VAE` как next-version hybrid upgrade
- `Diffusion` как strongest pure neural baseline
