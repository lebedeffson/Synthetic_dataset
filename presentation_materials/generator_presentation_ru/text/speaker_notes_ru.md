# Короткий текст для выступления

Основной метод у нас Matrix, потому что он самый прозрачный и научно защищаемый.
Следующая лучшая версия системы это Residual VAE: он сохраняет matrix prior, но резко снижает explainability pressure.
Среди чистых neural baseline лучшим в полном multi-seed benchmark оказался Diffusion.
GAN мы не берем, потому что у него хуже надежность по правилам.

## Цифры, которые можно озвучивать
- Matrix: Local MAE 0.0016, Pressure 0.0344
- Residual VAE: Local MAE 0.0010, Pressure 0.0027
- Diffusion: Local MAE 0.0008, TSTR R2 0.9999

## Финальная формулировка
Мы не заменяем нашу rule-guided модель нейросетью. Мы строим вокруг нее более сильную систему: Matrix как core и Residual VAE как upgrade.

## Scorecard top-3
- Diffusion: overall score 0.98
- VAE: overall score 0.85
- Residual VAE: overall score 0.60