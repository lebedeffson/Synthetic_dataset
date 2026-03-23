from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from generate_rule_guided_cvae import enrich_with_rule_features, make_raw_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = PROJECT_ROOT / "benchmarks" / "reports"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep_cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    return df[keep_cols].copy()


def rule_upper_bound(df: pd.DataFrame) -> np.ndarray:
    return np.clip(1.0 - 0.90 * df["Оценка_правил"].to_numpy(dtype=float), 0.0, 1.0)


def compliance_rate(mask: np.ndarray, ok_mask: np.ndarray) -> Dict[str, float]:
    applicable = int(np.sum(mask))
    if applicable == 0:
        return {"applicable": 0, "compliance_rate": None}
    return {"applicable": applicable, "compliance_rate": float(np.mean(ok_mask[mask]))}


def pairwise_monotonicity(
    df: pd.DataFrame,
    same_mask_fn,
    order_values: np.ndarray,
    survival: np.ndarray,
    min_gap: float,
    tolerance: float = 0.02,
) -> Dict[str, float]:
    n = len(df)
    comparisons = 0
    satisfied = 0
    for i in range(n):
        for j in range(n):
            if order_values[j] - order_values[i] < min_gap:
                continue
            if not same_mask_fn(i, j):
                continue
            comparisons += 1
            if survival[j] <= survival[i] + tolerance:
                satisfied += 1
    if comparisons == 0:
        return {"comparisons": 0, "compliance_rate": None}
    return {"comparisons": comparisons, "compliance_rate": float(satisfied / comparisons)}


def evaluate_article_compliance(df: pd.DataFrame, label: str) -> Dict:
    enriched = enrich_with_rule_features(df, assumed_interval_min=0.0, assumed_hypoxia=0.70)
    temp = enriched["Температура"].to_numpy(dtype=float)
    duration = enriched["Время"].to_numpy(dtype=float)
    radiation = enriched["Радиация"].to_numpy(dtype=float)
    survival = enriched["Выживаемость"].to_numpy(dtype=float)
    cem43 = enriched["CEM43"].to_numpy(dtype=float)

    dna = enriched["Подавление_репарации"].to_numpy(dtype=float)
    oxygen = enriched["Оксигенация_выигрыш"].to_numpy(dtype=float)
    synergy = enriched["Синергия"].to_numpy(dtype=float)
    radiosens = enriched["Радиосенсибилизация"].to_numpy(dtype=float)
    direct_cyto = enriched["Прямая_цитотоксичность"].to_numpy(dtype=float)
    high_temp_risk = enriched["Риск_высокой_температуры"].to_numpy(dtype=float)
    upper_bound = rule_upper_bound(enriched)

    rules = {
        "M1_dna_repair_window": compliance_rate(
            (temp >= 41.0) & (temp <= 43.0) & (duration >= 30.0) & (duration <= 60.0),
            dna >= 0.55,
        ),
        "M2_residual_damage_and_survival": compliance_rate(
            (radiation > 0.0) & (dna >= 0.55),
            survival <= (upper_bound + 0.02),
        ),
        "M3_mild_ht_oxygenation": compliance_rate(
            (temp >= 39.0) & (temp <= 42.0),
            oxygen >= 0.50,
        ),
        "M4_short_interval_radiosensitization": compliance_rate(
            (radiation > 0.0) & (temp >= 41.0),
            radiosens >= 0.55,
        ),
        "M6_combined_synergy_condition": compliance_rate(
            (radiation > 0.0) & (temp >= 41.0) & (temp <= 43.0) & (cem43 >= 8.0) & (cem43 <= 60.0),
            (synergy >= 0.50) & (survival <= (upper_bound + 0.02)),
        ),
        "M9_high_temperature_tradeoff": compliance_rate(
            (temp >= 43.5) & (duration >= 50.0),
            (direct_cyto >= 0.50) & (high_temp_risk >= 0.50),
        ),
    }

    same_temp_time = lambda i, j: abs(temp[i] - temp[j]) <= 0.20 and abs(duration[i] - duration[j]) <= 1.5
    same_rad = lambda i, j: abs(radiation[i] - radiation[j]) <= 0.35

    process_checks = {
        "monotonicity_by_radiation": pairwise_monotonicity(
            enriched,
            same_mask_fn=same_temp_time,
            order_values=radiation,
            survival=survival,
            min_gap=1.0,
            tolerance=0.02,
        ),
        "monotonicity_by_cem43": pairwise_monotonicity(
            enriched,
            same_mask_fn=same_rad,
            order_values=cem43,
            survival=survival,
            min_gap=4.0,
            tolerance=0.02,
        ),
        "support_compliance": {
            "real_ranges": {
                "Радиация": [0.0, 8.0],
                "Температура": [42.0, 44.0],
                "Время": [30.0, 45.0],
                "Выживаемость": [0.0, 0.53],
            },
            "compliance_rate": float(
                np.mean(
                    (radiation >= 0.0)
                    & (radiation <= 8.0)
                    & (temp >= 42.0)
                    & (temp <= 44.0)
                    & (duration >= 30.0)
                    & (duration <= 45.0)
                    & (survival >= 0.0)
                    & (survival <= 0.53)
                )
            ),
        },
    }

    numeric_rule_rates = [
        v["compliance_rate"]
        for v in list(rules.values()) + list(process_checks.values())
        if isinstance(v, dict) and "compliance_rate" in v and v["compliance_rate"] is not None
    ]

    return {
        "dataset": label,
        "n_rows": int(len(df)),
        "article_rule_compliance_mean": float(np.mean(numeric_rule_rates)),
        "article_rules": rules,
        "process_checks": process_checks,
    }


def main() -> None:
    dataset_specs = [
        ("rule_guided_cvae", PROJECT_ROOT / "synthetic_data" / "cvae_synthetic_dataset.csv"),
        ("rule_guided_kernel", PROJECT_ROOT / "synthetic_data_kernel" / "kernel_synthetic_dataset.csv"),
        ("notebook_cvae", PROJECT_ROOT / "synthetic_data_notebook_cvae" / "cvae_synthetic_dataset.csv"),
    ]

    results: List[Dict] = []
    for label, path in dataset_specs:
        if path.exists():
            df = load_dataset(path)
            results.append(evaluate_article_compliance(df, label))

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    json_path = COMPARISON_DIR / "article_rule_compliance_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    lines = ["# Article-Based Rule Compliance Comparison", ""]
    lines.append("| Dataset | Mean Compliance | Support Compliance | Radiation Monotonicity | CEM43 Monotonicity |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in results:
        support = item["process_checks"]["support_compliance"]["compliance_rate"]
        rad_mono = item["process_checks"]["monotonicity_by_radiation"]["compliance_rate"]
        cem_mono = item["process_checks"]["monotonicity_by_cem43"]["compliance_rate"]
        lines.append(
            f"| {item['dataset']} | {item['article_rule_compliance_mean']:.4f} | {support:.4f} | {rad_mono:.4f} | {cem_mono:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Mean Compliance` averages rule-wise and process-wise compliance rates for the rules that can be operationalized from the available columns.",
            "- `Support Compliance` measures staying within the real-data ranges.",
            "- `Radiation Monotonicity` checks that survival does not increase when radiation increases at approximately fixed temperature and time.",
            "- `CEM43 Monotonicity` checks that survival does not increase when thermal dose increases at approximately fixed radiation.",
        ]
    )

    md_path = COMPARISON_DIR / "article_rule_compliance_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
