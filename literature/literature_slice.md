# Literature Slice: Combined Temperature and Radiation Effects on Cancer Cells

## Search Scope

- Topic: locoregional hyperthermia / mild hyperthermia combined with ionizing radiation.
- Evidence units: cell lines, organoids, xenografts, animal models, retrospective clinical cohorts, randomized trials, and reviews.
- Main variables: temperature, heating duration, thermal dose, interval between hyperthermia and radiotherapy, sequence, tumor oxygenation, DNA repair, apoptosis, response, and toxicity.

## Evidence Table

| ID | Year | Evidence type | Model / context | Conditions | Main finding | Link |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | 1982 | Animal experiment | Malignant tumors, localized microwave HT | `40 C` vs `43 C`; local heating | Around `40 C` tumor oxygenation improved, while higher temperatures reduced oxygenation because of vascular restriction. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/7146320/) |
| E02 | 2001 | Review | Rodent, canine, and human tumors | Mild HT `39-42 C` | Mild HT can improve oxygenation during and up to `1-2` days after heating, helping radiation response beyond simple reoxygenation. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/11260653/) |
| E03 | 2015 | Mechanistic review | Cancer DNA repair literature | HT across repair pathways | Hyperthermia perturbs multiple DNA repair pathways, not only one target, supporting multi-mechanism radiosensitization. | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4554295/) |
| E04 | 2011 | In vitro mechanistic study | Cancer cells | Mild HT with PARP context | Mild HT inhibits homologous recombination, promotes BRCA2 degradation, and increases treatment sensitivity. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/21555554/) |
| E05 | 2017 | In vitro mechanistic study | Cancer cell models | Thermal dose variation for HR inhibition | HR inhibition and radiosensitization depend on thermal dose; `41-43 C` for about `30-60 min` is a key sensitizing range. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/28574821/) |
| E06 | 2020 | In vitro study | Six cervical cancer cell lines | `37-42 C`, interval `0-4 h`, both sequences, RT `2-8 Gy` | Shorter interval produced more residual DNA damage, apoptosis, and cell kill; sequence had little effect on survival. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32138173/) |
| E07 | 2024 | Multi-model translational study | Cell lines, organoids, xenografts, and 58 women with cervical cancer | HT-RT interval analysis | Radiosensitization critically depended on interval; shorter intervals showed stronger effect and were linked with better survival. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37820768/) |
| E08 | 2019 | Retrospective clinical cohort | 400 locally advanced cervical cancer patients | Interval roughly `30-230 min` in routine practice | Interval alone was not the dominant predictor once thermal dose and treatment quality were considered, showing that interval rules are context-dependent. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/30906734/) |
| E09 | 2005 | Randomized clinical trial | Superficial tumors `<= 3 cm` | RT plus HT with thermal dose targeting by `CEM43 T90` | Adequate thermal dose improved complete response; superficial lesions responded better when sufficient heating was achieved. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/15860867/) |
| E10 | 2019 | In vitro pilot study | HepG2 hepatocellular carcinoma cells | `37/40/43 C` plus `2/4/8 Gy`, assessed at `24/48/72 h` | The strongest cytotoxic pattern was `40 C + 4 Gy + 48 h`, with high apoptosis/necrosis and reduced VEGF/PDGF. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/31450899/) |
| E11 | 2022 | Clinical review | Hyperthermia treatment planning | Thermometric parameters | Temperature, duration, thermal dose, interval, and sequence all matter, but thermal dose is one of the most reproducible predictors. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35158893/) |
| E12 | 2016 | Systematic review and network meta-analysis | Locally advanced cervical cancer | HTRT vs RT; HTCTRT vs RT | HTRT improved complete response and loco-regional control over RT without clear excess of severe toxicity. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/27411568/) |
| E13 | 2025 | Phase II clinical trial | Previously irradiated recurrent head and neck cancer | HT `42 +/- 0.5 C` for `40 min`, within `2 h` after RT, repeated weekly | Combined treatment achieved high response with manageable toxicity, supporting repeated HT-RT schedules even in heavily treated disease. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39920700/) |

## Stable Patterns Extracted from the Literature

1. Mild hyperthermia can improve oxygenation and perfusion.
   The best-supported oxygenation window is roughly `39-42 C`, especially for hypoxic tumors.

2. A sensitizing DNA-repair window exists around `41-43 C`.
   In this range, HT can suppress homologous recombination and increase persistent radiation-induced DNA damage.

3. The interval between HT and RT usually matters.
   Very short intervals are often associated with stronger radiosensitization, more apoptosis, and more residual DNA damage.

4. Sequence is usually less important than interval.
   Several cervical-cell experiments found similar outcomes whether HT was given before or after RT, provided the interval stayed short.

5. Thermal dose is a major higher-level predictor.
   Clinical outcome often tracks achieved heating quality (`CEM43 T90`, related thermometric parameters, session quality) more reliably than nominal protocol labels.

6. Temperatures above `43 C` shift the mechanism.
   Direct cytotoxicity rises, but perfusion support may worsen and normal-tissue toxicity risk becomes more relevant.

7. Superficial and adequately heated lesions are easier to control.
   Depth, heating coverage, and achievable thermal dose remain practical determinants of response.

8. Repeated HT-RT schedules can be beneficial.
   Several clinical settings used repeated sessions with acceptable toxicity and better tumor response than RT alone.

## Contradictions That Must Remain Explicit in the KB

- Interval contradiction:
  E06 and E07 support a strong benefit from very short intervals.
  E08 shows that, in a large cervical cohort, interval alone was not independently dominant when thermal dose and treatment quality were considered.

- Temperature contradiction:
  Mild HT improves oxygenation, but stronger HT can damage perfusion.
  Therefore, "higher temperature is always better" is a false rule.

- Generalization contradiction:
  Some rules are robust across models, but some patterns remain tumor-type or protocol specific.
  The HCC `40 C + 4 Gy + 48 h` pattern from E10 should be treated as a local rule, not a universal one.

## Modeling Consequence

- Use fuzzy rules with confidence weights for the primary knowledge layer.
- Keep crisp alternatives for symbolic engines and rule auditing.
- Preserve conflicts and exceptions as explicit metadata rather than deleting them.
