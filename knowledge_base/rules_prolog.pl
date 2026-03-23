% Alternative crisp logical representation.
% Predicates:
%   evidence_item/6
%   rule/1
%   antecedent/2
%   consequent/2
%   confidence/2
%   evidence/2
%   exception/2

evidence_item(e01, 1982, animal_experiment, "Oxygenation of malignant tumors after localized microwave hyperthermia", "40 C improved oxygenation; higher temperatures worsened it", "https://pubmed.ncbi.nlm.nih.gov/7146320/").
evidence_item(e02, 2001, review, "Improvement of tumor oxygenation by mild hyperthermia", "39-42 C mild HT can improve oxygenation and radiation response", "https://pubmed.ncbi.nlm.nih.gov/11260653/").
evidence_item(e03, 2015, mechanistic_review, "Effects of hyperthermia on DNA repair pathways: one treatment to inhibit them all", "HT perturbs multiple DNA repair pathways", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4554295/").
evidence_item(e04, 2011, in_vitro_mechanistic, "Mild hyperthermia inhibits homologous recombination, induces BRCA2 degradation, and sensitizes cancer cells to poly (ADP-ribose) polymerase-1 inhibition", "Mild HT suppresses HR and degrades BRCA2", "https://pubmed.ncbi.nlm.nih.gov/21555554/").
evidence_item(e05, 2017, in_vitro_mechanistic, "The effect of thermal dose on hyperthermia-mediated inhibition of DNA repair through homologous recombination", "41-43 C for about 30-60 min is a key sensitizing range", "https://pubmed.ncbi.nlm.nih.gov/28574821/").
evidence_item(e06, 2020, in_vitro, "Radiosensitization by Hyperthermia: The Effects of Temperature, Sequence, and Time Interval in Cervical Cell Lines", "Shorter intervals caused more damage and apoptosis; sequence mattered less", "https://pubmed.ncbi.nlm.nih.gov/32138173/").
evidence_item(e07, 2024, translational_multimodel, "Radiosensitization by Hyperthermia Critically Depends on the Time Interval", "Interval critically influenced radiosensitization", "https://pubmed.ncbi.nlm.nih.gov/37820768/").
evidence_item(e08, 2019, retrospective_clinical, "The Effect of the Time Interval Between Radiation and Hyperthermia on Clinical Outcome in 400 Locally Advanced Cervical Carcinoma Patients", "Thermal dose could dominate interval alone", "https://pubmed.ncbi.nlm.nih.gov/30906734/").
evidence_item(e09, 2005, randomized_trial, "Randomized trial of hyperthermia and radiation for superficial tumors", "Adequate thermal dose improved response in superficial lesions", "https://pubmed.ncbi.nlm.nih.gov/15860867/").
evidence_item(e10, 2019, in_vitro_pilot, "Combined Hyperthermia and Radiation Therapy for Treatment of Hepatocellular Carcinoma", "40 C plus 4 Gy plus 48 h was strongest in HepG2", "https://pubmed.ncbi.nlm.nih.gov/31450899/").
evidence_item(e11, 2022, clinical_review, "Clinical Evidence for Thermometric Parameters to Guide Hyperthermia Treatment", "Thermometric parameters and thermal dose are key predictors", "https://pubmed.ncbi.nlm.nih.gov/35158893/").
evidence_item(e12, 2016, systematic_review, "Hyperthermia and radiotherapy with or without chemotherapy in locally advanced cervical cancer: a systematic review with conventional and network meta-analyses", "HTRT improved response and loco-regional control vs RT", "https://pubmed.ncbi.nlm.nih.gov/27411568/").
evidence_item(e13, 2025, phase_ii_trial, "Phase II clinical trial assessing the addition of hyperthermia to salvage concurrent chemoradiotherapy for unresectable recurrent head and neck cancer in previously irradiated patients", "Repeated HT within 2 h after RT gave high response with manageable toxicity", "https://pubmed.ncbi.nlm.nih.gov/39920700/").

rule(r01).
antecedent(r01, temperature(mild)).
antecedent(r01, duration(medium)).
antecedent(r01, hypoxia(high)).
consequent(r01, oxygenation_gain(high)).
consequent(r01, radiosensitization(medium)).
confidence(r01, 0.84).
evidence(r01, [e01, e02, e11]).

rule(r02).
antecedent(r02, temperature(sensitizing)).
antecedent(r02, duration(medium)).
consequent(r02, dna_repair_inhibition(high)).
consequent(r02, radiosensitization(high)).
confidence(r02, 0.88).
evidence(r02, [e03, e04, e05]).

rule(r03).
antecedent(r03, interval(very_short)).
antecedent(r03, temperature(sensitizing)).
antecedent(r03, radiation_dose(medium_or_high)).
consequent(r03, apoptosis_signal(high)).
consequent(r03, radiosensitization(high)).
confidence(r03, 0.86).
evidence(r03, [e06, e07]).

rule(r04).
antecedent(r04, interval(long)).
antecedent(r04, thermal_dose(low)).
consequent(r04, radiosensitization(low)).
consequent(r04, local_control_likelihood(low)).
confidence(r04, 0.73).
evidence(r04, [e06, e07, e08]).
exception(r04, "Do not apply as an absolute statement when thermal dose is adequate or high; E08 indicates interval alone may be non-dominant in clinical cervical practice.").

rule(r05).
antecedent(r05, sequence(any)).
antecedent(r05, interval(short)).
consequent(r05, sequence_importance(low)).
confidence(r05, 0.72).
evidence(r05, [e06]).
exception(r05, "Sequence may still matter in some specific protocols, but current support is weaker than for interval and thermal dose.").

rule(r06).
antecedent(r06, thermal_dose(high)).
antecedent(r06, tumor_depth(superficial)).
antecedent(r06, session_count(repeated)).
consequent(r06, local_control_likelihood(high)).
confidence(r06, 0.90).
evidence(r06, [e09, e11, e12]).

rule(r07).
antecedent(r07, temperature(cytotoxic_high)).
antecedent(r07, duration(long)).
consequent(r07, direct_cytotoxicity(high)).
consequent(r07, toxicity_risk(high)).
consequent(r07, oxygenation_gain(low)).
confidence(r07, 0.79).
evidence(r07, [e01, e05, e11]).

rule(r08).
antecedent(r08, tumor_type(hepatocellular)).
antecedent(r08, temperature(mild_around_40)).
antecedent(r08, radiation_dose(gy4)).
antecedent(r08, assessment_time(post_48h)).
consequent(r08, apoptosis_signal(high)).
consequent(r08, angiogenesis_marker_drop(high)).
confidence(r08, 0.68).
evidence(r08, [e10]).
exception(r08, "This is a local rule derived from a pilot HepG2 study and should not be generalized to all tumors.").

rule(r09).
antecedent(r09, protocol(repeated_ht_rt)).
antecedent(r09, interval(short_or_within_120min)).
antecedent(r09, temperature(clinically_sensitizing)).
consequent(r09, local_control_likelihood(high)).
consequent(r09, toxicity_risk(medium)).
confidence(r09, 0.82).
evidence(r09, [e12, e13]).

rule(r10).
antecedent(r10, tumor_depth(deep)).
antecedent(r10, thermal_dose(low)).
consequent(r10, local_control_likelihood(low)).
confidence(r10, 0.83).
evidence(r10, [e09, e11]).
