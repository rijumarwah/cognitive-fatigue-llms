# Cognitive Fatigue in Autoregressive Language Models

This repository contains code accompanying the paper:

**Cognitive Fatigue in Autoregressive Language Models: Formalization and Measurement**  
(Accepted as a Main Track Paper at ICML 2026)

We introduce the Fatigue Index (FI), an online diagnostic that measures long-horizon degradation in language models using prompt-directed attention, embedding drift, and entropy deviation.

## Repository Structure

- `fatigue/` – Core implementation of fatigue signals and FI
- `experiments/empirical_validation/` – Experiments for §7.1 (temporal dynamics, predictive validity, hysteresis)
- `experiments/architectural_stress/` – Experiments for §7.2 (context length, precision, scaling)
- `configs/` – Frozen experiment configurations
- `figures/` – Scripts output figures used in the paper

| Paper Section                  | Repository Location                                  |
|--------------------------------|------------------------------------------------------|
| FI definition                  | `fatigue/`                                           |
| Estimation details             | `fatigue/normalize.py`, `configs/`                   |
| §7.1 Empirical Validation      | `experiments/empirical_validation/`                  |
| §7.2 Architectural Stress      | `experiments/architectural_stress/`                  |
| Figures                        | `figures/main/`, `figures/appendix/`                 |

## Setup

```bash
pip install -r requirements.txt
