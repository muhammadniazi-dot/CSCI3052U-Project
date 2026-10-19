# Graph Structure as Predictive Signal in Bitcoin Money-Laundering Detection

**Team 11: LiveLaughLoveMachines (LLLM)**
**Course:** CSCI 3052U: Machine Learning I, Fall 2026

## Project Description
This project investigates whether the Bitcoin transaction graph structure provides predictive signal beyond node-level features for detecting illicit (money-laundering) transactions. We compare a naive baseline, three feature-only models (Logistic Regression, Random Forest, Gradient Boosting), and two Graph Neural Networks (GCN, GAT) on the Elliptic Bitcoin Dataset.

## Team Members
- Maria Jose Alarcon Osuna
- Muhammad Niazi
- Vincenzo Langone

## Dataset
Elliptic Bitcoin Dataset (CC BY-NC-ND 4.0)
Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
See `data/README.md` for download instructions.

## Repository Structure
- `notebooks/` — Jupyter notebooks for EDA and experiments
- `scripts/` — Standalone Python scripts (preprocessing, utilities)
- `figures/` — Generated visualizations (PNG)
- `data/` — Dataset download instructions (raw data not committed)
- `report/` — Milestone PDF reports

## How to Reproduce
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset per `data/README.md`
4. Run `notebooks/01_EDA.ipynb` to regenerate the Milestone 2 figures
