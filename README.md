# 🏠 Challenge: Real Estate Price Prediction with Regression

### 📌 Description
This repository contains our solution to the **Regression Challenge** for predicting real estate prices in Belgium using machine learning models. The challenge was organized as part of a team consolidation project with a focus on data preprocessing, model development, and performance analysis.

The Best Performing Model was produced with LightGBM + Optuna optimization, reaching a r2 of 0.797 when using the test dataset

### 🚀 Installation

Best Performing Model - LightGBM + Optuna optimization

1) Libraries required:

from lightgbm import early_stopping, log_evaluation, LGBMRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pickle
import optuna

2) Files required to run model

    - train_set.csv
    - test_set.csv
    - train_model_lightgbm_optuna.py (Python version 3.13.2)

3) Folder structure required:

ImmoEliza/03 challenge-regression
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_set.csv
│       └── test_set.csv
│
├── outputs/
│   ├── models/
│   │   ├── 
│   │
│   └── reports/
│       ├── 
│
└── src/
    ├── train_model_lightgbm_optuna.py


### ⚙️ Usage

1) Run train_model_lightgbm_optuna.py file
        From the terminal type: python train_model_lightgbm_optuna.py

### ⚙️ Outputs

The model will produce 2 output files:

    - model_lightgbm_Optuna.pkl
    - metrics_lightgbm_Optuna.txt => this file will include the following metrics:
                => Train_R2
                => Train_RMSE
                => Train_MAE
                => Test_R2
                => Test_RMSE
                => Test_MAE


### 🧠 Learning Objectives
- Apply linear regression in a real-world context  
- Preprocess and clean real estate data  
- Handle categorical variables and missing data  
- Tune and evaluate multiple regression models  
- Deliver a professional team presentation

---

### 🎯 Challenge Overview

| Aspect              | Details                             |
|---------------------|-------------------------------------|
| Duration            | 3 Days                              |
| Deadline (Code)     | 01/07/2025 – 16:30                  |
| Presentation        | 02/07/2025 – 16:00                  |
| Team Size           | 4 Members                           |
| Project Type        | Consolidation Challenge             |
| Client              | ImmoEliza                           |

---

### 👥 Contributors

| Name       |
|------------|
| Sofia      |
| Marc       |  
| Moussa     |  
| Kenny      |

---

### 🗓️ Timeline

| Date        | Milestone                        |
|-------------|----------------------------------|
| Day 1       | Repo setup, study request        |
| Day 2       | Preprocessing, data split        |
| Day 3       | Evaluation, tuning, documentation|
| Day 4       | Presentation & rehearsal         |

---

### 🧩 Personal Situation
Each team member contributed specific components while collaborating across Git and reviewing each other's modules through pull requests.

---

