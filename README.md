# 🏠 Challenge: Real Estate Price Prediction with Regression

### 📌 Description
This repository contains our solution to the **Regression Challenge** for predicting real estate prices in Belgium using machine learning models. The challenge was organized as part of a team consolidation project with a focus on data preprocessing, model development, and performance analysis.

The Best Performing Model was produced with LightGBM + Optuna optimization, reaching a r2 of 0.797 when using the test dataset  


### 🧠 Learning Objectives
- Apply linear regression in a real-world context  
- Preprocess and clean real estate data  
- Handle categorical variables and missing data  
- Tune and evaluate multiple regression models  
- Deliver a professional team presentation

### 🚀 Installation

Best Performing Model - LightGBM + Optuna optimization

1) Files required to run model

    - train_set.csv
    - test_set.csv
    - train_model_lightgbm_optuna.py (Python version 3.13.2)

2) Folder structure required:

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
│             ├──   
│  
└── src/  
    ├── train_model_lightgbm_optuna.py  
  

### ⚙️ Usage

1) To install all required packages, run this command in your terminal:  
pip install -r requirements.txt   
  
Run train_model_lightgbm_optuna.py file  
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

### 🧩 Individual contributions
Each team member contributed specific components while collaborating across Git and reviewing each other's modules through pull requests.  
### Sofia  
  
### Marc  



### Moussa  

### Kenny  

---

