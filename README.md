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
I used 5 Machine Learning Models to create the Price forecasting model, aiming to improve its accuracy. The main differences across these models were focused on adding/removing variables and testing for overfitting. Below is a summary of the models tested, their differences, performance and key learnings.

![image](https://github.com/user-attachments/assets/b67b8232-1917-4bd1-b2be-2d5311c2fe84)


### Marc  

Subfolder marc  
  
Cleaning in - clean.py,  
EDA in Jupyter notebook - eda.ipynb,  
ML model and pipelines in - model_pipeline.ipynb  

Models Explored :   
Linear Regression -
Linear Regression Log Tranformed -
Ridge Regression on Log-Transformed -
Random Forest -
Random Forest Top Feat/Clust -
Lasso -
Elasticnet? -
Catboost All -
Catboost Top Feat -
Catboost Optuna Tuned All -
Catboost Optuna Tuned Top Feat -
Catboost Refined Top Feat -
CatBoost Quantile Regression -
Catboost House -
Catboost Apartment -
Catboost per subtype -  

Visualization in folder /figures, outcomes below :  
![image](https://github.com/user-attachments/assets/4095b5a4-540c-43db-81f8-f8e17feb37de)

### Moussa  
models explored :    
random forest   
grandient boosting   
decision  tree   
KNN   
Lasso   
Linear regression   
Ridge

### Kenny  
- Started with zero experience in modeling — completely new to the concept
- Learned the foundations through Multiple Linear Regression (MLR)
- Expanded to 7 different models to compare performance:
-- MLR (Multiple Linear Regression)
-- RandomForest
-- GradientBoosting
-- Ridge
-- XGBoost
-- CatBoost
-- Lasso
- Learned how to optimize model performance by tuning hyperparameters — from tree depth to learning rates
- Built a loop to train and evaluate multiple models efficiently
- Explored model stacking to boost results by combining strengths
- Gained insights into categorical handling, target transformation, and the importance of tailored preprocessing


---

