# Bundle everything you need for prediction
joblib.dump((preprocessor, stacked_model), "stacked_model_bundle.pkl")