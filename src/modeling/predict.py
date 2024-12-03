import joblib

def predict(data, model_path="models/decision_tree.pkl"):
    model = joblib.load(model_path)
    predictions = model.predict(data)
    return predictions
