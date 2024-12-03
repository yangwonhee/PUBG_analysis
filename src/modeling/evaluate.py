from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for model_name, model in models.items():
        # 모델 학습
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)
        results[model_name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion
        }

        # 모델 저장
        joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}.pkl")

    return results
