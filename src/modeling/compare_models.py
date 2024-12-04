import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

def compare_and_save_best_model(models, X_train, X_test, y_train, y_test, save_dir="models/"):
    results = {}
    best_model = None
    best_accuracy = 0
    best_model_name = None

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        elapsed_time = time.time() - start_time

        results[model_name] = {"accuracy": accuracy, "time": elapsed_time}
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
            best_model_name = model_name

    if best_model:
        joblib.dump(best_model, os.path.join(save_dir, f"{best_model_name}.pkl"))
        print(f"Best model '{best_model_name}' saved with accuracy {best_accuracy:.4f}")
    return results

if __name__ == "__main__":
    PROCESSED_DIR = "./data/processed/"
    os.makedirs("models", exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "clustered_data.csv")
    data = pd.read_csv(data_path)

    features = ['player_dist_total', 'player_dmg', 'cluster', 'drive_type']
    target = 'team_placement'

    X = data[features]
    y = (data[target] <= 5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print("Comparing models...")
    results = compare_and_save_best_model(models, X_train, X_test, y_train, y_test)
    print("Model comparison completed.")
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Time={metrics['time']:.2f}s")
