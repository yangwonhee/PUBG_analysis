import time
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

def compare_and_save_best_model(models, X_train, X_test, y_train, y_test, save_dir="models/", w1=1, w2=0.01):
    results = {}
    best_model = None
    best_score = float('-inf')
    best_model_name = None

    for model_name, model in models.items():
        # time 확인
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        # accuracy 확인
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 시간과 정확도를 둘 다 반영하도록 score를 만듦
        score = w1 * accuracy - w2 * elapsed_time

        results[model_name] = {
            "accuracy": accuracy,
            "time": elapsed_time,
            "score": score
        }

        if score > best_score:
            best_model = model
            best_score = score
            best_model_name = model_name

    # 최고 모델 저장
    if best_model:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{best_model_name.replace(' ', '_').lower()}.pkl")
        joblib.dump(best_model, model_path)
        print(f"Best model '{best_model_name}' saved with score {best_score:.4f}, accuracy {results[best_model_name]['accuracy']:.4f}, and time {results[best_model_name]['time']:.2f}s")

    return results


if __name__ == "__main__":
    PROCESSED_DIR = "./data/clustered/"
    os.makedirs("models", exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "ERANGEL_2_clustered_data.csv")
    data = pd.read_csv(data_path)
    features = ['player_dist_total', 'player_dmg', 'cluster_0', 'cluster_1', 'cluster_2', 'drive_type']
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
