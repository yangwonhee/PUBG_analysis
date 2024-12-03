from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from src.modeling.evaluate import evaluate_models
from src.visualization.plots import plot_model_comparison, plot_confusion_matrix

# Sample data (replace with your processed dataset)
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# Evaluate models
results = evaluate_models(models, X_train, X_test, y_train, y_test)

# Plot model comparison
plot_model_comparison(results)

# Plot confusion matrices for each model
for model_name, metrics in results.items():
    plot_confusion_matrix(metrics["confusion_matrix"], model_name)
