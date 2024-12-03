import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results):
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, palette='viridis')
    plt.title('Model Comparison - Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.show()

def plot_confusion_matrix(confusion, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
