from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from colorama import Fore, Style


def evaluate(y_true, y_pred, name="Model"):
    print(f"\n{name}")
    print(f"{Fore.GREEN}{Style.BRIGHT}True anomaly rate: {sum(y_true) / len(y_true)}{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}{Style.BRIGHT}Predicted anomalies: {sum(y_pred)} out of {len(y_pred)} cases{Style.RESET_ALL}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print(f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    return precision, recall, f1
