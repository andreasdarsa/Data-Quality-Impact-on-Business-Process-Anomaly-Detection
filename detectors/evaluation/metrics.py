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


def evaluate_per_subtype(y_true, y_pred, subtypes, name="Model"):
    print(f"\n=== {name} per subtype ===")

    y_true = list(y_true)
    y_pred = list(y_pred)
    subtypes = list(subtypes)

    unique_subtypes = sorted(set(subtypes))

    for subtype in unique_subtypes:
        idx = [i for i, s in enumerate(subtypes) if s == subtype]

        if len(idx) == 0:
            continue

        y_t = [y_true[i] for i in idx]
        y_p = [y_pred[i] for i in idx]

        precision = precision_score(y_t, y_p, zero_division=0)
        recall = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)

        print(f"{subtype:30s} | P={precision:.2f} R={recall:.2f} F1={f1:.2f} | n={len(idx)}")
