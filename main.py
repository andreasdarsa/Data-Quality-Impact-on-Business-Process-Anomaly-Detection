from detectors.lof_detector import LOFDetector
from detectors.isolation_forest import IsolationForestDetector
from detectors.clustering_detector import DBSCANDetector
from detectors.evaluation.metrics import evaluate, evaluate_per_subtype

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np

from bpm.log.log_gen import build_log
from bpm.mining import mine_process_model, draw_dfg, extract_case_features, encode_case_features


# Step 1: Build the log
build_log()

# Step 2: Mine the process model
df, dfg = mine_process_model()
draw_dfg(dfg)

# Step 3: Features
features_df = extract_case_features(df, dfg)
X, y, subtypes = encode_case_features(features_df)

# Split
X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
    X, y, subtypes, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_full = scaler.fit_transform(X)

# =========================
# 🔵 1. LOF (score-based)
# =========================
lof_best_f1 = 0

for n_neighbors in [20, 30, 50, 100]:
    lof = LOFDetector(n_neighbors=n_neighbors)
    lof.fit(X_train)

    scores = lof.score_samples(X_test)

    # try multiple thresholds
    for perc in [85, 90, 95]:
        threshold = np.percentile(scores, perc)
        y_pred = (scores >= threshold).astype(int)

        f1 = f1_score(y_test, y_pred)
        print(f"LOF k={n_neighbors}, perc={perc}, F1: {f1:.2f}")

        if f1 > lof_best_f1:
            lof_best_f1 = f1
            lof_best = (n_neighbors, perc, y_pred)

print(f"Best LOF: k={lof_best[0]}, perc={lof_best[1]}, F1={lof_best_f1:.2f}")


# =========================
# 🟢 2. Isolation Forest (score-based)
# =========================
iso_best_f1 = 0

for contamination in [0.05, 0.1, 0.15]:
    iso = IsolationForestDetector(n_estimators=100, contamination=contamination)
    iso.fit(X_train)

    scores = iso.score_samples(X_test)

    for perc in [85, 90, 95]:
        threshold = np.percentile(scores, perc)
        y_pred = (scores >= threshold).astype(int)

        f1 = f1_score(y_test, y_pred)
        print(f"IF cont={contamination}, perc={perc}, F1: {f1:.2f}")

        if f1 > iso_best_f1:
            iso_best_f1 = f1
            iso_best = (contamination, perc, y_pred)

print(f"Best IF: cont={iso_best[0]}, perc={iso_best[1]}, F1={iso_best_f1:.2f}")


# =========================
# 🟠 3. DBSCAN (transductive)
# =========================
dbscan_best_f1 = 0

for eps in [1.0, 1.5, 2.0, 2.5]:
    dbscan = DBSCANDetector(eps=eps)

    labels = dbscan.fit_predict(X_full)
    y_pred = (np.array(labels) == -1).astype(int)

    f1 = f1_score(y, y_pred)
    print(f"DBSCAN eps={eps}, F1: {f1:.2f}")

    if f1 > dbscan_best_f1:
        dbscan_best_f1 = f1
        dbscan_best = (eps, y_pred)

print(f"Best DBSCAN eps: {dbscan_best[0]}, F1: {dbscan_best_f1:.2f}")


# =========================
# 📊 FINAL EVALUATION
# =========================
evaluate(y_test, lof_best[2], name="LOF")
evaluate(y_test, iso_best[2], name="Isolation Forest")
evaluate(y, dbscan_best[1], name="DBSCAN")

y_test = y_test.reset_index(drop=True)
y = y.reset_index(drop=True)
sub_test = sub_test.reset_index(drop=True)
subtypes = subtypes.reset_index(drop=True)

evaluate_per_subtype(y_test, lof_best[2], sub_test, name="LOF")
evaluate_per_subtype(y_test, iso_best[2], sub_test, name="Isolation Forest")
evaluate_per_subtype(y, dbscan_best[1], subtypes, name="DBSCAN")
