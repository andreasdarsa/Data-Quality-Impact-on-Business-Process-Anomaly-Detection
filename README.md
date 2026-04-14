# Experimental Design – Data Quality Impact on Business Process Anomaly Detection

## 1. Γενική Στρατηγική

- Χρησιμοποιούμε **ένα βασικό dataset (ίδια βάση δεδομένων)**.
- Δημιουργούμε **2 παραλλαγές (scenarios)**:
  1. Clean / Baseline
  2. Low-quality (Noise)

Σε κάθε scenario εφαρμόζονται οι ίδιες μέθοδοι ανίχνευσης ώστε να είναι συγκρίσιμα τα αποτελέσματα.

---

# 2. Dataset Scenarios

## 2.1 Baseline Dataset (Clean)

**Χαρακτηριστικά:**
- Καλή ποιότητα δεδομένων
- Περιλαμβάνει anomalies (injected στο αρχικό dataset)
- Χωρίς noise

**Σκοπός:**
- Μέτρηση βασικής απόδοσης (reference performance)
- Δημιουργία baseline metrics

**Μετρήσεις:**
- Precision
- Recall
- F1-score
- ROC-AUC / PR-AUC
- Confusion Matrix

---

## 2.2 Low-Quality Dataset (Noise)

**Χαρακτηριστικά:**
- Ίδιο dataset
- Εισαγωγή προβλημάτων ποιότητας:
  - Missing values
  - Duplicate events
  - Timestamp inconsistencies
  - Random event corruption

**Σκοπός:**
- Μέτρηση επίδρασης ποιότητας δεδομένων στην ανίχνευση

**Σύγκριση:**
- Metrics συγκρίνονται με Baseline
- Ανάλυση αύξησης False Positives / False Negatives

---

# 3. Μέθοδοι Ανίχνευσης (Applied in All Scenarios)
Αρχικά, χρησιμοποιώντας τον **Inductive Miner**, προσπαθούμε:
- Να εξορύξουμε το αντίστοιχο business process model
- Να κατασκευάσουμε το **Directly Follows Graph**
- Να εξάγουμε χαρακτηριστικά για κάθε case από το Process Model
που θα χρησιμοποιηθούν σαν είσοδος στους αλγορίθμους
που θα συγκρίνουμε

Θα εφαρμοστούν διαφορετικές κατηγορίες μεθόδων:

1. LOF (Local Outlier Factor)
2. Isolation Forest
3. DBSCAN (clustering-based)

Όλες οι μέθοδοι εφαρμόζονται σε:
- Baseline (κανονικό dataset χωρίς προβλήματα δεδομένων)
- Degraded (διαφορετικές εκδοχές του ίδιου dataset με κάποιας μορφής αλλοίωση ποιότητας)

---

# 4. Research Questions

RQ1: Πώς αποδίδουν οι μέθοδοι σε καθαρές συνθήκες;  
RQ2: Πώς επηρεάζεται η απόδοση από χαμηλή ποιότητα δεδομένων;  
RQ3: Πώς επηρεάζεται η απόδοση από drift;  
RQ4 (optional): Ποια μέθοδος είναι πιο robust σε συνδυασμό noise + drift;

---

# 5. Πειραματική Λογική

Baseline → Reference Performance  
Noise → Impact of Data Quality

---

# 6. Αναμενόμενη Ανάλυση

- Σύγκριση μεθόδων ανά scenario
- Ranking μεθόδων ανά robustness
- Ανάλυση FP/FN patterns
- Discussion για trade-offs μεταξύ accuracy και robustness
