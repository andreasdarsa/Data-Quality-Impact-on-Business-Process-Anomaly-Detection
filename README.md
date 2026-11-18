# Η επίδραση της ποιότητας δεδομένων στην ανίχνευση ανωμαλιών σε επιχειρησιακές διαδικασίες

## 1. Περιγραφή του έργου

Το repository περιλαμβάνει την πειραματική υλοποίηση της μελέτης της επίδρασης της υποβάθμισης της ποιότητας δεδομένων στην ανίχνευση ανωμαλιών σε επιχειρησιακές διαδικασίες (Business Process Event Logs).

Στόχος είναι να διερευνηθεί κατά πόσο προβλήματα στην ποιότητα των event logs μπορούν να επηρεάσουν την ικανότητα των ανιχνευτών να διακρίνουν τις πραγματικές ανωμαλίες από τις κανονικές εκτελέσεις μιας διαδικασίας.

Η πειραματική προσέγγιση περιλαμβάνει:

- Παραγωγή ή προετοιμασία event logs με κανονικές και ανώμαλες εκτελέσεις.
- Ελεγχόμενη υποβάθμιση της ποιότητας των δεδομένων.
- Κατασκευή Directly-Follows Graph (DFG) για την αναπαράσταση της κανονικής ροής της διαδικασίας.
- Εξαγωγή χαρακτηριστικών από τα event logs.
- Εκπαίδευση και αξιολόγηση διαφορετικών αλγορίθμων ανίχνευσης ανωμαλιών.
- Ανάλυση των επιπτώσεων της υποβάθμισης τόσο συνολικά όσο και ανά τύπο ανωμαλίας.

Το έργο εξετάζει ειδικότερα τις ακόλουθες ερευνητικές κατευθύνσεις:

1. Σε ποιο βαθμό οι μετρικές process discovery αποτυπώνουν ή αποκρύπτουν τις επιπτώσεις της υποβάθμισης της ποιότητας των event logs;
2. Σε ποιο βαθμό οι ανιχνευτές ανωμαλιών συγχέουν τις ατέλειες καταγραφής με ανώμαλες εκτελέσεις διαδικασιών;
3. Ποιες υποβαθμίσεις έχουν τη μεγαλύτερη επίδραση και πώς διαφοροποιείται η ευαισθησία μεταξύ των μεθόδων ανίχνευσης;

Το παρόν repository αφορά το σκέλος της **ανίχνευσης ανωμαλιών**.

---

## 2. Πειραματική μεθοδολογία

Η βασική πειραματική διαδικασία ακολουθεί τα παρακάτω βήματα:

```text
Event Log
    |
    v
Data Validation and Preprocessing
    |
    v
Train / Test Split
    |
    +------------------------------+
    |                              |
    v                              v
Clean Training Set           Clean Test Set
    |                              |
    v                              |
DFG Construction                   |
    |                              |
    v                              |
Feature Extraction                 |
    |                              |
    v                              |
Detector Training                  |
    |                              |
    +------------------------------+
                   |
                   v
          Clean Test Evaluation
                   |
                   v
          Test Data Degradation
                   |
                   v
         Degraded Test Evaluation
                   |
                   v
          Comparative Analysis
```

### 2.1 Training and test sets

Τα δεδομένα διαχωρίζονται σε σύνολα εκπαίδευσης και ελέγχου με stratified train/test split, ώστε να διατηρείται σε μεγάλο βαθμό η αναλογία κανονικών και ανώμαλων περιπτώσεων.

Η εκπαίδευση των ανιχνευτών πραγματοποιείται στο αρχικό, μη υποβαθμισμένο training set, το οποίο περιλαμβάνει τόσο κανονικές όσο και ανώμαλες περιπτώσεις.

### 2.2 Directly-Follows Graph

Το Directly-Follows Graph (DFG) χρησιμοποιείται για την αναπαράσταση της ροής δραστηριοτήτων της επιχειρησιακής διαδικασίας.

Στα πειράματα των πραγματικών event logs, το DFG κατασκευάζεται αποκλειστικά από τις κανονικές περιπτώσεις του training set και στη συνέχεια εφαρμόζεται pruning με ελάχιστη συχνότητα ακμής ίση με 2.

Το ίδιο DFG χρησιμοποιείται για την εξαγωγή χαρακτηριστικών από τα training και test sets.

### 2.3 Model training and evaluation

Οι ανιχνευτές εκπαιδεύονται μία φορά ανά train/test split.

Η υποβάθμιση εφαρμόζεται αποκλειστικά στα test data. Τα καθαρά και τα υποβαθμισμένα δεδομένα αξιολογούνται με τα ίδια ήδη εκπαιδευμένα μοντέλα, τον ίδιο scaler και τα ίδια thresholds.

Με αυτόν τον τρόπο, οι διαφορές στις προβλέψεις μπορούν να αποδοθούν στην υποβάθμιση των δεδομένων και όχι σε επανεκπαίδευση ή επαναπροσδιορισμό των ορίων ανίχνευσης.

---

## 3. Datasets

Το project περιλαμβάνει ένα συνθετικό dataset και, στο branch `feat/real-log-anomaly-detection`, δύο επιπλέον datasets βασισμένα σε πραγματικά event logs.

### 3.1 Synthetic event log

Το συνθετικό dataset παράγεται μέσω του `src/generation/log_gen.py` και περιλαμβάνει:

| Κατηγορία | Αριθμός περιπτώσεων |
|---|---:|
| Normal | 8,000 |
| Structural anomalies | 400 |
| Temporal anomalies | 400 |
| Contextual anomalies | 400 |
| **Total** | **9,200** |

Οι ανωμαλίες εισάγονται μέσω του `src/generation/anomalies.py`.

Οι τρεις βασικές κατηγορίες περιλαμβάνουν διαφορετικούς τύπους αλλοίωσης της εκτέλεσης:

- **Structural:** λανθασμένη σειρά δραστηριοτήτων, ελλείπουσες ή διπλές δραστηριότητες.
- **Temporal:** ασυνήθιστες διάρκειες, αρνητικές χρονικές διαφορές και μεγάλα χρονικά κενά.
- **Contextual:** λανθασμένη ανάθεση πόρων, τιμές εκτός αναμενόμενων ορίων και ασυμβατότητα μεταξύ προτεραιότητας και διάρκειας.

### 3.2 BPIC12

Στο branch `feat/real-log-anomaly-detection` χρησιμοποιείται το αρχείο `bpic12-0.3-1.json`, το οποίο περιέχει event log με ετικέτες ανωμαλιών από το BINet.

| Χαρακτηριστικό | Τιμή |
|---|---:|
| Cases | 13,087 |
| Events | 165,140 |
| Normal cases | 9,115 |
| Anomalous cases | 3,972 |
| Anomaly rate | 30.35% |

Οι ανωμαλίες περιλαμβάνουν τους ακόλουθους τύπους:

- Early
- Insert
- Late
- Rework
- SkipSequence

Το `src/data/adapters/binet_adapter.py` αναλαμβάνει τη μετατροπή του αρχικού JSON σε ενιαία εσωτερική αναπαράσταση event logs. Διατηρούνται αποκλειστικά τα COMPLETE lifecycle transitions, ενώ τα START transitions χρησιμοποιούνται για την εξαγωγή διάρκειας δραστηριοτήτων όπου είναι διαθέσιμα τα αντίστοιχα ζεύγη.

Οι ετικέτες ανωμαλιών προέρχονται από το BINet export και δεν πρέπει να ερμηνεύονται ως ανεξάρτητα επαληθευμένες πραγματικές ανωμαλίες.

### 3.3 BPIC13

Το δεύτερο πραγματικό dataset προέρχεται από το BPIC13 και χρησιμοποιείται μέσω του αρχείου `bpic13-0.3-1.json`.

| Χαρακτηριστικό | Τιμή |
|---|---:|
| Cases | 1,487 |
| Events | 6,891 |
| Normal cases | 1,125 |
| Anomalous cases | 362 |
| Anomaly rate | 24.34% |

Οι διαθέσιμοι τύποι ανωμαλιών είναι:

- Early
- Insert
- Late
- Rework
- SkipSequence

Η μετατροπή των δεδομένων πραγματοποιείται μέσω του `src/data/adapters/bpic13_adapter.py`.

Το συγκεκριμένο event log διαθέτει completion timestamps, χωρίς διαθέσιμα event start timestamps ή πληροφορία ποσού. Για τον λόγο αυτό, τα χαρακτηριστικά που απαιτούν event durations και case amount εξαιρούνται από το BPIC13 feature matrix.

---

## 4. Data Quality Degradation

Η υποβάθμιση προσομοιώνεται σε τέσσερις διαστάσεις ποιότητας δεδομένων:

| Dimension | Περιγραφή | Εφαρμοζόμενος μηχανισμός |
|---|---|---|
| Accuracy | Λανθασμένες χρονικές τιμές | Duration/timestamp corruption |
| Completeness | Ελλείποντα γεγονότα | Random event removal |
| Consistency | Ασυνέπεια στην ανάθεση πόρων | Resource mismatch |
| Timeliness | Καθυστερημένη καταγραφή γεγονότων | Event delay |

Κάθε διάσταση εξετάζεται σε τρία επίπεδα υποβάθμισης:

- 10%
- 20%
- 30%

Η υλοποίηση των μηχανισμών βρίσκεται στο `src/degradation/`.

Η εφαρμογή της υποβάθμισης γίνεται αποκλειστικά στα test logs, διατηρώντας τις αρχικές ετικέτες ανωμαλιών.

---

## 5. Feature Extraction

Για κάθε case εξάγεται ένα σύνολο αριθμητικών χαρακτηριστικών, τα οποία οργανώνονται σε τέσσερις βασικές κατηγορίες.

### 5.1 Basic features

- Number of events
- Number of unique activities
- Total case duration
- Case amount, όπου είναι διαθέσιμο

### 5.2 Temporal features

- Mean event duration
- Standard deviation of event durations
- Minimum and maximum event duration
- Mean inter-event gap
- Standard deviation of inter-event gaps
- Minimum and maximum inter-event gap

### 5.3 Resource-related features

- Number of unique resources
- Unknown resource ratio
- Dominant resource ratio
- Resource diversity

### 5.4 Control-flow features

Τα χαρακτηριστικά αυτά υπολογίζονται με βάση το DFG:

- Presence of loops
- Trace length
- Number of unknown edges
- Ratio of unknown edges
- Number of rare edges
- Mean and minimum edge frequency
- DFG path log-likelihood

Η εξαγωγή χαρακτηριστικών υλοποιείται στο `src/features/`.

---

## 6. Anomaly Detection Algorithms

Το project αξιολογεί τέσσερις αλγορίθμους ανίχνευσης ανωμαλιών.

| Algorithm | Configuration |
|---|---|
| K-Nearest Neighbors (KNN) | 5 nearest neighbors |
| Local Outlier Factor (LOF) | 20 neighbors, novelty detection |
| Isolation Forest | contamination = auto |
| DBSCAN | eps = 0.8, min_samples = 5 |

Πριν από την εκπαίδευση, εφαρμόζεται `StandardScaler`, ο οποίος προσαρμόζεται αποκλειστικά στο training set.

Για τους KNN, LOF και Isolation Forest, το όριο ανίχνευσης υπολογίζεται ως το 90ό εκατοστημόριο των anomaly scores στο training set.

Το DBSCAN ακολουθεί διαφορετική διαδικασία: εκπαιδεύεται στο training set και οι test περιπτώσεις αξιολογούνται με βάση την απόστασή τους από το πλησιέστερο training core sample. Περιπτώσεις με απόσταση μεγαλύτερη από το `eps` ταξινομούνται ως ανώμαλες.

Η υλοποίηση των μοντέλων βρίσκεται στο `src/models/`.

---

## 7. Evaluation Metrics

Η αξιολόγηση βασίζεται στις ακόλουθες μετρικές:

- **Precision:** ποιο ποσοστό των περιπτώσεων που χαρακτηρίστηκαν ως ανώμαλες είναι πράγματι ανώμαλο.
- **Recall:** ποιο ποσοστό των πραγματικών ανωμαλιών εντοπίστηκε.
- **F1-score:** αρμονικός μέσος όρος precision και recall.
- **False Positive Rate (FPR):** ποσοστό των κανονικών περιπτώσεων που ταξινομήθηκαν λανθασμένα ως ανώμαλες.

Επιπλέον, καταγράφονται τα στοιχεία του confusion matrix (TP, TN, FP, FN), ενώ πραγματοποιείται αξιολόγηση ανά subtype ανωμαλίας.

### 7.1 Paired false-positive decomposition

Για την περαιτέρω διερεύνηση της δεύτερης ερευνητικής ερώτησης, το project περιλαμβάνει ξεχωριστή ανάλυση των false positives.

Οι ψευδώς θετικές προβλέψεις διαχωρίζονται σε:

- **FP affected:** false positives μεταξύ των περιπτώσεων που επηρεάστηκαν από την υποβάθμιση.
- **FP unaffected:** false positives μεταξύ των περιπτώσεων που δεν επηρεάστηκαν.
- **FP newly affected:** νέες false-positive προβλέψεις σε κανονικές περιπτώσεις που επηρεάστηκαν από την υποβάθμιση και δεν είχαν χαρακτηριστεί ως false positives στην καθαρή αξιολόγηση.

Η ανάλυση συγκρίνει τις προβλέψεις των ίδιων μοντέλων στα καθαρά και στα υποβαθμισμένα test sets.

Η υλοποίηση βρίσκεται στο `experiments/fp_decomposition.py` και παράγει τόσο αναλυτικά αποτελέσματα όσο και συγκεντρωτικές μετρικές.

---

## 8. Experimental Configuration

### Synthetic experiments

Στο βασικό branch (`main`) χρησιμοποιούνται πέντε seeds:

```python
SEEDS = [42, 43, 44, 45, 46]
```

Το train/test split είναι stratified και χρησιμοποιεί test size 30%.

### Real-log experiments

Στο branch `feat/real-log-anomaly-detection` χρησιμοποιούνται:

```python
SPLIT_SEEDS = [42, 43, 44, 45, 46]

DEGRADATION_SEEDS = [
    101, 102, 103, 104, 105
]

TEST_SIZE = 0.3

DEGRADATION_LEVELS = [
    0.1, 0.2, 0.3
]

THRESHOLD_PERCENTILE = 90
```

Για κάθε dataset, ο πειραματικός σχεδιασμός περιλαμβάνει:

- 5 train/test splits.
- 5 ανεξάρτητες επαναλήψεις υποβάθμισης.
- 4 διαστάσεις ποιότητας.
- 3 επίπεδα υποβάθμισης.
- 4 anomaly detection algorithms.

Συνολικά προκύπτουν:

```text
5 × 5 × 4 × 3 = 300
```

διαμορφώσεις υποβάθμισης ανά dataset.

Κάθε διαμόρφωση αξιολογείται με όλους τους τέσσερις ανιχνευτές.

---

## 9. Repository Structure

Η βασική δομή του project είναι:

```text
.
├── experiments/
│   ├── analysis.ipynb
│   ├── baseline.py
│   ├── configs.py
│   ├── generate_dataset.py
│   ├── make_split.py
│   └── run_degradation.py
│
├── src/
│   ├── data/
│   │   ├── adapters/
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── splitter.py
│   │
│   ├── degradation/
│   │   ├── accuracy.py
│   │   ├── completeness.py
│   │   ├── consistency.py
│   │   ├── timeliness.py
│   │   └── dispatcher.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── per_subtype.py
│   │   └── reporting.py
│   │
│   ├── features/
│   │   ├── basic.py
│   │   ├── temporal.py
│   │   ├── resource.py
│   │   ├── control_flow.py
│   │   └── extractor.py
│   │
│   ├── generation/
│   │   ├── log_gen.py
│   │   └── anomalies.py
│   │
│   ├── models/
│   │   ├── knn.py
│   │   ├── lof.py
│   │   ├── iforest.py
│   │   ├── dbscan.py
│   │   └── runner.py
│   │
│   ├── process_mining/
│   │   └── dfg.py
│   │
│   └── utils/
│       ├── paths.py
│       └── seed.py
│
├── tests/
├── requirements.txt
└── README.md
```

Στο branch `feat/real-log-anomaly-detection` υπάρχουν επιπλέον scripts για την προετοιμασία, την αξιολόγηση και την ανάλυση των BPIC12 και BPIC13 datasets:

```text
experiments/
├── make_binet_splits.py
├── bpic13_split.py
├── binet_baseline.py
├── bpic13_baseline.py
├── bpic13_degradation.py
├── fp_decomposition.py
├── bpic12_prediction_analysis.py
└── bpic13_prediction_analysis.py

src/data/adapters/
├── binet_adapter.py
└── bpic13_adapter.py
```

---

## 10. Installation

Απαιτείται Python και οι βιβλιοθήκες που περιλαμβάνονται στο `requirements.txt`.

Δημιουργία και ενεργοποίηση virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Εγκατάσταση dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. Execution

### 11.1 Synthetic experiments — main branch

Για την παραγωγή των συνθετικών event logs:

```bash
python -m experiments.generate_dataset
```

Για τη δημιουργία των train/test splits:

```bash
python -m experiments.make_split
```

Για την εκτέλεση του baseline experiment:

```bash
python -m experiments.baseline
```

Για την εκτέλεση των degradation experiments:

```bash
python -m experiments.run_degradation
```

### 11.2 BPIC12 experiments — real-log branch

Το αρχικό αρχείο `bpic12-0.3-1.json` πρέπει να τοποθετηθεί στον φάκελο:

```text
data/raw/
```

Δημιουργία των train/test splits:

```bash
python -m experiments.make_binet_splits
```

Εκτέλεση baseline:

```bash
python -m experiments.binet_baseline
```

Εκτέλεση degradation experiments:

```bash
python -m experiments.run_degradation
```

### 11.3 BPIC13 experiments — real-log branch

Το αρχικό αρχείο `bpic13-0.3-1.json` πρέπει να τοποθετηθεί στον φάκελο:

```text
data/raw/
```

Δημιουργία των train/test splits:

```bash
python -m experiments.bpic13_split
```

Εκτέλεση baseline:

```bash
python -m experiments.bpic13_baseline
```

Εκτέλεση degradation experiments:

```bash
python -m experiments.bpic13_degradation
```

### 11.4 False-positive decomposition

Για την ανάλυση των false positives:

```bash
python -m experiments.fp_decomposition --dataset both
```

Εναλλακτικά, μπορεί να επιλεγεί ένα συγκεκριμένο dataset:

```bash
python -m experiments.fp_decomposition --dataset bpic12
```

```bash
python -m experiments.fp_decomposition --dataset bpic13
```

Τα scripts ανάλυσης των BPIC12 και BPIC13 χρησιμοποιούνται για τη συγκέντρωση και την περαιτέρω επεξεργασία των πειραματικών αποτελεσμάτων.

---

## 12. Results and Generated Files

Τα πειραματικά αποτελέσματα αποθηκεύονται τοπικά στους ακόλουθους φακέλους:

```text
results/
├── baseline/
├── degradation/
└── fp_decomposition/
```

Τα υποβαθμισμένα event logs αποθηκεύονται στον φάκελο:

```text
data/degraded/
```

Τα ενδιάμεσα και προετοιμασμένα δεδομένα βρίσκονται στον:

```text
data/processed/
```

Τα αποτελέσματα περιλαμβάνουν:

- Baseline evaluation metrics.
- Degradation evaluation metrics.
- Confusion matrices.
- Per-subtype evaluation metrics.
- False-positive decomposition results.
- Aggregated results for comparative analysis.

Τα raw datasets, τα παραγόμενα logs και τα πειραματικά outputs δεν είναι απαραίτητο να περιλαμβάνονται στο Git repository. Για την αναπαραγωγή των πειραμάτων απαιτούνται τα αντίστοιχα αρχικά δεδομένα και η εκτέλεση των σχετικών scripts.

---

## 13. Limitations

Τα αποτελέσματα πρέπει να ερμηνεύονται λαμβάνοντας υπόψη τους ακόλουθους περιορισμούς:

- Το συνθετικό dataset βασίζεται σε προκαθορισμένα μοτίβα εκτέλεσης και ελεγχόμενους τύπους ανωμαλιών.
- Τα BPIC12 και BPIC13 datasets έχουν διαφορετικά χαρακτηριστικά και διαφορετική προέλευση ετικετών ανωμαλιών.
- Οι ετικέτες του BPIC12 προέρχονται από το BINet export και δεν αποτελούν ανεξάρτητη επιβεβαίωση πραγματικών ανωμαλιών.
- Η διαθεσιμότητα χρονικών και contextual χαρακτηριστικών διαφέρει μεταξύ των datasets.
- Η αξιολόγηση βασίζεται σε συγκεκριμένους αλγορίθμους και παραμέτρους και δεν καλύπτει το σύνολο των διαθέσιμων τεχνικών anomaly detection.
- Η ελεγχόμενη υποβάθμιση προσομοιώνει συγκεκριμένους μηχανισμούς προβλημάτων ποιότητας και δεν αναπαριστά απαραίτητα κάθε πιθανό πρόβλημα που εμφανίζεται σε πραγματικά πληροφοριακά συστήματα.

---

## 14. Future Work

Πιθανές επεκτάσεις της μελέτης περιλαμβάνουν:

- Αξιολόγηση πρόσθετων αλγορίθμων anomaly detection.
- Εξέταση διαφορετικών στρατηγικών επιλογής thresholds.
- Επέκταση σε περισσότερα πραγματικά event logs.
- Διερεύνηση συνδυαστικών σεναρίων υποβάθμισης ποιότητας.
- Ανάπτυξη μηχανισμών ανίχνευσης που λαμβάνουν υπόψη την ποιότητα των εισερχόμενων δεδομένων.
- Μελέτη της συμπεριφοράς των ανιχνευτών σε online event streams.

---

## License and Data

Το repository περιέχει τον κώδικα της πειραματικής υλοποίησης. Τα αρχικά event logs δεν περιλαμβάνονται απαραίτητα στο repository και πρέπει να αποκτηθούν από τις αντίστοιχες πηγές, σύμφωνα με τους όρους διάθεσής τους.