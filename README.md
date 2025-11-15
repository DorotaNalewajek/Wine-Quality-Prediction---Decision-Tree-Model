# 🍷 Wine Quality Machine Learning

Mini machine-learning project that predicts **wine quality** based on its chemical properties.  
The repo contains **two classic ML models**:

- 🌳 **Decision Tree Classifier**
- 🤝 **k-Nearest Neighbours (k-NN)**

## Goal: practise the full ML workflow on a real dataset – from loading CSVs to evaluating and comparing models.

---

## 📁 Project Structure

wine_quality_machine_learning/
├── decision_tree/
│   └── wine_quality_decisiontree_load_from.py   # Decision Tree model
├── knn/
│   └── winequality_knn_model.py                 # k-NN model
├── wine_quality_data/
│   └── data/
│       ├── wine_quality-red.csv
│       ├── wine_quality-white.csv
│       └── wine_quality.names.csv
├── .gitignore
└── README.md


## 📊 Data Description

The project uses a public Wine Quality dataset (red & white wine).
Each row describes one wine sample with features such as:
	•	fixed acidity, volatile acidity, citric acid
	•	residual sugar, chlorides, free / total sulfur dioxide
	•	density, pH, sulphates, alcohol
	•	quality – the target label (integer score, e.g. 3–8)

All CSV files live in wine_quality_data/data/.

----

## ⚙️ Requirements

- Python **3.10+**
- Recommended packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install them (once) with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

# ▶️ How to Run

## 1️⃣ Decision Tree model (🌳)

```bash
cd decision_tree
python wine_quality_decisiontree_load_from.py
```

##  - The script:
	1.	Loads data from ../wine_quality_data/data/.
	2.	Splits it into train / test sets.
	3.	Trains a DecisionTreeClassifier.
	4.	Prints accuracy and basic metrics.
	5.	Optionally plots the tree / feature importance.

⸻

## 2️⃣ k-Nearest Neighbours model (🤝)

```bash
cd knn
python winequality_knn_model.py
```

## 📈 Results – High Level

The main focus is learning, not leaderboard scores – models are only lightly tuned.


## In my experiments:
	•	🌳 Decision Tree: around ~80% test accuracy
	•	🤝 k-NN: similar accuracy, depending on k and scaling

Exact numbers may vary between runs (random train/test split).

⸻

## 🎯 What I Practised Here
	•	Working with real CSV data in Python.
	•	Building a full ML pipeline with scikit-learn:
	•	loading → preprocessing → training → evaluation → visualisation.
	•	Comparing two classic algorithms:
	•	Decision Tree vs k-Nearest Neighbours.
	•	Organising a small ML repo for my portfolio:
	•	clear folder structure for data and models,
	•	readable, beginner-friendly code layout.

## 🚀 Possible Next Steps
	•	Add hyperparameter search (GridSearchCV / RandomizedSearchCV).
	•	Use cross-validation instead of a single train/test split.
	•	Add more visualisations (confusion matrix, feature importance).
	•	Try extra models (Random Forest, Gradient Boosting, etc.).
	•	Wrap the best model into a small API or CLI tool.

⸻

## 👩‍💻 Author

Dorota Nalewajek – future AI / ML developer & wine-quality detective 🍷🤖



