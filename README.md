# 🍷 Wine Quality Prediction - Decision Tree Model

## 📌 Opis projektu
Ten projekt przewiduje jakość wina na podstawie jego cech chemicznych.  
Używamy modelu **drzewa decyzyjnego**, aby klasyfikować próbki win jako dobre lub złe.

## 📊 Dane
Dane pochodzą z [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
Zawierają cechy chemiczne win czerwonych i białych oraz ich ocenę jakości.

- **`winequality-red.csv`** → Dane dla czerwonego wina 🍷  
- **`winequality-white.csv`** → Dane dla białego wina 🍾  
- **`winequality.names.csv`** → Opis kolumn  

## 🛠 Instalacja

Aby uruchomić projekt, zainstaluj wymagane biblioteki:
```bash
pip install pandas numpy scikit-learn matplotlib

```

## 📈 Wyniki

Model drzewa decyzyjnego osiąga ~80% dokładności w przewidywaniu jakości wina.

