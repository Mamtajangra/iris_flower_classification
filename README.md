#  Iris Flower Classification using Machine Learning

This project classifies **Iris flowers into 3 species** — *Setosa*, *Versicolor*, and *Virginica* — using 4 simple measurements and a machine learning model.

---

## About the Dataset

The **Iris flower dataset** is one of the most famous datasets in pattern recognition. It consists of **150 flower samples**, with 4 numeric features for each:

- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

**Goal:** Build a model to predict the flower species from these 4 measurements.

---

##  Visual Reference (Flower Species)

| Setosa | Versicolor | Virginica |
|--------|------------|------------|
| ![Setosa](images/setosa.jpg) | ![Versicolor](images/versicolor.jpg) | ![Virginica](images/virginica.jpg) |

---

##  Project Workflow

1. Load and explore the dataset.
2. Preprocess using `StandardScaler` for normalization.
3. Train a `Logistic Regression` model.
4. Predict on test set and evaluate with `accuracy_score`.

---

##  Sample Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", accuracy)

