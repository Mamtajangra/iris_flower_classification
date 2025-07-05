# Iris Flower Classification using Machine Learning

This project demonstrates how to classify **Iris flowers into three species** — *Setosa*, *Versicolor*, and *Virginica* — using four simple measurements and a machine learning model.

---

## About the Dataset

The **Iris flower dataset** is a classic dataset in pattern recognition and statistics. It contains **150 samples** of iris flowers, each described by four features:

- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

**Goal:** Predict the species of an iris flower based on these four measurements.

---

## Visual Reference: Iris Flower Species

| Setosa | Versicolor | Virginica |
|---<img src="images/iris_setosha.jpg" width="200"/>----|--- <img src="images/iris_versicolor.jpg" width="200"/>--------|---<img src="images/iris_virginica.jpg" width="200"/> -------|


---

## Detailed Explanation of Species

### 1. Iris Setosa
- **Description:** Iris Setosa is easily identified by its small size and bright blue or violet petals.
- **Features:** Short petals and sepals, broad leaves.
- **Habitat:** Native to Europe and Asia, often found in meadows and wetlands.

### 2. Iris Versicolor
- **Description:** Also known as the Blue Flag Iris, Versicolor has medium-sized flowers with a mix of blue, purple, and violet shades.
- **Features:** Medium petal and sepal length, slightly narrower than Setosa.
- **Habitat:** Common in North America, especially in marshes and along stream banks.

### 3. Iris Virginica
- **Description:** Virginica, or the Southern Blue Flag, is known for its larger flowers and deeper blue or purple color.
- **Features:** Longest petals and sepals among the three, slender leaves.
- **Habitat:** Found in eastern North America, prefers wet meadows and marshes.

---

## Project Workflow

1. **Load and explore the dataset**
2. **Preprocess** the data using `StandardScaler` for normalization
3. **Train** a `Logistic Regression` model
4. **Predict** on the test set and evaluate with `accuracy_score`

---

## Sample Code

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
```

---

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:
```sh
pip install pandas numpy matplotlib seaborn scikit