Machine Learning Model Evaluation Dashboard

This project is an interactive **Streamlit application** that performs **feature selection**, trains **multiple machine learning classification models**, evaluates them using standard metrics, and visualizes the results in a clean dashboard.

The dataset used is a **student placement dataset**, and the goal is to compare different ML models on their ability to predict placement outcomes.


## Features

### **Automated Feature Selection**
Uses **SelectKBest (chi‑square)** to select the top‑k features most correlated with the target (`placement_status`).

### **Train/Test Split + Scaling**
- Encodes categorical features  
- Splits dataset into training and testing sets  
- Applies **StandardScaler** for normalization  

### **Model Training & Evaluation**
Trains the following classification models:

- Logistic Regression  
- Decision Tree  
- K‑Nearest Neighbors  
- Naive Bayes  
- Random Forest  
- XGBoost (optional if installed)

Each model is evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### **Interactive Streamlit Dashboard**
- Displays evaluation metrics in a table  
- Provides bar‑chart comparisons for each metric  
- Highlights the best model dynamically  

---

## Project Structure

```
ml_board/
│
├── data/
│   └── student_placement_dataset.csv
│
├── app.py
├── README.md
└── requirements.txt
```

---

## Code Overview

### Feature Selection

```python
def feature_selection(df, top_k=12):
    X = df.drop(['Student_ID', 'placement_status', 'salary_lpa'], axis=1)
    X[X.columns] = X[X.columns].apply(LabelEncoder().fit_transform)

    y = [1 if v=='Placed' else 0 for v in df['placement_status'].to_list()]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=chi2, k=top_k)
    selector.fit(X_scaled, y)

    selected_features = X.columns[selector.get_support()]
    return selected_features.to_list()
```

### Train/Test Split

```python
def train_test_datasplit(df, selected_features):
    df[selected_features] = df[selected_features].apply(LabelEncoder().fit_transform)
    X = df[selected_features]
    y = [1 if v=='Placed' else 0 for v in df['placement_status'].to_list()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test
```

### Model Training & Evaluation

```python
def train_and_test(model_dict, X_train, X_test, y_train, y_test):
    df_result = []
    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        evaluation = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": float(roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC Score": float(matthews_corrcoef(y_test, y_pred))
        }

        df_result.append(pd.DataFrame(evaluation, index=[0]))

    df_master = pd.concat(df_result).reset_index(drop=True)
    return df_master
```
Evaluation metric:
<img width="1045" height="173" alt="image" src="https://github.com/user-attachments/assets/4b13842f-8501-4429-af41-68aa800fe359" />

-------------------------------------------------

Observations:
<img width="798" height="602" alt="image" src="https://github.com/user-attachments/assets/5be71f7c-1bec-46bf-aa5b-8d7740635a13" />

---

## Running the Streamlit App

### Install dependencies

```
pip install -r requirements.txt
```

### Run the app

```
streamlit run app.py
```

### Open the browser  
Streamlit will automatically open the dashboard at:

```
http://localhost:8501
```

---

## Dashboard Preview

The app includes:

- A metrics table  
- A dropdown to select evaluation metric  
- A bar chart comparing all models  
- A highlight of the best model  

---

## Requirements

Add this to `requirements.txt`:

```
streamlit
pandas
scikit-learn
plotly
xgboost
```
