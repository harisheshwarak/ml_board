import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def feature_selection(df, top_k=12):
    """A UDF to select the features, which are highly correlated with target feature.
    
    """
    X = df.drop(['Student_ID', 'placement_status', 'salary_lpa'], axis=1)
    X[X.columns] = X[X.columns].apply(LabelEncoder().fit_transform)
    
    y = [1 if v=='Placed' else 0 for v in df['placement_status'].to_list() ]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SelectKBest with chi2
    selector = SelectKBest(score_func=chi2, k=top_k)
    selector.fit(X_scaled, y)
    
    # Get selected feature names and scores
    selected_features = X.columns[selector.get_support()]
    scores = selector.scores_[selector.get_support()]

    return selected_features.to_list()

def train_test_datasplit(df, selected_features):
    """A UDF to split the dataset into train, test.
    """

    df[selected_features] = df[selected_features].apply(LabelEncoder().fit_transform)
    X = df[selected_features]
    y = [1 if v=='Placed' else 0 for v in df['placement_status'].to_list() ]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_test(model_dict, X_train, X_test, y_train, y_test):
    """
    """
    df_result = []
    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        evaluation = {}
        evaluation["Model"] = model_name
        evaluation["Accuracy"] = accuracy_score(y_test, y_pred)
        evaluation["AUC Score"] = float(roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A")
        evaluation["Precision"] = precision_score(y_test, y_pred)
        evaluation["Recall"] = recall_score(y_test, y_pred)
        evaluation["F1 Score"] = f1_score(y_test, y_pred)
        evaluation["MCC Score"] = float(matthews_corrcoef(y_test, y_pred))
        df_name = f"temp_{model_name}"
        df_name = pd.DataFrame(evaluation, index=[0])
        df_result.append(df_name)
    df_master = pd.concat(df_result)
    df_master.reset_index(drop=True, inplace=True)
    return df_master
model_dict = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')    
}
def main():
    """
    """
    dataset = pd.read_csv("./data/student_placement_dataset.csv")
    features_set = feature_selection(dataset)
    X_train, X_test, y_train, y_test = train_test_datasplit(dataset, features_set)
    df_result = train_and_test(model_dict, X_train, X_test, y_train, y_test)
    return df_result

df = main()

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")

st.title("Model Evaluation Dashboard")
st.write("This dashboard displays metrics for all classification models.")

# Display Table
st.subheader(" Metrics Table")
st.dataframe(df, use_container_width=True)

st.subheader("Compare Models by Metric")

metric = st.selectbox(
    "Select a metric to visualize",
    ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
)

fig = px.bar(
    df,
    x="Model",
    y=metric,
    color="Model",
    title=f"{metric} Comparison Across Models",
    text=metric,
    height=500
)

fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Best Model Based on Selected Metric")

best_model = df.loc[df[metric].idxmax()]
st.success(f"**{best_model['Model']}** is the best model for **{metric}** with a score of **{best_model[metric]:.2f}**.")