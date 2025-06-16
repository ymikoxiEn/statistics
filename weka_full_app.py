import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(page_title="Statistics", layout="wide")
st.title("Statistical Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("⚙️ Preprocessing Options")
    remove_na = st.checkbox("🔧 Replace Missing Values")
    normalize = st.checkbox("📏 Normalize Numeric Data")
    encode = st.checkbox("🔤 Encode Categorical Features")

    target_col = st.selectbox("🎯 Select Target Column", df.columns)
    features = df.drop(columns=[target_col])
    target = df[target_col]

    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object', 'category']).columns

    numeric_pipeline = []
    if remove_na:
        numeric_pipeline.append(('imputer', SimpleImputer(strategy='mean')))
    if normalize:
        numeric_pipeline.append(('scaler', StandardScaler()))

    cat_pipeline = []
    if remove_na:
        cat_pipeline.append(('imputer', SimpleImputer(strategy='most_frequent')))
    if encode:
        cat_pipeline.append(('encoder', OneHotEncoder(handle_unknown='ignore')))

    preprocessor = ColumnTransformer([
        ('num', Pipeline(numeric_pipeline), numeric_features),
        ('cat', Pipeline(cat_pipeline), categorical_features)
    ])

    st.subheader("🧪 Model Selection & Training")
    test_size = st.slider("🔍 Test Size (for splitting)", 0.1, 0.5, 0.2, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    if st.button("🚀 Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📈 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("✅ Classifier Accuracy")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.success("✅ Model Trained and Evaluated!")

    st.sidebar.title("🔍 Select Analysis Type")
    option = st.sidebar.selectbox("Choose category", [
        "Correlation Analysis", "ANOVA", "T-Test", "Frequency Count", "Regression"
    ])

    if option == "Correlation Analysis":
        st.header("📈 Correlation Analysis")
        numeric_df = df.select_dtypes(include=[np.number])

        st.subheader("1. Pearson Correlation")
        pearson_corr = numeric_df.corr(method='pearson')
        st.dataframe(pearson_corr.round(3))
        fig, ax = plt.subplots()
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("2. Spearman Correlation")
        spearman_corr = numeric_df.corr(method='spearman')
        st.dataframe(spearman_corr.round(3))

        st.subheader("3. Kendall's Tau")
        kendall_corr = numeric_df.corr(method='kendall')
        st.dataframe(kendall_corr.round(3))

    elif option == "ANOVA":
        st.subheader("📊 ANOVA Test")
        group_col = st.selectbox("Select Grouping Column (categorical)", df.columns)
        value_col = st.selectbox("Select Numeric Column for Comparison", df.select_dtypes(include=np.number).columns)

        if group_col and value_col:
            try:
                 groups = [group[value_col].dropna() for name, group in df.groupby(group_col)]
                 f_stat, p_val = stats.f_oneway(*groups)
                 st.write(f"**F-statistic:** {f_stat:.4f}")
                 st.write(f"**P-value:** {p_val:.4f}")
            except Exception as e:
                 st.error(f"Error in ANOVA computation: {e}")
        else:
             st.warning("Please select both a group column and a value column.")


    elif option == "T-Test":
        st.header("📊 Independent T-Test")
        group_col = st.selectbox("Select Grouping Column (categorical)", df.columns)
        value_col = st.selectbox("Select Numeric Column for Comparison", df.select_dtypes(include=np.number).columns)
        groups = list(df[group_col].dropna().unique())
        if len(groups) >= 2:
            data1 = df[df[group_col] == groups[0]][value_col].dropna()
            data2 = df[df[group_col] == groups[1]][value_col].dropna()
            t_stat, p_val = stats.ttest_ind(data1, data2)
            st.write(f"T-statistic: {t_stat:.3f}")
            st.write(f"p-value: {p_val:.5f}")

    elif option == "Frequency Count":
        st.header("📋 Frequency Count")
        freq_col = st.selectbox("Select Column", df.columns)
        st.dataframe(df[freq_col].value_counts().reset_index().rename(columns={'index': freq_col, freq_col: 'Count'}))

    elif option == "Regression":
        st.header("📉 Linear Regression")
        x_col = st.selectbox("X (Independent Variable)", numeric_features)
        y_col = st.selectbox("Y (Dependent Variable)", numeric_features)
        X = sm.add_constant(df[x_col].dropna())
        y = df[y_col].dropna()
        model = sm.OLS(y, X).fit()
        st.write(model.summary())
