import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


def interpret_with_ollama(prompt, model="llama3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"❌ Error contacting Ollama: {e}"

# Upload interface
st.title("Statistical Analyzer & AI Interpreter")
ref_file = st.file_uploader("📂 Upload Reference File (TXT or CSV)", type=["txt", "csv"], key="reference_file")
dataset_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"], key="dataset_file")

# Auto-clear session state only if files are removed
if 'ref_file_uploaded' not in st.session_state:
    st.session_state['ref_file_uploaded'] = False
if 'dataset_uploaded' not in st.session_state:
    st.session_state['dataset_uploaded'] = False

if not ref_file:
    st.session_state['ref_file_uploaded'] = False
if not dataset_file:
    st.session_state['dataset_uploaded'] = False

# Stop if either file missing
if not ref_file:
    st.warning("⚠️ Please upload a reference file before proceeding.")
    st.stop()
if not dataset_file:
    st.warning("⚠️ Please upload a dataset CSV file in the sidebar to begin.")
    st.stop()

# Process reference file
file_content = ""
file_extension = ref_file.name.split(".")[-1]
try:
    if file_extension == "txt":
        file_content = ref_file.read().decode("utf-8", errors="replace")
    elif file_extension == "csv":
        df_uploaded = pd.read_csv(ref_file)
        file_content = df_uploaded.head(20).to_string(index=False)
    st.session_state['ref_file_uploaded'] = True
    st.session_state['file_content'] = file_content
except Exception as e:
    st.error(f"Failed to read reference file: {e}")

# Load dataset file
try:
    df = pd.read_csv(dataset_file)
    st.session_state['df'] = df
    st.session_state['dataset_uploaded'] = True
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Preview reference content
st.text_area("📄 File Preview", st.session_state['file_content'], height=200)
st.markdown("---")

# Initialize df from session state if available
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Preprocessing Options
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

    st.subheader("🧪 Model Training")
    classifier_option = st.selectbox("🧠 Select Classifier", [
        "Decision Tree", "Random Forest", "SVM", "Neural Network", "Naive Bayes", "KMeans (Clustering)"
    ])

    test_size_percent = st.slider("Test Size (%)", 10, 90, 20, 5, key="model_test_size")
    test_size = test_size_percent / 100

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size)

    if classifier_option == "Decision Tree":
        classifier = DecisionTreeClassifier()
    elif classifier_option == "Random Forest":
        classifier = RandomForestClassifier()
    elif classifier_option == "SVM":
        classifier = SVC()
    elif classifier_option == "Neural Network":
        classifier = MLPClassifier(max_iter=1000)
    elif classifier_option == "Naive Bayes":
        classifier = GaussianNB()
    elif classifier_option == "KMeans (Clustering)":
        classifier = KMeans(n_clusters=3)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    if st.button("🚀 Train Model"):
        if classifier_option == "KMeans (Clustering)":
            # 🔧 Fit clustering model
            model.fit(features)
            labels = model.named_steps['classifier'].labels_
            st.session_state['cluster_labels'] = labels

            # 📊 Display cluster counts
            st.subheader("📊 Cluster Labels")
            st.write(pd.Series(labels).value_counts())

            # 🔥 Heatmap of feature averages per cluster
            cluster_df = features.copy()
            cluster_df['Cluster'] = labels
            cluster_corr = cluster_df.groupby("Cluster").mean().T
            st.subheader("🔍 Cluster Feature Averages Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cluster_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # 🔭 PCA for Cluster Visualization
            from sklearn.decomposition import PCA
            st.subheader("🔭 2D Cluster Visualization (PCA)")

            features_for_pca = features.copy().dropna()
            if features_for_pca.shape[0] < 2:
                st.warning("Not enough complete rows for PCA visualization after removing NaNs.")
            else:
                pca = PCA(n_components=2)
                reduced_features = pca.fit_transform(features_for_pca)

                # Retrain KMeans on PCA-ready data
                kmeans = model.named_steps['classifier']
                pca_labels = kmeans.predict(features_for_pca)

                pca_df = pd.DataFrame(reduced_features, columns=["PC1", "PC2"])
                pca_df['Cluster'] = pca_labels

                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=60, ax=ax2)
                ax2.set_title("KMeans Cluster Visualization (2D PCA)")
                st.pyplot(fig2)

    else:
        # 🔍 Supervised model training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 🧠 Save for interpretation
        st.session_state['y_pred'] = y_pred
        st.session_state['y_test'] = y_test
        st.session_state['report'] = classification_report(y_test, y_pred)
        st.session_state['accuracy'] = accuracy_score(y_test, y_pred)
        st.session_state['conf_mat'] = confusion_matrix(y_test, y_pred)

        # 📊 Display results
        st.subheader("📊 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("✅ Accuracy")
        st.write(f"Accuracy: {st.session_state['accuracy']:.3f}")

        st.subheader("📉 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(st.session_state['conf_mat'], annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)


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


    elif option == "ANOVA":
        st.header("📊 ANOVA Test (SPSS-style Output)")
        group_col = st.selectbox("Select Grouping Column (categorical)", df.columns)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != group_col]

        if group_col and numeric_cols:
            try:
                anova_results = []
                for value_col in numeric_cols:
                    grouped = df[[group_col, value_col]].dropna().groupby(group_col)
                    groups = [group[value_col].values for name, group in grouped]
                    f_stat, p_val = stats.f_oneway(*groups)

                    grand_mean = df[value_col].mean()
                    ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
                    ss_within = sum(((group - group.mean())**2).sum() for group in groups)
                    df_between = len(groups) - 1
                    df_within = df.shape[0] - len(groups)
                    ms_between = ss_between / df_between
                    ms_within = ss_within / df_within
                    f_calc = ms_between / ms_within
                    N_total = sum(len(group) for group in groups)

                    anova_results.append({
                        "Variable": value_col,
                        "N": N_total,
                        "F-statistic": f_calc,
                        "P-value": p_val,
                        "Sum of Squares Between": ss_between,
                        "df Between": df_between,
                        "MS Between": ms_between,
                        "Sum of Squares Within": ss_within,
                        "df Within": df_within,
                        "MS Within": ms_within
                    })

                anova_df = pd.DataFrame(anova_results)
                st.dataframe(anova_df.round(4))

            except Exception as e:
                st.error(f"Error in ANOVA computation: {e}")
        else:
            st.warning("Please select a grouping column and ensure there are numeric columns to evaluate.")

    elif option == "T-Test":
        st.header("📊 Independent T-Test (SPSS-style Output)")

        group_col = st.selectbox("Select Grouping Column (categorical)", df.columns)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != group_col]

        if group_col and numeric_cols:
            unique_groups = df[group_col].dropna().unique()
            if len(unique_groups) != 2:
                st.warning("T-Test requires exactly 2 groups in the selected column.")
            else:
                try:
                    ttest_results = []
                    group1, group2 = unique_groups[0], unique_groups[1]
                    df_clean = df[[group_col] + numeric_cols].dropna()

                    for col in numeric_cols:
                        data1 = df_clean[df_clean[group_col] == group1][col]
                        data2 = df_clean[df_clean[group_col] == group2][col]

                        n1 = len(data1)
                        n2 = len(data2)

                        if n1 < 2 or n2 < 2:
                            continue

                        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

                        ttest_results.append({
                            "Variable": col,
                            f"{group1} (N)": n1,
                            f"{group1} Mean": round(data1.mean(), 3),
                            f"{group2} (N)": n2,
                            f"{group2} Mean": round(data2.mean(), 3),
                            "T-Statistic": round(t_stat, 3),
                            "Sig. (2-tailed)": round(p_val, 5),
                            "Total Valid N": n1 + n2
                        })

                    if ttest_results:
                        ttest_df = pd.DataFrame(ttest_results)
                        st.dataframe(ttest_df)
                    else:
                        st.warning("Not enough data to perform T-Tests.")

                except Exception as e:
                    st.error(f"Error in T-Test computation: {e}")
        else:
            st.warning("Please select a valid grouping column and ensure numeric columns are available.")


    elif option == "Frequency Count":
        st.header("📋 Frequency Count")
        freq_col = st.selectbox("Select Column", df.columns)
        st.dataframe(df[freq_col].value_counts().reset_index().rename(columns={'index': freq_col, freq_col: 'Count'}))

    elif option == "Regression":
        st.header("📉 Regression")
        x_col = st.selectbox("X (Independent)", numeric_features)
        y_col = st.selectbox("Y (Dependent)", numeric_features)
        reg_df = df[[x_col, y_col]].dropna()
        X_scaled = StandardScaler().fit_transform(reg_df[[x_col]])
        X = sm.add_constant(X_scaled)
        y = reg_df[y_col]
        reg_model = sm.OLS(y, X).fit()

        # ✅ Save to session for AI interpretation
        st.session_state['regression_model'] = reg_model
        st.session_state['regression_summary'] = reg_model.summary().as_text()

        st.write(reg_model.summary())

    # AI INTERPRETATION SECTION
    st.markdown("---")
    st.subheader("Ask AI for Statistical Interpretation")

    selected_model = st.selectbox("⚙️ Choose Ollama model", ["llama3", "llama3.2"])

    ai_option = st.selectbox("Ask AI to interpret:", ["Model Training Results",
        "Dataset Overview", "Correlation Results", "ANOVA Results", "T-Test Results", "Regression Results", "Frequency Count",
    ])

    extra_context = ""
    if file_content:
        extra_context = "\n\nHere is the uploaded file content to use as context:\n" + file_content[:3000]

    prompt = ""
    if ai_option == "Dataset Overview":
        prompt = f"Interpret this dataset summary and highlight potential findings, in simplified/human tone summary, include the preprocessing, is the dataset clean or needed to be cleaned first, the size of the data set and the possible statistical methods to be used:\n{df.describe(include='all').to_string()}{extra_context}"
        
    elif ai_option == "Correlation Results":
        try:
            pearson_corr = df.select_dtypes(include=[np.number]).corr(method='pearson')
            prompt = f"Explain the following correlation matrix, in simplified/human tone summary in one paragraph:\n{pearson_corr.to_string()}{extra_context}"
        except:
            st.warning("Please run correlation analysis first.")

    elif ai_option == "ANOVA Results":
        try:
            group_col = st.selectbox("Grouping Column for AI (ANOVA)", df.columns, key="ai_anova_group")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != group_col]
            grouped_texts = []

            for value_col in numeric_cols:
                grouped = df[[group_col, value_col]].dropna().groupby(group_col)
                groups = [group[value_col].values for name, group in grouped]
                f_stat, p_val = stats.f_oneway(*groups)
                grand_mean = df[value_col].mean()
                ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
                ss_within = sum(((group - group.mean())**2).sum() for group in groups)
                df_between = len(groups) - 1
                df_within = df.shape[0] - len(groups)
                ms_between = ss_between / df_between
                ms_within = ss_within / df_within
                f_calc = ms_between / ms_within

                grouped_texts.append(f"Variable: {value_col}\nSS Between: {ss_between:.4f}, df: {df_between}, MS: {ms_between:.4f}\nSS Within: {ss_within:.4f}, df: {df_within}, MS: {ms_within:.4f}\nF: {f_calc:.4f}, P: {p_val:.4f}\n")

            prompt =  "Interpret these ANOVA results comparing groups, in simplified/human tone summary in one paragraph:\n" + "\n".join(grouped_texts) + extra_context
             
        except:
            st.warning("Please run ANOVA analysis first.")

    elif ai_option == "T-Test Results":
        try:
            group_col = st.selectbox("Grouping Column for AI (T-Test)", df.columns, key="ai_ttest_group")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != group_col]
            groups = list(df[group_col].dropna().unique())
            if len(groups) >= 2:
                ttest_texts = []
                for value_col in numeric_cols:
                    data1 = df[df[group_col] == groups[0]][value_col].dropna()
                    data2 = df[df[group_col] == groups[1]][value_col].dropna()
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    ttest_texts.append(f"Variable: {value_col}\nGroup 1: {groups[0]}, Group 2: {groups[1]}\nT: {t_stat:.4f}, P: {p_val:.4f}\n")

                prompt = "Interpret these T-Test results for numeric variables across two groups, in simplified/human tone summary in one paragraph :\n" + "\n".join(ttest_texts) + extra_context
        except:
            st.warning("Please run T-Test analysis first.")

    elif ai_option == "Regression Results":
        try:
            summary = st.session_state.get('regression_summary', '')
            if summary:
                prompt = f"Interpret the regression output:\n{summary}{extra_context}"
            else:
                st.warning("Please run regression analysis first.")
        except:
            st.warning("Please run regression analysis first.")

    elif ai_option == "Frequency Count":
        try:
            freq_col = st.selectbox("Column for Frequency Count (AI)", df.columns, key="ai_freq_col")
            freq_data = df[freq_col].value_counts().reset_index()
            freq_data.columns = [freq_col, 'Count']
            freq_str = freq_data.to_string(index=False)
            prompt = f"Interpret the following frequency count of the column, in simplified/human tone summary in one paragraph '{freq_col}': {freq_str}{extra_context}"
        except:
            st.warning("Please run frequency count first.")
    
    if ai_option == "Model Training Results":
        if 'y_pred' in st.session_state:
            report = st.session_state['report']
            acc = st.session_state['accuracy']
            cm = pd.DataFrame(st.session_state['conf_mat']).to_string(index=False, header=False)
            prompt = f"""Please interpret the following model training results, in simplified/human tone summary in one paragraph : 
            Classification Report:{report}
            Accuracy: {acc:.4f}
            Confusion Matrix: {cm}{extra_context}"""
        else:
            st.warning("Please train the model first.")
  
    
    if prompt and st.button("Get AI Insight"):
        interpretation = interpret_with_ollama(prompt, model=selected_model)
        st.text_area("📌 AI's Interpretation:", interpretation, height=300)

    # AI Summary Section
    st.subheader("Summarize All AI Interpretations")
    if st.button("Summarize Everything"):
        st.markdown("This will generate a summary of:")
        st.markdown("✅ Dataset overview\n✅ Correlation\n✅ ANOVA\n✅ T-Test\n✅ Regression summary\n✅ Frequency count\n✅ Model training")

        summary_sections = []

        if 'df' in st.session_state:
            df = st.session_state['df']
            summary_sections.append("Dataset Summary:\n" + df.describe(include='all').to_string())
            summary_sections.append("Correlation Matrix:\n" + df.corr().to_string())

        if 'anova_summary' in st.session_state:
            summary_sections.append("ANOVA Results:\n" + st.session_state['anova_summary'])

        if 'ttest_summary' in st.session_state:
            summary_sections.append("T-Test Results:\n" + st.session_state['ttest_summary'])

        if 'regression_summary' in st.session_state:
            summary_sections.append("Regression Summary:\n" + st.session_state['regression_summary'])

        if 'frequency_summary' in st.session_state:
            summary_sections.append("Frequency Counts:\n" + st.session_state['frequency_summary'])

        if 'report' in st.session_state:
            summary_sections.append("Model Training Report:\n" + st.session_state['report'])
            summary_sections.append(f"Accuracy: {st.session_state['accuracy']:.4f}")
            summary_sections.append("Confusion Matrix:\n" + pd.DataFrame(st.session_state['conf_mat']).to_string())

        full_text = "\n\n".join(summary_sections)

        if full_text:
            st.subheader("AI Summary of All Sections")
            prompt = (
                "Please summarize the following statistical analysis into a simplified/human tone summary consisting of two paragraphs. "
                "The first paragraph should summarize the key findings across the dataset overview, correlation, ANOVA, T-Test, regression, frequency count, and model training. "
                "The second paragraph should provide a conclusion based on the findings, and a third paragraph should offer recommendations or implications of the analysis. \n"
                f"{full_text}\n\nReference File:\n{file_content[:10000]}"
            )
            interpretation = interpret_with_ollama(prompt)
            st.text_area("AI Summary", interpretation, height=300)
        else:
            st.warning("No summaries available yet. Run each analysis section first.")
