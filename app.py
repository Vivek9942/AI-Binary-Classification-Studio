import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Binary Classification Studio",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Main Title
# ---------------------------
st.title("🤖 AI Binary Classification Studio")
st.markdown("Upload any **Binary Classification CSV Dataset** and perform the complete Machine Learning workflow.")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Upload Dataset",
    "📄 Dataset Preview",
    "🧹 Data Preprocessing",
    "🤖 Model Training",
    "🔮 Prediction"
])

# ---------------------------
# Upload Dataset
# ---------------------------
with tab1:

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.session_state["df"] = df

        st.success("Dataset uploaded successfully!")

        rows, cols = df.shape

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", rows)
        col2.metric("Columns", cols)
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())

        binary_columns = [

            col for col in df.columns

            if df[col].dropna().nunique() == 2

        ]

        target_column = st.selectbox(
            "Select Target Column",
            binary_columns
        )

        st.session_state["target_column"] = target_column
        target_dtype = df[target_column].dtype

        from pandas.api.types import is_numeric_dtype

        if is_numeric_dtype(df[target_column]):

            st.subheader("Prediction Labels")

            class0_label = st.text_input(
                "Prediction for Class 0",
                value="Class 0"
            )

            class1_label = st.text_input(
                "Prediction for Class 1",
                value="Class 1"
            )

            st.session_state["class0_label"] = class0_label
            st.session_state["class1_label"] = class1_label

with tab2:

    if "df" in st.session_state:

        df = st.session_state["df"]

        st.subheader("Dataset Preview")

        preview_option = st.radio(
            "Select Preview",
            ["First 5 Rows", "Last 5 Rows", "Random 5 Rows"],
            horizontal=True
        )

        if preview_option == "First 5 Rows":
            st.dataframe(df.head(), use_container_width=True)

        elif preview_option == "Last 5 Rows":
            st.dataframe(df.tail(), use_container_width=True)

        else:
            st.dataframe(df.sample(5), use_container_width=True)

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info_df, use_container_width=True)

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

    else:
        st.info("Please upload a dataset first.")

with tab3:

    if "df" in st.session_state:

        df = st.session_state["df"]
        # Save original dataset
        st.session_state["original_df"] = df.copy()

        # Create a copy of original dataset
        cleaned_df = df.copy()
        # Save original datatypes
        original_dtypes = df.dtypes.to_dict()

        st.session_state["original_dtypes"] = original_dtypes

        # Remove duplicate rows
        cleaned_df.drop_duplicates(inplace=True)
        # Remove constant columns

        constant_columns = [

            col for col in cleaned_df.columns

            if cleaned_df[col].nunique(dropna=False) <= 1

        ]

        cleaned_df.drop(
            columns=constant_columns,
            inplace=True
        )

        # Remove ID columns automatically
        id_columns = [
            col for col in cleaned_df.columns
            if "id" in col.lower()
        ]

        cleaned_df.drop(columns=id_columns, inplace=True, errors="ignore")

        # Convert boolean columns to integers

        bool_columns = cleaned_df.select_dtypes(
            include="bool"
        ).columns

        for col in bool_columns:

            cleaned_df[col] = cleaned_df[col].astype(int)

        # Separate numerical and categorical columns

        target_column = st.session_state["target_column"]

        numerical_columns = [

            col for col in cleaned_df.select_dtypes(
                include=["int64", "float64"]
            ).columns

            if col != target_column

        ]

        st.session_state["numerical_columns"] = list(
        numerical_columns
        )

        target_column = st.session_state["target_column"]

        categorical_columns = [

            col for col in cleaned_df.select_dtypes(
                include=["object"]
            ).columns

            if col != target_column

        ]
        st.session_state["categorical_columns"] = list(
            categorical_columns
        )
        # Fill missing numerical values

        for col in numerical_columns:
            cleaned_df[col].fillna(
                cleaned_df[col].median(),
                inplace=True
            )

        # Fill missing categorical values

        for col in categorical_columns:
            cleaned_df[col].fillna(
                cleaned_df[col].mode()[0],
                inplace=True
            )

        label_encoders = {}

        # Store original categories
        category_values = {}

        for col in categorical_columns:

            category_values[col] = sorted(
                cleaned_df[col].dropna().unique().tolist()
            )

        st.session_state["category_values"] = category_values

        # -----------------------------
        # Encode Target Column
        # -----------------------------

        target_encoder = None

        if cleaned_df[target_column].dtype == "object":

            target_encoder = LabelEncoder()

            cleaned_df[target_column] = target_encoder.fit_transform(
                cleaned_df[target_column]
            )

        st.session_state["target_encoder"] = target_encoder

        for col in categorical_columns:

            if col == target_column:
                continue

            encoder = LabelEncoder()

            cleaned_df[col] = encoder.fit_transform(
                cleaned_df[col]
            )

            label_encoders[col] = encoder

        # Save cleaned dataset
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["label_encoders"] = label_encoders



        st.subheader("🧹 Data Preprocessing")

        rows_after = cleaned_df.shape[0]
        cols_after = cleaned_df.shape[1]
        duplicate_rows = cleaned_df.duplicated().sum()
        missing_values = cleaned_df.isnull().sum().sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", rows_after)
        col2.metric("Columns", cols_after)
        col3.metric("Duplicate Rows", duplicate_rows)
        col4.metric("Missing Values", missing_values)

        st.subheader("Cleaned Dataset")

        st.dataframe(
            cleaned_df.head(),
            use_container_width=True
        )

        st.download_button(
            label="⬇ Download Cleaned Dataset",
            data=cleaned_df.to_csv(index=False),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    else:

        st.info("Please upload a dataset first.")

with tab4:

    if "cleaned_df" in st.session_state:

        cleaned_df = st.session_state["cleaned_df"]

        target_column = st.session_state["target_column"]

        st.subheader("🤖 Model Training")

        X = cleaned_df.drop(columns=[target_column], errors="ignore")
        

        # Remove score columns that directly leak the target
        leakage_columns = [
            "Final_Exam_Score",
            "Final Score",
            "final_exam_score",
            "Score",
            "Marks",
            "FinalMarks"
        ]

        X = X.drop(
            columns=[col for col in leakage_columns if col in X.columns],
            errors="ignore"
        )

        st.session_state["feature_columns"] = X.columns.tolist()

        y = cleaned_df[target_column]

        target_encoder = None

        if y.dtype == "object":

            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

        st.session_state["target_encoder"] = target_encoder

        try:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

        except ValueError:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "Random Forest": RandomForestClassifier(random_state=42),

            "Gradient Boosting": GradientBoostingClassifier(random_state=42)

        }

        results = []

        best_accuracy = 0
        best_model = None
        best_model_name = ""

        for model_name, model in models.items():

            if model_name == "Logistic Regression":

                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)

            else:

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            positive_label = sorted(y.unique())[-1]

            precision = precision_score(
                y_test,
                predictions,
                pos_label=positive_label,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                pos_label=positive_label,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                pos_label=positive_label,
                zero_division=0
            )
            results.append([
                model_name,
                accuracy,
                precision,
                recall,
                f1
            ])

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        )
        st.session_state["results_df"] = results_df
        st.session_state["best_model"] = best_model
        st.session_state["best_model_name"] = best_model_name
        st.session_state["scaler"] = scaler

        st.subheader("Training Results")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.success(f"🏆 Best Model: {best_model_name}")

        st.metric(
            "Best Accuracy",
            f"{best_accuracy:.4f}"
        )       
    else:

        st.info("Please upload and preprocess a dataset first.")

with tab5:

    if "best_model" in st.session_state:

        best_model = st.session_state["best_model"]

        cleaned_df = st.session_state["cleaned_df"]

        target_column = st.session_state["target_column"]

        label_encoders = st.session_state["label_encoders"]

        scaler = st.session_state["scaler"]

        st.subheader("🔮 Prediction")

        feature_columns = [
            col for col in cleaned_df.columns
            if col != target_column
        ]

        # Remove leakage columns
        leakage_columns = [
            "Final_Exam_Score",
            "Final Score",
            "final_exam_score",
            "Score",
            "Marks",
            "FinalMarks"
        ]

        feature_columns = [
            col for col in feature_columns
            if col not in leakage_columns
        ]
        user_input = {}

        category_values = st.session_state["category_values"]

        numerical_columns = st.session_state["numerical_columns"]

        for col in feature_columns:

            if col in category_values:

                user_input[col] = st.selectbox(
                    col,
                    category_values[col]
                )

            else:

                default_value = float(cleaned_df[col].median())

                user_input[col] = st.number_input(
                    col,
                    value=default_value
                )

        input_df = pd.DataFrame([user_input])
        if st.button("🔮 Predict"):

            prediction_data = input_df.copy()

            categorical_columns = st.session_state["categorical_columns"]

            for col in categorical_columns:

                if col not in prediction_data.columns:
                    continue

                try:

                    prediction_data[col] = label_encoders[col].transform(
                        prediction_data[col]
                    )

                except ValueError:

                    st.error(
                        f"Invalid value selected for '{col}'."
                    )

                    st.stop()

            # Arrange columns exactly like training
            prediction_data = prediction_data.reindex(
                columns=st.session_state["feature_columns"]
            )

            # Scale only for Logistic Regression
            if best_model.__class__.__name__ == "LogisticRegression":

                prediction_data = scaler.transform(prediction_data)
            prediction = best_model.predict(
                prediction_data
            )[0]

            target_encoder = st.session_state["target_encoder"]

            if target_encoder is not None:

                prediction_text = target_encoder.inverse_transform([prediction])[0]

            else:

                class0 = st.session_state.get(
                    "class0_label",
                    str(sorted(st.session_state["original_df"][target_column].unique())[0])
                )

                class1 = st.session_state.get(
                    "class1_label",
                    str(sorted(st.session_state["original_df"][target_column].unique())[1])
                )
                if prediction == 0:
                    prediction_text = class0
                else:
                    prediction_text = class1

            if hasattr(best_model, "predict_proba"):

                probability = best_model.predict_proba(
                    prediction_data
                )[0]

                confidence = float(max(probability))

            else:

                confidence = 1.0


            st.success(f"Prediction : {prediction_text}")
            st.subheader("Prediction Confidence")

            st.progress(confidence)

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.divider()

            st.caption(
                f"Model Used : {st.session_state['best_model_name']}"
            )

    else:

        st.info("Please train the models first.")