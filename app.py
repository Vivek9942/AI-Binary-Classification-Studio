from imblearn.over_sampling import SMOTE
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score
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
st.markdown(
    "Upload any **Binary Classification CSV Dataset**. "
    "The **last column** is automatically used as the target label."
)

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

        # -----------------------------------------
        # Target column = last column, automatically
        # -----------------------------------------
        target_column = df.columns[-1]

        st.subheader("🎯 Target Column (auto-detected: last column)")
        st.markdown(f"Target Column → **`{target_column}`**")

        unique_vals = df[target_column].dropna().unique()
        n_unique = len(unique_vals)

        st.write(f"Unique values found: **{n_unique}**")
        st.write("Values:", list(unique_vals))

        if n_unique != 2:

            st.error(
                f"❌ This app only supports **binary classification**. "
                f"The last column `{target_column}` has {n_unique} unique "
                f"values, not 2. Please make sure the last column of your "
                f"CSV is the binary label you want to predict."
            )

            # Don't let the rest of the app proceed with a bad target
            st.session_state.pop("target_column", None)
            st.session_state.pop("target_label_map", None)

        else:

            st.session_state["target_column"] = target_column

            # Work out the encoding order that will actually be used later.
            # LabelEncoder (and a plain sort) both assign 0 -> smaller/
            # alphabetically-first value, 1 -> the other one. We mirror
            # that here so the preview matches training exactly.
            sorted_vals = sorted(unique_vals, key=lambda v: str(v))

            preview_label_map = {
                0: str(sorted_vals[0]),
                1: str(sorted_vals[1])
            }

            st.session_state["target_label_map"] = preview_label_map

            st.subheader("🔢 Encoding that will be used for training")

            enc_col1, enc_col2 = st.columns(2)
            enc_col1.info(f"**0** → {preview_label_map[0]}")
            enc_col2.info(f"**1** → {preview_label_map[1]}")

            st.caption(
                "This is the exact encoding the model will be trained on. "
                "Predictions will be shown using this same mapping."
            )

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

    if "df" in st.session_state and "target_column" in st.session_state:

        df = st.session_state["df"]
        target_column = st.session_state["target_column"]

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

            if col != target_column and cleaned_df[col].nunique(dropna=False) <= 1

        ]

        cleaned_df.drop(
            columns=constant_columns,
            inplace=True
        )

        # Remove ID columns automatically
        id_columns = [
            col for col in cleaned_df.columns
            if "id" in col.lower() and col != target_column
        ]

        cleaned_df.drop(columns=id_columns, inplace=True, errors="ignore")

        # Convert boolean columns to integers

        bool_columns = cleaned_df.select_dtypes(
            include="bool"
        ).columns

        for col in bool_columns:

            cleaned_df[col] = cleaned_df[col].astype(int)

        # Separate numerical and categorical columns

        numerical_columns = [

            col for col in cleaned_df.select_dtypes(
                include=["int64", "float64"]
            ).columns

            if col != target_column

        ]

        st.session_state["numerical_columns"] = list(
            numerical_columns
        )

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

        if not is_numeric_dtype(cleaned_df[target_column]):

            target_encoder = LabelEncoder()

            cleaned_df[target_column] = target_encoder.fit_transform(
                cleaned_df[target_column].astype(str)
            )

            target_label_map = {
                int(i): str(cls) for i, cls in enumerate(target_encoder.classes_)
            }

        else:

            sorted_vals = sorted(cleaned_df[target_column].dropna().unique())
            target_label_map = {
                int(v): str(v) for v in sorted_vals
            }

        st.session_state["target_encoder"] = target_encoder
        st.session_state["target_label_map"] = target_label_map

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

        st.subheader("Confirmed Target Encoding")

        enc_col1, enc_col2 = st.columns(2)
        enc_col1.success(f"**0** → {target_label_map.get(0, '0')}")
        enc_col2.success(f"**1** → {target_label_map.get(1, '1')}")

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

        st.info(
            "Please upload a dataset with a valid 2-class target "
            "column (last column) first."
        )

with tab4:

    if "cleaned_df" in st.session_state:

        cleaned_df = st.session_state["cleaned_df"]

        target_column = st.session_state["target_column"]

        st.subheader("🤖 Model Training")
        # ----------------------------------
        # Select Model
        # ----------------------------------

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            (
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting",
                "Extra Trees"
            )
        )

        train_button = st.button("🚀 Train Selected Model")

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

        # Target column is already encoded to 0/1 ints from tab3
        y = cleaned_df[target_column]

        if train_button:

            try:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )

                # =====================================================
                # Detect Dataset Imbalance
                # =====================================================

                class_counts = y_train.value_counts()

                class_percentages = y_train.value_counts(normalize=True) * 100

                majority_class = class_counts.idxmax()
                minority_class = class_counts.idxmin()

                majority_percentage = class_percentages.max()
                minority_percentage = class_percentages.min()

                # Dataset is considered imbalanced if one class exceeds 70%
                is_imbalanced = majority_percentage > 70

                st.subheader("📊 Class Distribution")

                distribution_df = pd.DataFrame({
                    "Class": class_counts.index,
                    "Count": class_counts.values,
                    "Percentage": class_percentages.values.round(2)
                })

                st.dataframe(distribution_df, use_container_width=True)

                if is_imbalanced:

                    st.warning(
                        f"""
                ⚠️ Imbalanced Dataset Detected

                Majority Class : {majority_class}
                ({majority_percentage:.2f}%)

                Minority Class : {minority_class}
                ({minority_percentage:.2f}%)
                """
                    )

                else:

                    st.success(
                        f"""
                ✅ Balanced Dataset

                Largest Class :
                {majority_percentage:.2f}%
                """
                    )

                # Store for later use
                st.session_state["is_imbalanced"] = is_imbalanced

                # ==========================================================
                # Apply SMOTE only for Imbalanced Dataset
                # ==========================================================

                if is_imbalanced:

                    smote = SMOTE(
                        random_state=42,
                        k_neighbors=5
                    )

                    X_train, y_train = smote.fit_resample(
                        X_train,
                        y_train
                    )

                    st.success("✅ SMOTE Applied Successfully")

                    after_counts = y_train.value_counts()

                    after_percentage = (
                        y_train.value_counts(normalize=True) * 100
                    )

                    after_df = pd.DataFrame({

                        "Class": after_counts.index,

                        "Count": after_counts.values,

                        "Percentage": after_percentage.values.round(2)

                    })

                    st.subheader("📊 Class Distribution After SMOTE")

                    st.dataframe(
                        after_df,
                        use_container_width=True
                    )

            except ValueError:

                is_imbalanced = False

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

                "Logistic Regression": LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    random_state=42
                ),

                "Extra Trees": ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )

            }

            model = models[selected_model]

            if selected_model == "Logistic Regression":

                model.fit(X_train_scaled, y_train)

                probabilities = model.predict_proba(X_test_scaled)[:, 1]

                roc_auc = roc_auc_score(
                    y_test,
                    probabilities
                )

                precisions, recalls, thresholds = precision_recall_curve(
                    y_test,
                    probabilities
                )

                f1_scores = (
                    2 * precisions * recalls /
                    (precisions + recalls + 1e-10)
                )

                best_idx = f1_scores[:-1].argmax()

                best_threshold = float(
                    thresholds[best_idx]
                )

                st.session_state["best_threshold"] = best_threshold

                predictions = (
                    probabilities >= best_threshold
                ).astype(int)

                confidence_probabilities = probabilities

            else:

                # Clear any leftover LR threshold from a previous run
                st.session_state["best_threshold"] = 0.5

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )

                if hasattr(model, "predict_proba"):

                    confidence_probabilities = model.predict_proba(
                        X_test
                    )[:, 1]

                else:

                    confidence_probabilities = None

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            positive_label = sorted(
                y.unique()
            )[-1]

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

            results = [[
                selected_model,
                accuracy,
                precision,
                recall,
                f1
            ]]

            best_model = model
            best_model_name = selected_model
            best_predictions = predictions

            best_score = f1 if is_imbalanced else accuracy

            results_df = pd.DataFrame(
                results,
                columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
            )

            # -----------------------------------------------------------
            # Save THIS model as the one and only active model.
            # Predict tab (tab5) will always use whichever model was
            # trained last, so switching the dropdown + retraining
            # always overwrites the active model.
            # -----------------------------------------------------------
            st.session_state["results_df"] = results_df
            st.session_state["best_model"] = best_model
            st.session_state["best_model_name"] = best_model_name
            st.session_state["scaler"] = scaler
            st.session_state["best_predictions"] = best_predictions
            st.session_state["is_imbalanced"] = is_imbalanced

            st.subheader("Training Results")
            st.dataframe(results_df, use_container_width=True)

            st.success(f"🏆 Trained Model: {best_model_name}")

            if is_imbalanced:
                st.metric("F1 Score", f"{best_score:.4f}")
            else:
                st.metric("Accuracy", f"{best_score:.4f}")

            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, best_predictions)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.dataframe(cm_df, use_container_width=True)

        elif "best_model_name" in st.session_state:

            st.info(
                f"Active trained model → **{st.session_state['best_model_name']}**. "
                f"Pick a different model above and click Train to switch."
            )

    else:

        st.info("Please upload and preprocess a dataset first.")


with tab5:

    if "best_model" not in st.session_state:
        st.info("Please train a model first.")
    else:

        best_model = st.session_state["best_model"]
        cleaned_df = st.session_state["cleaned_df"]
        target_column = st.session_state["target_column"]
        label_encoders = st.session_state["label_encoders"]
        scaler = st.session_state["scaler"]

        target_label_map = st.session_state.get(
            "target_label_map",
            {0: "0", 1: "1"}
        )

        st.subheader("🔮 Prediction")

        st.caption(
            f"Predicting with active model → **{st.session_state['best_model_name']}**"
        )

        feature_columns = [
            c for c in cleaned_df.columns
            if c != target_column
        ]

        leakage_columns = [
            "Final_Exam_Score",
            "Final Score",
            "final_exam_score",
            "Score",
            "Marks",
            "FinalMarks"
        ]

        feature_columns = [
            c for c in feature_columns
            if c not in leakage_columns
        ]

        category_values = st.session_state["category_values"]

        with st.form("prediction_form"):

            user_input = {}

            for col in feature_columns:

                if col in category_values:

                    user_input[col] = st.selectbox(
                        col,
                        category_values[col]
                    )

                else:

                    user_input[col] = st.number_input(
                        col,
                        value=float(cleaned_df[col].median())
                    )

            submitted = st.form_submit_button(
                "🔮 Predict"
            )

        if submitted:

            prediction_data = pd.DataFrame([user_input])

            categorical_columns = st.session_state[
                "categorical_columns"
            ]

            for col in categorical_columns:

                if col not in prediction_data.columns:
                    continue

                try:

                    prediction_data[col] = (
                        label_encoders[col]
                        .transform(prediction_data[col])
                    )

                except ValueError:

                    st.error(
                        f"Invalid category selected for '{col}'."
                    )
                    st.stop()

            prediction_data = prediction_data.reindex(
                columns=st.session_state["feature_columns"]
            )

            model_name = best_model.__class__.__name__

            probability = None
            threshold = None

            if model_name == "LogisticRegression":

                prediction_scaled = scaler.transform(
                    prediction_data
                )

                probability = float(
                    best_model.predict_proba(
                        prediction_scaled
                    )[0][1]
                )

                threshold = st.session_state.get(
                    "best_threshold",
                    0.5
                )

                prediction = int(
                    probability >= threshold
                )

                confidence = max(
                    probability,
                    1 - probability
                )

            else:

                prediction = int(
                    best_model.predict(
                        prediction_data
                    )[0]
                )

                if hasattr(best_model, "predict_proba"):

                    probs = best_model.predict_proba(
                        prediction_data
                    )[0]

                    confidence = float(max(probs))
                    probability = float(probs[1])

                else:

                    confidence = 1.0

            prediction_name = target_label_map.get(
                prediction,
                str(prediction)
            )

            st.success(
                f"Prediction : {prediction} ({prediction_name})"
            )

            st.subheader("Prediction Confidence")

            st.progress(confidence)

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            if probability is not None:

                st.metric(
                    "Positive Class Probability",
                    f"{probability*100:.2f}%"
                )

            if model_name == "LogisticRegression" and threshold is not None:

                st.metric(
                    "Decision Threshold",
                    f"{threshold:.3f}"
                )

            st.divider()

            st.caption(
                f"Model Used : {st.session_state['best_model_name']}"
            )
