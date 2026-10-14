"""
Hybrid KNN-Decision Tree burnout risk demo - CONFIGURATION B
(Information Gain-selected KNN: top-N attributes ranked by the Decision Tree layer, K up to 51).

Setup:   pip install gradio scikit-learn pandas joblib
Run:     python app.py (keep the AI Student Impact CSV in this folder)
"""
import glob
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_FILE = "burnout_hybrid_model_b.joblib"
DATABASE_FILE = os.environ.get("BURNOUT_DATABASE_PATH", "student_burnout.sqlite3")
MODEL_VERSION = 4
CLASSES = ["Low", "Medium", "High"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
K_VALUES = [3, 5, 7, 9, 11, 15, 25, 51]
N_VALUES = [2, 3, 4, 5, 8, 10]  # plus "all attributes" (added at run time)
CV_FOLDS = 5
SEED = 42
TARGET = "Burnout_Risk_Level"
CATEGORICAL_COLUMNS = [
    "Major_Category",
    "Year_of_Study",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Institutional_Policy",
]
FEATURE_COLUMNS = [
    "Major_Category",
    "Year_of_Study",
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Tool_Diversity",
    "Paid_Subscription",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Institutional_Policy",
    "Anxiety_Level_During_Exams",
    "Post_Semester_GPA",
    "Skill_Retention_Score",
]

CHOICES = {
    "Major_Category": ["Arts", "Business", "Humanities", "Medical", "STEM"],
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Primary_Use_Case": [
        "Copywriting/Drafting",
        "Debugging/Troubleshooting",
        "Direct_Answer_Generation",
        "Ideation",
        "Summarizing_Reading",
    ],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
    "Institutional_Policy": [
        "Strict_Ban",
        "Allowed_With_Citation",
        "Actively_Encouraged",
    ],
}
NUMERIC_RANGES = {
    "Pre_Semester_GPA": (0, 4),
    "Post_Semester_GPA": (0, 4),
    "Weekly_GenAI_Hours": (0, None),
    "Tool_Diversity": (0, None),
    "Traditional_Study_Hours": (0, None),
    "Perceived_AI_Dependency": (0, None),
    "Anxiety_Level_During_Exams": (0, None),
    "Skill_Retention_Score": (0, None),
}


def prepare(frame):
    """Encode categorical fields and derive the semester GPA change."""
    frame = frame.copy()
    frame["Paid_Subscription"] = frame["Paid_Subscription"].map(
        lambda value: int(str(value).strip().lower() in ("yes", "true", "1"))
    )
    frame["GPA_Change"] = frame["Post_Semester_GPA"] - frame["Pre_Semester_GPA"]
    return pd.get_dummies(frame, columns=CATEGORICAL_COLUMNS, dtype=int)


def find_dataset():
    matches = sorted(glob.glob("ai_student_impact*.csv"))
    if not matches:
        raise FileNotFoundError(
            "Put the AI Student Impact CSV in this folder "
            "(filename starting with 'ai_student_impact')."
        )
    return matches[0]


def validate_dataset_frame(frame, require_student_id=True):
    frame = frame.copy()
    required = set(FEATURE_COLUMNS + [TARGET])
    if require_student_id:
        required.add("Student_ID")
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")

    frame = (
        frame.drop(columns=["Student_ID"], errors="ignore")
        .drop_duplicates()
        .dropna()
        .copy()
    )
    unknown_targets = sorted(set(frame[TARGET].unique()).difference(CLASS_TO_ID))
    if unknown_targets:
        raise ValueError(f"Unexpected burnout-risk labels: {unknown_targets}")

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | (values < minimum)
        if maximum is not None:
            invalid |= values > maximum
        if invalid.any():
            raise ValueError(
                f"Dataset column {column} has {int(invalid.sum())} values "
                f"outside the expected range {minimum} to {maximum}."
            )
        frame[column] = values

    return frame


def load_dataset(path):
    return validate_dataset_frame(pd.read_csv(path))


def read_uploaded_dataset(uploaded_file):
    """Read and validate a supported student dataset without retaining its student IDs."""
    path = Path(os.fspath(uploaded_file))
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        frame = pd.read_excel(path)
    else:
        raise ValueError("Choose a CSV or Excel file (.csv, .xlsx, .xls, or .xlsm).")
    return validate_dataset_frame(frame, require_student_id=False)


def initialize_database():
    database_path = Path(DATABASE_FILE)
    if str(database_path) != ":memory:":
        database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_FILE)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS import_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                row_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS student_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL REFERENCES import_batches(id),
                burnout_risk_level TEXT NOT NULL,
                weekly_genai_hours REAL NOT NULL,
                gpa_change REAL NOT NULL,
                record_data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_student_records_batch
                ON student_records(batch_id);
            CREATE INDEX IF NOT EXISTS idx_student_records_risk
                ON student_records(burnout_risk_level);
            """
        )
        connection.commit()


def import_dataset_to_database(uploaded_file):
    frame = read_uploaded_dataset(uploaded_file)
    if frame.empty:
        raise ValueError("The file has no valid student records to import.")

    source_path = Path(os.fspath(uploaded_file))
    saved_records = frame.drop(columns=["Student_ID"], errors="ignore").copy()
    saved_records["GPA_Change"] = (
        saved_records["Post_Semester_GPA"] - saved_records["Pre_Semester_GPA"]
    )
    connection = sqlite3.connect(DATABASE_FILE)
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO import_batches (filename, row_count) VALUES (?, ?)",
                (source_path.name, len(saved_records)),
            )
            batch_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO student_records (
                    batch_id, burnout_risk_level, weekly_genai_hours,
                    gpa_change, record_data
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        batch_id,
                        record[TARGET],
                        float(record["Weekly_GenAI_Hours"]),
                        float(record["GPA_Change"]),
                        json.dumps(record, default=str),
                    )
                    for record in saved_records.to_dict(orient="records")
                ),
            )
    finally:
        connection.close()
    return len(saved_records)


def get_upload_dashboard():
    with closing(sqlite3.connect(DATABASE_FILE)) as connection:
        batches = pd.read_sql_query(
            """
            SELECT filename AS File, imported_at AS "Imported at",
                   row_count AS "Rows imported"
            FROM import_batches
            ORDER BY id DESC
            """,
            connection,
        )
        risk_counts = pd.read_sql_query(
            """
            SELECT burnout_risk_level AS "Burnout risk",
                   COUNT(*) AS "Student count"
            FROM student_records
            GROUP BY burnout_risk_level
            """,
            connection,
        )
        risk_averages = pd.read_sql_query(
            """
            SELECT burnout_risk_level AS "Burnout risk",
                   AVG(weekly_genai_hours) AS "Average GenAI hours",
                   AVG(gpa_change) AS "Average GPA change"
            FROM student_records
            GROUP BY burnout_risk_level
            """,
            connection,
        )

    risk_counts = (
        risk_counts.set_index("Burnout risk")
        .reindex(CLASSES, fill_value=0)
        .rename_axis("Burnout risk")
        .reset_index()
    )
    risk_averages = (
        risk_averages.set_index("Burnout risk")
        .reindex(CLASSES, fill_value=0)
        .rename_axis("Burnout risk")
        .reset_index()
    )
    if int(risk_counts["Student count"].sum()) == 0:
        insights = "No uploaded records yet. Upload a dataset to see summaries."
    else:
        total_records = int(risk_counts["Student count"].sum())
        largest_group = risk_counts.loc[risk_counts["Student count"].idxmax()]
        highest_hours = risk_averages.loc[
            risk_averages["Average GenAI hours"].idxmax()
        ]
        avg_gpa_change = float(
            (
                risk_averages["Average GPA change"]
                * risk_counts["Student count"]
            ).sum()
            / total_records
        )
        insights = (
            f"**{total_records:,} valid student records** across {len(batches)} "
            f"uploaded file(s). The largest risk group is **{largest_group['Burnout risk']}** "
            f"({int(largest_group['Student count']):,} records, "
            f"{largest_group['Student count'] / total_records:.1%}). "
            f"The highest average GenAI use is in the **{highest_hours['Burnout risk']}** "
            f"group ({highest_hours['Average GenAI hours']:.1f} hours/week). "
            f"Average GPA change across imported records: **{avg_gpa_change:+.3f}**."
        )
    return batches, risk_counts, risk_averages, insights


def import_uploaded_file(uploaded_file):
    if not uploaded_file:
        raise gr.Error("Choose a CSV or Excel file before importing.")
    try:
        row_count = import_dataset_to_database(uploaded_file)
    except (ImportError, OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise gr.Error(f"Could not import this file: {exc}") from exc
    batches, risk_counts, risk_averages, insights = get_upload_dashboard()
    status = (
        f"Imported **{row_count:,} cleaned records** into the local SQLite database. "
        "Student_ID values are not stored."
    )
    return status, batches, risk_counts, risk_averages, risk_averages, insights


def refresh_upload_dashboard():
    batches, risk_counts, risk_averages, insights = get_upload_dashboard()
    return batches, risk_counts, risk_averages, risk_averages, insights


initialize_database()


def _entropy(counts):
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0
    )
    safe = np.where(probabilities > 0, probabilities, 1.0)
    return -(probabilities * np.log2(safe)).sum(axis=1)


def rank_columns_by_gain(features, labels):
    """Rank every encoded column by its best-split Information Gain (training data only)."""
    parent = _entropy(np.bincount(labels, minlength=len(CLASSES)).reshape(1, -1))[0]
    gains_by_column = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype=float)
        order = np.argsort(values, kind="stable")
        sorted_values, sorted_labels = values[order], labels[order]
        unique_values, starts = np.unique(sorted_values, return_index=True)
        if len(unique_values) < 2:
            continue
        grouped = np.stack(
            [
                np.add.reduceat((sorted_labels == class_id).astype(np.int64), starts)
                for class_id in range(len(CLASSES))
            ],
            axis=1,
        )
        left = np.cumsum(grouped, axis=0)[:-1]
        right = grouped.sum(axis=0, keepdims=True) - left
        left_n, right_n = left.sum(axis=1), right.sum(axis=1)
        gains = parent - (
            left_n / len(labels) * _entropy(left) + right_n / len(labels) * _entropy(right)
        )
        gains_by_column[column] = float(gains.max())
    return sorted(gains_by_column, key=gains_by_column.get, reverse=True)


def get_cv_results(features, labels, ranked_columns):
    """Configuration B: for each top-N attribute set, scan K on identical stratified folds.

    Neighbours are retrieved once per fold; K and N are chosen by cross-validated
    macro-F1 on the training partition only.
    """
    n_values = list(N_VALUES) + [len(ranked_columns)]
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(features, labels))
    max_k = max(K_VALUES)
    rows = []

    for n in n_values:
        columns = ranked_columns[:n]
        fold_scores = {k: {"accuracy": [], "macro_f1": []} for k in K_VALUES}
        for train_indices, validation_indices in folds:
            scaler = MinMaxScaler()
            train_features = scaler.fit_transform(features.iloc[train_indices][columns])
            validation_features = scaler.transform(features.iloc[validation_indices][columns])
            fold_labels = labels[train_indices]
            truth = labels[validation_indices]

            neighbors = KNeighborsClassifier(
                n_neighbors=max_k, metric="euclidean", algorithm="brute"
            )
            neighbors.fit(train_features, fold_labels)
            neighbor_indices = neighbors.kneighbors(
                validation_features, n_neighbors=max_k, return_distance=False
            )
            neighbor_labels = fold_labels[neighbor_indices]
            for k in K_VALUES:
                votes = np.stack(
                    [(neighbor_labels[:, :k] == class_id).sum(axis=1)
                     for class_id in range(len(CLASSES))],
                    axis=1,
                )
                predictions = votes.argmax(axis=1)
                fold_scores[k]["accuracy"].append(accuracy_score(truth, predictions))
                fold_scores[k]["macro_f1"].append(
                    f1_score(truth, predictions, average="macro", zero_division=0)
                )
        for k in K_VALUES:
            rows.append(
                {
                    "Top N": n,
                    "K": k,
                    "CV accuracy": float(np.mean(fold_scores[k]["accuracy"])),
                    "CV macro F1": float(np.mean(fold_scores[k]["macro_f1"])),
                }
            )

    grid = pd.DataFrame(rows)
    best = grid.sort_values(
        ["CV macro F1", "CV accuracy"], ascending=False, ignore_index=True
    ).iloc[0]
    selected_n, selected_k = int(best["Top N"]), int(best["K"])
    best_per_n = (
        grid.sort_values(["CV macro F1", "CV accuracy"], ascending=False)
        .groupby("Top N", as_index=False)
        .first()
        .sort_values("Top N", ignore_index=True)
    )
    return selected_n, selected_k, best_per_n


def get_tree_explanation(features, labels, feature_columns):
    tree_model = make_pipeline(
        MinMaxScaler(),
        DecisionTreeClassifier(criterion="entropy", random_state=SEED),
    )
    tree_model.fit(features, labels)
    tree = tree_model.named_steps["decisiontreeclassifier"]

    attribute_importance = {}
    for column, importance in zip(feature_columns, tree.feature_importances_):
        attribute = next(
            (
                categorical
                for categorical in CATEGORICAL_COLUMNS
                if column.startswith(f"{categorical}_")
            ),
            column,
        )
        attribute_importance[attribute] = (
            attribute_importance.get(attribute, 0.0) + float(importance)
        )

    def entropy(counts):
        totals = counts.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals != 0,
        )
        safe_probabilities = np.where(probabilities > 0, probabilities, 1.0)
        return -(probabilities * np.log2(safe_probabilities)).sum(axis=1)

    best_splits = {}
    parent_entropy = entropy(
        np.bincount(labels, minlength=len(CLASSES)).reshape(1, -1)
    )[0]
    for column in feature_columns:
        values = features[column].to_numpy(dtype=float)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_labels = labels[order]
        unique_values, starts = np.unique(sorted_values, return_index=True)
        if len(unique_values) < 2:
            continue

        grouped_counts = np.stack(
            [
                np.add.reduceat((sorted_labels == class_id).astype(np.int64), starts)
                for class_id in range(len(CLASSES))
            ],
            axis=1,
        )
        cumulative_counts = np.cumsum(grouped_counts, axis=0)[:-1]
        left_counts = cumulative_counts
        right_counts = grouped_counts.sum(axis=0, keepdims=True) - left_counts
        left_sizes = left_counts.sum(axis=1)
        right_sizes = right_counts.sum(axis=1)
        left_entropy = entropy(left_counts)
        right_entropy = entropy(right_counts)
        gains = parent_entropy - (
            left_sizes / len(labels) * left_entropy
            + right_sizes / len(labels) * right_entropy
        )
        best_index = int(np.argmax(gains))
        attribute = next(
            (
                categorical
                for categorical in CATEGORICAL_COLUMNS
                if column.startswith(f"{categorical}_")
            ),
            column,
        )
        candidate = {
            "Attribute": attribute,
            "Best split feature": column,
            "Threshold": float(
                unique_values[best_index]
                + (unique_values[best_index + 1] - unique_values[best_index]) / 2
            ),
            "Left records": int(left_sizes[best_index]),
            "Right records": int(right_sizes[best_index]),
            "Left entropy": float(left_entropy[best_index]),
            "Right entropy": float(right_entropy[best_index]),
            "Best split information gain": float(gains[best_index]),
        }
        if (
            attribute not in best_splits
            or candidate["Best split information gain"]
            > best_splits[attribute]["Best split information gain"]
        ):
            best_splits[attribute] = candidate

    ranking = pd.DataFrame(best_splits.values())
    ranking["Normalized tree importance"] = ranking["Attribute"].map(
        attribute_importance
    ).fillna(0.0)
    ranking = ranking.sort_values(
        "Best split information gain", ascending=False, ignore_index=True
    )

    tree_data = tree.tree_
    root_feature_index = int(tree_data.feature[0])
    root_feature = feature_columns[root_feature_index]
    root_attribute = next(
        (
            categorical
            for categorical in CATEGORICAL_COLUMNS
            if root_feature.startswith(f"{categorical}_")
        ),
        root_feature,
    )
    left = int(tree_data.children_left[0])
    right = int(tree_data.children_right[0])
    weighted_child_entropy = (
        tree_data.weighted_n_node_samples[left] * tree_data.impurity[left]
        + tree_data.weighted_n_node_samples[right] * tree_data.impurity[right]
    ) / tree_data.weighted_n_node_samples[0]
    root_information_gain = tree_data.impurity[0] - weighted_child_entropy
    scaler = tree_model.named_steps["minmaxscaler"]
    root_threshold = (
        tree_data.threshold[0] - scaler.min_[root_feature_index]
    ) / scaler.scale_[root_feature_index]
    root_summary = (
        f"Decision Tree root split: **{root_feature}** "
        f"(attribute: **{root_attribute}**, threshold "
        f"{root_threshold:.4f} in the original feature scale). Root entropy: "
        f"{tree_data.impurity[0]:.4f}; weighted child entropy: "
        f"{weighted_child_entropy:.4f}; information gain: "
        f"**{root_information_gain:.4f}**. "
        "This tree is an explanation/ranking layer only; KNN produces predictions."
    )
    return ranking, root_summary


def train_and_save(path):
    frame = load_dataset(path)
    labels = frame[TARGET].map(CLASS_TO_ID).to_numpy(dtype=int)
    features = prepare(frame.drop(columns=TARGET))
    feature_columns = list(features.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=SEED,
    )

    ranked_columns = rank_columns_by_gain(X_train, y_train)
    selected_n, selected_k, cv_results = get_cv_results(X_train, y_train, ranked_columns)
    selected_columns = ranked_columns[:selected_n]

    def build_model(k):
        return make_pipeline(
            MinMaxScaler(),
            KNeighborsClassifier(n_neighbors=k, metric="euclidean", algorithm="brute"),
        )

    cv_results["Selected"] = np.where(
        (cv_results["Top N"] == selected_n) & (cv_results["K"] == selected_k),
        "yes",
        "",
    )

    model = build_model(selected_k)
    model.fit(X_train[selected_columns], y_train)
    predictions = model.predict(X_test[selected_columns])
    probabilities = model.predict_proba(X_test[selected_columns])
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, predictions, labels=list(range(len(CLASSES))))
    metrics = pd.DataFrame(
        [
            {
                "Class": name,
                "Precision": report[name]["precision"],
                "Recall": report[name]["recall"],
                "F1-score": report[name]["f1-score"],
                "Support": int(report[name]["support"]),
            }
            for name in CLASSES
        ]
        + [
            {
                "Class": "Macro average",
                "Precision": report["macro avg"]["precision"],
                "Recall": report["macro avg"]["recall"],
                "F1-score": report["macro avg"]["f1-score"],
                "Support": int(report["macro avg"]["support"]),
            }
        ]
    )
    accuracy = float(accuracy_score(y_test, predictions))
    distribution = pd.DataFrame(
        {
            "Class": CLASSES,
            "Actual count": np.bincount(y_test, minlength=len(CLASSES)),
            "Predicted count": np.bincount(predictions, minlength=len(CLASSES)),
        }
    )
    confusion_table = pd.DataFrame(
        confusion,
        index=pd.Index(CLASSES, name="Actual"),
        columns=pd.Index(CLASSES, name="Predicted"),
    ).reset_index()
    tree_ranking, tree_root_summary = get_tree_explanation(X_train, y_train, feature_columns)

    bundle = {
        "version": MODEL_VERSION,
        "dataset": path,
        "dataset_size": len(frame),
        "selected_k": selected_k,
        "selected_n": selected_n,
        "selected_columns": selected_columns,
        "model": model,
        "columns": feature_columns,
        "accuracy": accuracy,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "confusion_table": confusion_table,
        "distribution": distribution,
        "cv_results": cv_results,
        "tree_ranking": tree_ranking,
        "tree_root_summary": tree_root_summary,
        "test_probabilities": probabilities,
        "test_labels": y_test,
        "test_predictions": predictions,
    }
    joblib.dump(bundle, MODEL_FILE)
    print(
        f"Hybrid model trained and saved. Selected top-{selected_n} attributes, K={selected_k}; "
        f"held-out accuracy: {accuracy:.4f}"
    )
    return bundle


DATASET_FILE = find_dataset()
DATASET_SIGNATURE = (
    os.path.getsize(DATASET_FILE),
    os.path.getmtime(DATASET_FILE),
)
if os.path.exists(MODEL_FILE):
    bundle = joblib.load(MODEL_FILE)
else:
    bundle = {}
if (
    bundle.get("version") != MODEL_VERSION
    or bundle.get("dataset") != DATASET_FILE
    or bundle.get("dataset_signature") != DATASET_SIGNATURE
):
    bundle = train_and_save(DATASET_FILE)
    bundle["dataset_signature"] = DATASET_SIGNATURE
    joblib.dump(bundle, MODEL_FILE)

if "K" not in bundle["cv_results"].columns:
    bundle["cv_results"] = bundle["cv_results"].rename(columns={"Best K": "K"})
bundle["cv_results"] = bundle["cv_results"].drop(
    columns=["Test accuracy", "Test macro F1"], errors="ignore"
)
bundle["cv_results"]["Selected"] = np.where(
    (bundle["cv_results"]["Top N"] == bundle["selected_n"])
    & (bundle["cv_results"]["K"] == bundle["selected_k"]),
    "yes",
    "",
)
model = bundle["model"]
COLUMNS = bundle["columns"]
SELECTED_K = bundle["selected_k"]
SELECTED_N = bundle["selected_n"]
SELECTED_COLUMNS = bundle["selected_columns"]
TEST_ACCURACY = bundle["accuracy"]


def predict(
    major,
    year,
    pre_gpa,
    hours,
    use_case,
    skill,
    tools,
    paid,
    trad_hours,
    dependency,
    policy,
    anxiety,
    post_gpa,
    retention,
):
    row = pd.DataFrame(
        [
            {
                "Major_Category": major,
                "Year_of_Study": year,
                "Pre_Semester_GPA": pre_gpa,
                "Weekly_GenAI_Hours": hours,
                "Primary_Use_Case": use_case,
                "Prompt_Engineering_Skill": skill,
                "Tool_Diversity": tools,
                "Paid_Subscription": paid,
                "Traditional_Study_Hours": trad_hours,
                "Perceived_AI_Dependency": dependency,
                "Institutional_Policy": policy,
                "Anxiety_Level_During_Exams": anxiety,
                "Post_Semester_GPA": post_gpa,
                "Skill_Retention_Score": retention,
            }
        ]
    )
    features = prepare(row).reindex(columns=COLUMNS, fill_value=0)[SELECTED_COLUMNS]
    probabilities = model.predict_proba(features)[0]
    prediction = CLASSES[int(np.argmax(probabilities))]
    summary = (
        f"### Predicted burnout risk: **{prediction.upper()}**\n\n"
        f"**Configuration B:** top {SELECTED_N} attributes, K={SELECTED_K} "
        f"(chosen by {CV_FOLDS}-fold training-only cross-validation)\n\n"
        "**Attributes used:** " + ", ".join(SELECTED_COLUMNS) + "\n\n"
        f"Model held-out accuracy: {TEST_ACCURACY:.1%}. "
        "This class-project demo is not a diagnostic tool."
    )
    return {name: float(score) for name, score in zip(CLASSES, probabilities)}, summary


APP_CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.app-hero {
    padding: 30px 34px; border-radius: 20px; margin-bottom: 20px;
    color: #f8fafc; background: linear-gradient(125deg, #172554, #4338ca 58%, #0891b2);
    box-shadow: 0 16px 40px rgba(30, 41, 99, .18);
}
.app-hero h1 {font-size: clamp(1.8rem, 4vw, 2.6rem); margin: 0 0 8px; color: white;}
.app-hero p {max-width: 760px; margin: 0; color: #e0e7ff; font-size: 1rem;}
#input-panel, #result-panel {
    padding: 20px !important; border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important; background: white !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
}
.dark #input-panel {
    border-color: #334155 !important; background: #1e293b !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .18);
}
.dark #result-panel {
    border-color: #334155 !important; background: #1e293b !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .18);
}
.hero-kicker {
    display: inline-block; margin-bottom: 12px; padding: 5px 10px; border-radius: 999px;
    background: rgba(255,255,255,.15); color: #e0f2fe; font-size: .78rem;
    font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
}
.section-title {margin: 4px 0 2px; font-size: 1.1rem; font-weight: 700;}
.section-hint {margin: 0 0 12px; color: #64748b; font-size: .9rem;}
.dark .section-hint {color: #94a3b8;}
.dark .fine-print {color: #94a3b8;}
.prediction-note {
    padding: 13px 16px; border: 1px solid #bae6fd; border-radius: 12px;
    background: #f0f9ff; color: #0c4a6e; line-height: 1.55;
}
.dark .prediction-note {
    border-color: #164e63; background: #0c2638; color: #bae6fd;
}
.fine-print {color: #64748b; font-size: .86rem;}
#predict-button {min-height: 50px; border-radius: 12px; font-size: 1rem; font-weight: 700;}
"""
APP_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0b1220",
    body_text_color="#0f172a",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#475569",
    body_text_color_subdued_dark="#94a3b8",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_border_color="#e2e8f0",
    block_border_color_dark="#334155",
    block_label_background_fill="#f1f5f9",
    block_label_background_fill_dark="#334155",
    block_label_text_color="#334155",
    block_label_text_color_dark="#e2e8f0",
    input_background_fill="#ffffff",
    input_background_fill_dark="#0f172a",
    input_border_color="#cbd5e1",
    input_border_color_dark="#475569",
    slider_color="#4f46e5",
    slider_color_dark="#818cf8",
)

with gr.Blocks(title="Student Burnout Risk Predictor") as demo:
    gr.HTML(
        """
        <header class="app-hero">
            <span class="hero-kicker">Student well-being · ML demo</span>
            <h1>Burnout Risk Predictor</h1>
            <p>Explore how study habits and generative-AI use relate to a modelled
            burnout-risk category. Enter a student profile to get a prediction
            and see the model's class probabilities.</p>
        </header>
        """
    )
    with gr.Tabs():
        with gr.Tab("Upload data & insights"):
            gr.Markdown(
                "## Import student data\n"
                "Upload CSV or Excel files (.csv, .xlsx, .xls, .xlsm) with the student "
                "impact columns used by this project: `Major_Category`, `Year_of_Study`, "
                "`Pre_Semester_GPA`, `Post_Semester_GPA`, `Weekly_GenAI_Hours`, "
                "`Primary_Use_Case`, `Prompt_Engineering_Skill`, `Tool_Diversity`, "
                "`Paid_Subscription`, `Traditional_Study_Hours`, "
                "`Perceived_AI_Dependency`, `Institutional_Policy`, "
                "`Anxiety_Level_During_Exams`, `Skill_Retention_Score`, and "
                "`Burnout_Risk_Level`. `Student_ID` is optional and is never stored. "
                "Duplicate and incomplete rows are removed, and numeric ranges and "
                "burnout labels are validated. Imported records are saved in a local "
                "SQLite database; use anonymized data only."
            )
            with gr.Row():
                uploaded_file = gr.File(
                    label="Student dataset",
                    file_types=[".csv", ".xlsx", ".xls", ".xlsm"],
                    type="filepath",
                )
                import_button = gr.Button("Import file", variant="primary")
                refresh_button = gr.Button("Refresh dashboard")
            import_status = gr.Markdown()
            upload_batches, upload_counts, upload_averages, upload_insights = (
                get_upload_dashboard()
            )
            batch_table = gr.Dataframe(
                value=upload_batches,
                interactive=False,
                label="Recent imports",
            )
            upload_insights_output = gr.Markdown(value=upload_insights)
            with gr.Row():
                risk_chart = gr.BarPlot(
                    value=upload_counts,
                    x="Burnout risk",
                    y="Student count",
                    title="Imported records by burnout-risk level",
                )
                hours_chart = gr.BarPlot(
                    value=upload_averages,
                    x="Burnout risk",
                    y="Average GenAI hours",
                    title="Average weekly GenAI use by risk level",
                )
            gpa_chart = gr.BarPlot(
                value=upload_averages,
                x="Burnout risk",
                y="Average GPA change",
                title="Average GPA change by risk level",
            )
            import_button.click(
                import_uploaded_file,
                inputs=[uploaded_file],
                outputs=[
                    import_status,
                    batch_table,
                    risk_chart,
                    hours_chart,
                    gpa_chart,
                    upload_insights_output,
                ],
            )
            refresh_button.click(
                refresh_upload_dashboard,
                outputs=[
                    batch_table,
                    risk_chart,
                    hours_chart,
                    gpa_chart,
                    upload_insights_output,
                ],
            )

        with gr.Tab("Make a prediction"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_id="input-panel"):
                    gr.Markdown('<div class="section-title">Student profile</div>')
                    gr.Markdown(
                        '<div class="section-hint">Academic background and AI-use patterns</div>'
                    )
                    with gr.Row():
                        major = gr.Dropdown(
                            CHOICES["Major_Category"], value="STEM", label="Major category"
                        )
                        year = gr.Dropdown(
                            CHOICES["Year_of_Study"],
                            value="Sophomore",
                            label="Year of study",
                        )
                    with gr.Row():
                        use_case = gr.Dropdown(
                            CHOICES["Primary_Use_Case"],
                            value="Ideation",
                            label="Primary AI use case",
                        )
                        skill = gr.Dropdown(
                            CHOICES["Prompt_Engineering_Skill"],
                            value="Intermediate",
                            label="Prompt engineering skill",
                        )
                    policy = gr.Dropdown(
                        CHOICES["Institutional_Policy"],
                        value="Allowed_With_Citation",
                        label="Institutional AI policy",
                    )
                    with gr.Row():
                        paid = gr.Checkbox(value=False, label="Paid AI subscription")
                        tools = gr.Slider(
                            1, 5, value=3, step=1, label="AI tools used (1-5)"
                        )
                    gr.Markdown('<div class="section-title">Study & well-being</div>')
                    gr.Markdown(
                        '<div class="section-hint">Hours are per week; GPA is 0-4; '
                        "dependency and anxiety are 1-10; retention is 10-100. "
                        "These are not clinical assessments.</div>"
                    )
                    with gr.Row():
                        hours = gr.Slider(
                            0, 40, value=8, step=0.5, label="GenAI hrs/wk"
                        )
                        trad = gr.Slider(
                            1, 36, value=11, step=0.5, label="Study hrs/wk"
                        )
                    with gr.Row():
                        pre_gpa = gr.Slider(
                            0, 4, value=3.2, step=0.01, label="GPA before"
                        )
                        post_gpa = gr.Slider(
                            0, 4, value=3.3, step=0.01, label="GPA after"
                        )
                    with gr.Row():
                        dependency = gr.Slider(
                            1, 10, value=4, step=1, label="Dependency"
                        )
                        anxiety = gr.Slider(
                            1, 10, value=4, step=1, label="Exam anxiety"
                        )
                    retention = gr.Slider(
                        10, 100, value=76, step=1, label="Skill retention"
                    )
                    btn = gr.Button(
                        "Predict burnout risk",
                        variant="primary",
                        elem_id="predict-button",
                    )
                with gr.Column(scale=1, elem_id="result-panel"):
                    gr.Markdown('<div class="section-title">Prediction</div>')
                    gr.Markdown(
                        '<div class="section-hint">Your result will appear here when you click the button.</div>'
                    )
                    out_label = gr.Label(
                        label="Risk category and nearest-neighbor vote",
                        num_top_classes=3,
                    )
                    out_text = gr.Markdown(
                        '<div class="prediction-note">Adjust the profile on the left, '
                        "then select <strong>Predict burnout risk</strong> to see the result.</div>"
                    )
                    gr.Markdown(
                        f"**Model:** Euclidean KNN (Configuration B) · top {SELECTED_N} attributes · K={SELECTED_K}  \n"
                        f"**Held-out accuracy:** {TEST_ACCURACY:.1%}"
                    )
                    gr.Markdown(
                        '<div class="fine-print">For a class-project demonstration only. '
                        "This model is not a clinical assessment and should not be used "
                        "to make decisions about an individual student.</div>"
                    )
            btn.click(
                predict,
                [
                    major,
                    year,
                    pre_gpa,
                    hours,
                    use_case,
                    skill,
                    tools,
                    paid,
                    trad,
                    dependency,
                    policy,
                    anxiety,
                    post_gpa,
                    retention,
                ],
                [out_label, out_text],
            )

        with gr.Tab("Model performance"):
            gr.Markdown("## Attribute-count (N) and K selection")
            gr.Markdown(
                f"Encoded predictors are ranked by Information Gain on the training "
                f"partition. For each top-N set, K is scanned over {K_VALUES} with "
                f"{CV_FOLDS}-fold stratified CV, and the (N, K) pair with the highest "
                "cross-validated macro F1 is selected (accuracy breaks ties). The "
                "selected pair is marked below. The held-out test set is reserved for "
                "the final evaluation of that selected model."
            )
            gr.Dataframe(value=bundle["cv_results"], interactive=False)

            gr.Markdown("## Held-out evaluation")
            gr.Markdown(
                f"Stratified 80:20 split; top {SELECTED_N} attributes; selected K={SELECTED_K}; accuracy "
                f"{TEST_ACCURACY:.1%}. Precision, recall, and F1 are shown "
                "per class and macro-averaged."
            )
            with gr.Row():
                gr.Dataframe(value=bundle["metrics"], interactive=False, label="Metrics")
                gr.Dataframe(
                    value=bundle["distribution"],
                    interactive=False,
                    label="Actual vs predicted class counts",
                )
            gr.Dataframe(
                value=bundle["confusion_table"],
                interactive=False,
                label="Confusion matrix (rows = actual, columns = predicted)",
            )

            gr.Markdown("## Decision Tree explanation")
            gr.Markdown(
                bundle["tree_root_summary"]
                + "\n\nThe ranking reports each attribute's best single split "
                "(one-hot categorical attributes use the best individual dummy split), "
                "plus its normalized importance accumulated across the fitted tree."
            )
            gr.Dataframe(
                value=bundle["tree_ranking"],
                interactive=False,
                label="Attribute information-gain ranking",
            )


if __name__ == "__main__":
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except ImportError:
        in_colab = False
    demo.launch(share=in_colab, theme=APP_THEME, css=APP_CSS)
