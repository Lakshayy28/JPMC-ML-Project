from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    average_precision: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    train_rows: int
    test_rows: int


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = list(features.select_dtypes(include=["number", "bool"]).columns)
    categorical_features = [column for column in features.columns if column not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _metric_or_none(metric_fn, y_true, y_score_or_pred):
    try:
        return float(metric_fn(y_true, y_score_or_pred))
    except ValueError:
        return None


def train_binary_models(
    feature_frame: pd.DataFrame,
    *,
    target_column: str = "label",
    id_columns: tuple[str, ...] = (),
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, dict[str, float | int | None]]:
    if feature_frame[target_column].nunique() < 2:
        raise ValueError("At least two label classes are required to train a binary model.")

    model_frame = feature_frame.drop(columns=[*id_columns])
    y = model_frame.pop(target_column).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        model_frame,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 and y.value_counts().min() > 1 else None,
    )

    preprocessor = _build_preprocessor(model_frame)
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }

    results: dict[str, dict[str, float | int | None]] = {}
    for model_name, estimator in estimators.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        result = ModelResult(
            model_name=model_name,
            average_precision=_metric_or_none(average_precision_score, y_test, probabilities),
            precision=_metric_or_none(lambda truth, pred: precision_score(truth, pred, zero_division=0), y_test, predictions),
            recall=_metric_or_none(lambda truth, pred: recall_score(truth, pred, zero_division=0), y_test, predictions),
            f1=_metric_or_none(lambda truth, pred: f1_score(truth, pred, zero_division=0), y_test, predictions),
            roc_auc=_metric_or_none(roc_auc_score, y_test, probabilities),
            train_rows=int(len(x_train)),
            test_rows=int(len(x_test)),
        )
        results[model_name] = asdict(result)

    return results


def train_all_baselines(
    feature_sets: dict[str, pd.DataFrame],
    *,
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    return {
        "transaction": train_binary_models(
            feature_sets["transaction"],
            id_columns=("transaction_id", "source_account_id", "destination_account_id", "label_source"),
            random_state=random_state,
            test_size=test_size,
        ),
        "party": train_binary_models(
            feature_sets["party"],
            id_columns=("party_id",),
            random_state=random_state,
            test_size=test_size,
        ),
    }
