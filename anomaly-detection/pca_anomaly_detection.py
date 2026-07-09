# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA


# class PCAAnomalyDetector:

#     def __init__(self, variance_threshold=0.95):
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=variance_threshold)

#     def fit(self, X_train):
#         X_scaled = self.scaler.fit_transform(X_train)
#         self.pca.fit(X_scaled)

#     def score(self, X):

#         X_scaled = self.scaler.transform(X)

#         # Compress
#         X_pca = self.pca.transform(X_scaled)

#         # Reconstruct
#         X_reconstructed = self.pca.inverse_transform(X_pca)

#         # Per-feature reconstruction error
#         feature_errors = np.abs(
#             X_scaled - X_reconstructed
#         )

#         # Row-level anomaly score
#         row_errors = np.mean(
#             (X_scaled - X_reconstructed) ** 2,
#             axis=1
#         )

#         return row_errors, feature_errors

#     @staticmethod
#     def normalize_scores(errors):

#         min_err = errors.min()
#         max_err = errors.max()

#         scores = (
#             (errors - min_err)
#             / (max_err - min_err + 1e-10)
#         ) * 100

#         return scores

#     @staticmethod
#     def get_top_features(
#         feature_errors,
#         feature_names,
#         top_k=7
#     ):

#         results = []

#         for row in feature_errors:

#             idx = np.argsort(row)[::-1][:top_k]

#             results.append(
#                 [feature_names[i] for i in idx]
#             )

#         return results


# # ------------------------------------------
# # Main
# # ------------------------------------------

# df = pd.read_csv("sample_data.csv")

# df["Time"] = pd.to_datetime(df["Time"])

# feature_cols = [
#     c for c in df.columns
#     if c != "Time"
# ]

# # Fill missing values
# df[feature_cols] = (
#     df[feature_cols]
#     .ffill()
#     .bfill()
# )

# # Normal training period
# train_mask = (
#     (df["Time"] >= "2004-01-01")
#     &
#     (df["Time"] <= "2004-01-05 23:59")
# )

# X_train = df.loc[
#     train_mask,
#     feature_cols
# ].values

# X_all = df[feature_cols].values

# # Train PCA
# model = PCAAnomalyDetector(
#     variance_threshold=0.95
# )

# model.fit(X_train)

# # Score all rows
# row_errors, feature_errors = model.score(X_all)

# # Normalize to 0-100
# scores = model.normalize_scores(
#     row_errors
# )

# # Root-cause features
# top_features = model.get_top_features(
#     feature_errors,
#     feature_cols,
#     top_k=7
# )

# # Output
# df["Abnormality_score"] = scores

# for i in range(7):
#     df[f"top_feature_{i+1}"] = [
#         x[i] if len(x) > i else ""
#         for x in top_features
#     ]

# df.to_csv(
#     "pca_anomaly_results.csv",
#     index=False
# )

# print("Done.")

"""
PCA-Based Multivariate Time Series Anomaly Detection

Outputs:
    - Abnormality_score (0-100)
    - top_feature_1 ... top_feature_7

Usage:
    python pca_anomaly_detection.py \
        --input data.csv \
        --output results.csv \
        --train-start "1/1/2004 0:00" \
        --train-end "1/5/2004 23:59" \
        --timestamp-col "Time"
"""

import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------

class DataProcessor:

    def __init__(self, timestamp_col: Optional[str] = None):
        self.timestamp_col = timestamp_col
        self.feature_cols = []

    def load(self, path: str) -> pd.DataFrame:

        df = pd.read_csv(path)

        if (
            self.timestamp_col
            and self.timestamp_col in df.columns
        ):
            df[self.timestamp_col] = pd.to_datetime(
                df[self.timestamp_col]
            )

            df = (
                df.sort_values(self.timestamp_col)
                .reset_index(drop=True)
            )

        return df

    def get_feature_columns(
        self,
        df: pd.DataFrame
    ) -> list:

        exclude = (
            {self.timestamp_col}
            if self.timestamp_col
            else set()
        )

        cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not cols:
            raise ValueError(
                "No numeric feature columns found."
            )

        self.feature_cols = cols
        return cols

    def clean(
        self,
        df: pd.DataFrame,
        cols: list
    ) -> pd.DataFrame:

        df[cols] = df[cols].ffill().bfill()

        stds = df[cols].std()

        constant_cols = (
            stds[stds == 0]
            .index
            .tolist()
        )

        if constant_cols:

            df = df.drop(
                columns=constant_cols
            )

            self.feature_cols = [
                c for c in self.feature_cols
                if c not in constant_cols
            ]

        return df


# ---------------------------------------------------------------------
# PCA Model
# ---------------------------------------------------------------------

class PCAAnomalyModel:

    def __init__(
        self,
        variance_threshold: float = 0.95
    ):

        self.scaler = StandardScaler()

        self.pca = PCA(
            n_components=variance_threshold,
            svd_solver="full"
        )

    def fit(
        self,
        X_train: np.ndarray
    ):

        X_scaled = self.scaler.fit_transform(
            X_train
        )

        self.pca.fit(X_scaled)

    def reconstruction_errors(
        self,
        X: np.ndarray
    ):

        X_scaled = self.scaler.transform(X)

        X_pca = self.pca.transform(
            X_scaled
        )

        X_reconstructed = (
            self.pca.inverse_transform(
                X_pca
            )
        )

        feature_errors = np.square(
            X_scaled - X_reconstructed
        )

        row_errors = np.mean(
            feature_errors,
            axis=1
        )

        return (
            row_errors,
            feature_errors
        )


# ---------------------------------------------------------------------
# Score Smoothing
# ---------------------------------------------------------------------

class ScoreSmoother:

    def __init__(
        self,
        window: int = 5
    ):
        self.window = window

    def smooth(
        self,
        scores: np.ndarray
    ) -> np.ndarray:

        if self.window <= 1:
            return scores.copy()

        return (
            pd.Series(scores)
            .rolling(
                window=self.window,
                center=True,
                min_periods=1
            )
            .mean()
            .values
        )


# ---------------------------------------------------------------------
# Score Normalization
# ---------------------------------------------------------------------

class ScoreNormalizer:

    def fit(
        self,
        train_scores: np.ndarray
    ):

        self.normal_ceiling = float(
            np.percentile(
                train_scores,
                99
            )
        )

    def normalize(
        self,
        raw_scores: np.ndarray
    ) -> np.ndarray:

        n = len(raw_scores)

        scores = np.zeros(
            n,
            dtype=float
        )

        normal_mask = (
            raw_scores
            <= self.normal_ceiling
        )

        anomaly_mask = ~normal_mask

        if normal_mask.any():

            lo = raw_scores[
                normal_mask
            ].min()

            hi = self.normal_ceiling

            span = (
                hi - lo
                if hi > lo
                else 1.0
            )

            scores[normal_mask] = (
                (
                    raw_scores[
                        normal_mask
                    ] - lo
                )
                / span
            ) * 9.0

        if anomaly_mask.any():

            anom = raw_scores[
                anomaly_mask
            ]

            ranks = np.argsort(
                np.argsort(anom)
            )

            m = len(anom)

            scores[anomaly_mask] = (
                9.0
                + (
                    ranks
                    / max(m - 1, 1)
                )
                * 91.0
            )

        return np.clip(
            scores,
            0.0,
            100.0
        )


# ---------------------------------------------------------------------
# Feature Attribution
# ---------------------------------------------------------------------

class FeatureAttributor:

    def __init__(
        self,
        feature_cols: list
    ):
        self.feature_cols = feature_cols

    def top_features(
        self,
        feature_errors: np.ndarray,
        top_k: int = 7
    ):

        results = []

        for row in feature_errors:

            idx = np.argsort(
                row
            )[::-1]

            names = [
                self.feature_cols[i]
                for i in idx[:top_k]
                if row[i] > 0
            ]

            results.append(names)

        return results


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def add_output_columns(
    df,
    scores,
    top_features
):

    df = df.copy()

    df["Abnormality_score"] = np.round(
        scores,
        4
    )

    for k in range(1, 8):

        df[f"top_feature_{k}"] = [

            feats[k - 1]
            if len(feats) >= k
            else ""

            for feats in top_features
        ]

    return df


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def run(
    input_csv_path,
    output_csv_path,
    train_start,
    train_end,
    timestamp_col="Time",
    smooth_window=5
):

    processor = DataProcessor(
        timestamp_col
    )

    df = processor.load(
        input_csv_path
    )

    feature_cols = (
        processor.get_feature_columns(df)
    )

    df = processor.clean(
        df,
        feature_cols
    )

    feature_cols = (
        processor.feature_cols
    )

    X_full = (
        df[feature_cols]
        .values
        .astype(float)
    )

    if (
        timestamp_col
        and timestamp_col in df.columns
    ):

        ts = df[timestamp_col]

        train_mask = (
            (ts >= pd.Timestamp(train_start))
            &
            (ts <= pd.Timestamp(train_end))
        )

    else:

        train_mask = pd.Series(
            [False] * len(df)
        )

        train_mask.iloc[:120] = True

    n_train = int(
        train_mask.sum()
    )

    if n_train < 72:

        raise ValueError(
            f"Training period contains only "
            f"{n_train} rows."
        )

    X_train = X_full[
        train_mask
    ]

    model = PCAAnomalyModel(
        variance_threshold=0.95
    )

    model.fit(X_train)

    (
        raw_errors,
        feature_errors
    ) = model.reconstruction_errors(
        X_full
    )

    smoother = ScoreSmoother(
        window=smooth_window
    )

    smoothed_errors = (
        smoother.smooth(
            raw_errors
        )
    )

    normalizer = (
        ScoreNormalizer()
    )

    normalizer.fit(
        smoothed_errors[
            train_mask
        ]
    )

    scores = (
        normalizer.normalize(
            smoothed_errors
        )
    )

    train_scores = scores[
        train_mask
    ]

    print(
        f"Training Mean: "
        f"{train_scores.mean():.2f}"
    )

    print(
        f"Training Max: "
        f"{train_scores.max():.2f}"
    )

    attributor = (
        FeatureAttributor(
            feature_cols
        )
    )

    top_feats = (
        attributor.top_features(
            feature_errors
        )
    )

    df_out = (
        add_output_columns(
            df,
            scores,
            top_feats
        )
    )

    df_out.to_csv(
        output_csv_path,
        index=False
    )

    print(
        f"Saved results to "
        f"{output_csv_path}"
    )

    return df_out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--train-start",
        required=True
    )

    parser.add_argument(
        "--train-end",
        required=True
    )

    parser.add_argument(
        "--timestamp-col",
        default="Time"
    )

    parser.add_argument(
        "--smooth-window",
        default=5,
        type=int
    )

    args = parser.parse_args()

    run(
        input_csv_path=args.input,
        output_csv_path=args.output,
        train_start=args.train_start,
        train_end=args.train_end,
        timestamp_col=args.timestamp_col,
        smooth_window=args.smooth_window
    )