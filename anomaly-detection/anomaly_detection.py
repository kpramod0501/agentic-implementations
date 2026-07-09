# import argparse
# import warnings
# from typing import Optional

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler

# warnings.filterwarnings("ignore")

# # Data loading and preprocessing

# class DataProcessor:
#     """Loads, validates, and cleans the input CSV."""

#     def __init__(self, timestamp_col: Optional[str] = None) -> None:
#         self.timestamp_col = timestamp_col
#         self.feature_cols: list[str] = []

#     def load(self, path: str) -> pd.DataFrame:
#         """Load CSV and parse timestamps."""
#         df = pd.read_csv(path)
#         print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns from '{path}'")

#         if self.timestamp_col and self.timestamp_col in df.columns:
#             df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
#             df = df.sort_values(self.timestamp_col).reset_index(drop=True)

#         return df

#     def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
#         """Return numeric columns, excluding the timestamp column."""
#         exclude = {self.timestamp_col} if self.timestamp_col else set()
#         cols = [
#             c for c in df.columns
#             if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
#         ]
#         if not cols:
#             raise ValueError("No numeric feature columns found in the dataset.")
#         self.feature_cols = cols
#         print(f"[INFO] Feature columns ({len(cols)}): {cols}")
#         return cols

#     def clean(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#         """Forward-fill then back-fill missing values; drop constant columns."""
#         df[cols] = df[cols].ffill().bfill()

#         # Drop zero-variance columns (they contribute nothing)
#         stds = df[cols].std()
#         constant = stds[stds == 0].index.tolist()
#         if constant:
#             print(f"[WARN] Dropping constant columns: {constant}")
#             df = df.drop(columns=constant)
#             self.feature_cols = [c for c in self.feature_cols if c not in constant]

#         return df


# # Model Training

# class AnomalyModel:
#     """Isolation Forest trained on the normal period."""

#     def __init__(self, contamination: float = 0.01, n_estimators: int = 200,
#                  random_state: int = 42) -> None:
#         self.scaler = StandardScaler()
#         self.model = IsolationForest(
#             contamination=contamination,
#             n_estimators=n_estimators,
#             random_state=random_state,
#             n_jobs=-1,
#         )

#     def fit(self, X_train: np.ndarray) -> None:
#         """Fit scaler and Isolation Forest on normal-period data."""
#         X_scaled = self.scaler.fit_transform(X_train)
#         self.model.fit(X_scaled)
#         print(f"[INFO] Model trained on {len(X_train)} normal samples.")

#     def raw_scores(self, X: np.ndarray) -> np.ndarray:
#         """
#         Return raw anomaly scores (higher = more anomalous).
#         Isolation Forest's decision_function returns negative values for
#         anomalies; we negate so larger → more anomalous.
#         """
#         X_scaled = self.scaler.transform(X)
#         return -self.model.decision_function(X_scaled)   # shape: (n,)


# # Score normalisation  (0–100)

# class ScoreNormalizer:
#     """
#     Converts raw model scores to a 0–100 scale calibrated so that
#     training-period scores stay below 10 on average.

#     Strategy
#     --------
#     1. Fit a "normal ceiling" = 99th-percentile raw score from the training
#        period.  Any raw score at or below this ceiling maps to [0, 9].
#     2. Scores above the ceiling are mapped to (9, 100] via percentile
#        ranking within the above-ceiling portion.
#     3. Tiny Gaussian noise prevents perfectly-flat regions.
#     """

#     def fit(self, raw_train: np.ndarray) -> None:
#         """Learn the normal ceiling from training-period raw scores."""
#         self.normal_ceiling = float(np.percentile(raw_train, 99))
#         print(f"[INFO] Normal ceiling (99th pct of training): {self.normal_ceiling:.4f}")

#     def normalize(self, raw: np.ndarray) -> np.ndarray:
#         n = len(raw)
#         scores = np.zeros(n, dtype=float)
#         rng = np.random.default_rng(0)

#         normal_mask = raw <= self.normal_ceiling
#         anomaly_mask = ~normal_mask

#         # --- normal region: map linearly to [0, 9] ---
#         if normal_mask.any():
#             lo = raw[normal_mask].min()
#             hi = self.normal_ceiling
#             span = hi - lo if hi > lo else 1.0
#             scores[normal_mask] = ((raw[normal_mask] - lo) / span) * 9.0

#         # --- anomaly region: percentile-rank to (9, 100] ---
#         if anomaly_mask.any():
#             anom_raw = raw[anomaly_mask]
#             ranks = np.argsort(np.argsort(anom_raw))
#             m = len(anom_raw)
#             scores[anomaly_mask] = 9.0 + (ranks / max(m - 1, 1)) * 91.0

#         # tiny noise to avoid exact duplicates
#         scores += rng.normal(0, 0.05, n)
#         return np.clip(scores, 0.0, 100.0)


# # Feature attribution

# class FeatureAttributor:
#     """
#     Leave-one-out attribution: for each row, measure how much the anomaly
#     score drops when each feature is replaced by its training-period mean.
#     Larger drop → higher contribution.
#     """

#     MIN_CONTRIBUTION_PCT = 0.01   # 1 % threshold

#     def __init__(self, model: AnomalyModel, feature_cols: list[str],
#                  train_means: np.ndarray) -> None:
#         self.model = model
#         self.feature_cols = feature_cols
#         self.train_means = train_means          # shape: (n_features,)

#     def top_features(self, X: np.ndarray, raw_scores: np.ndarray,
#                      top_k: int = 7) -> list[list[str]]:
#         """
#         Returns a list (one entry per row) of up to top_k feature names.
#         Rows with very low raw scores (clearly normal) get empty lists fast.
#         """
#         n_rows, n_feat = X.shape
#         results: list[list[str]] = []

#         print(f"[INFO] Computing feature attribution for {n_rows} rows …")

#         for i in range(n_rows):
#             row = X[i].copy()
#             base_score = raw_scores[i]

#             contributions = np.zeros(n_feat)
#             for j in range(n_feat):
#                 masked = row.copy()
#                 masked[j] = self.train_means[j]          # replace with mean
#                 masked_score = self.model.raw_scores(masked.reshape(1, -1))[0]
#                 contributions[j] = base_score - masked_score  # drop = contribution

#             total = np.sum(np.abs(contributions))
#             if total == 0:
#                 results.append([])
#                 continue

#             pct = contributions / total
#             # keep only features with >1% contribution AND positive contribution
#             eligible = [
#                 (self.feature_cols[j], contributions[j])
#                 for j in range(n_feat)
#                 if contributions[j] > 0 and abs(pct[j]) > self.MIN_CONTRIBUTION_PCT
#             ]
#             # sort by contribution desc, then alphabetically for ties
#             eligible.sort(key=lambda x: (-x[1], x[0]))
#             results.append([name for name, _ in eligible[:top_k]])

#         return results


# # Output Writing

# def add_output_columns(df: pd.DataFrame, scores: np.ndarray,
#                        top_features: list[list[str]]) -> pd.DataFrame:
#     """Append Abnormality_score and top_feature_1..7 to the dataframe."""
#     df = df.copy()
#     df["Abnormality_score"] = np.round(scores, 4)

#     for k in range(1, 8):
#         col_name = f"top_feature_{k}"
#         df[col_name] = [
#             feats[k - 1] if len(feats) >= k else ""
#             for feats in top_features
#         ]

#     return df


# # Main Pipeline

# def run(
#     input_csv_path: str,
#     output_csv_path: str,
#     train_start: str = "1/1/2004 0:00",
#     train_end: str = "1/5/2004 23:59",
#     timestamp_col: str = "timestamp",
# ) -> pd.DataFrame:
#     """
#     End-to-end anomaly detection pipeline.

#     Parameters
#     ----------
#     input_csv_path  : Path to the input CSV file.
#     output_csv_path : Where to write the enriched CSV.
#     train_start     : Start of the normal (training) period.
#     train_end       : End of the normal (training) period.
#     timestamp_col   : Name of the timestamp column (or None).

#     Returns
#     -------
#     pd.DataFrame with 8 new columns appended.
#     """
#     np.random.seed(42)

#     # 1. Load & clean ───────────────────────────────────────────────
#     processor = DataProcessor(timestamp_col=timestamp_col)
#     df = processor.load(input_csv_path)
#     feature_cols = processor.get_feature_columns(df)
#     df = processor.clean(df, feature_cols)
#     feature_cols = processor.feature_cols       # may be updated after dropping constants

#     X_full = df[feature_cols].values.astype(float)

#     # 2. Split training window ───────────────────────────────────────
#     if timestamp_col and timestamp_col in df.columns:
#         ts = df[timestamp_col]
#         train_mask = (ts >= pd.Timestamp(train_start)) & (ts <= pd.Timestamp(train_end))
#     else:
#         # Fallback: use first 120 rows as training
#         print("[WARN] No timestamp column; using first 120 rows as training data.")
#         train_mask = pd.Series([False] * len(df))
#         train_mask.iloc[:120] = True

#     n_train = train_mask.sum()
#     if n_train < 72:
#         raise ValueError(
#             f"Training period has only {n_train} rows; minimum 72 required."
#         )
#     print(f"[INFO] Training rows: {n_train}  |  Analysis rows: {len(df)}")

#     X_train = X_full[train_mask]
#     train_means = X_train.mean(axis=0)

#     # 3. Train model ─────────────────────────────────────────────────
#     model = AnomalyModel()
#     model.fit(X_train)

#     # 4. Score all rows ──────────────────────────────────────────────
#     raw = model.raw_scores(X_full)

#     normalizer = ScoreNormalizer()
#     normalizer.fit(raw[train_mask])          # calibrate from training period
#     scores = normalizer.normalize(raw)

#     # Quick sanity-check on training period
#     train_scores = scores[train_mask]
#     print(
#         f"[VALIDATION] Training-period scores  "
#         f"mean={train_scores.mean():.2f}  max={train_scores.max():.2f}"
#     )
#     if train_scores.mean() >= 10:
#         print("[WARN] Training-period mean score ≥ 10; check training window.")

#     # 5. Feature attribution ─────────────────────────────────────────
#     attributor = FeatureAttributor(model, feature_cols, train_means)
#     top_feats = attributor.top_features(X_full, raw)

#     # 6. Build output ────────────────────────────────────────────────
#     df_out = add_output_columns(df, scores, top_feats)
#     df_out.to_csv(output_csv_path, index=False)
#     print(f"[INFO] Results written to '{output_csv_path}'")

#     return df_out



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Multivariate Time Series Anomaly Detection")
#     parser.add_argument("--input",         required=True,  help="Path to input CSV")
#     parser.add_argument("--output",        required=True,  help="Path for output CSV")
#     parser.add_argument("--train-start",   default="1/1/2004 0:00",  help="Training start datetime")
#     parser.add_argument("--train-end",     default="1/5/2004 23:59", help="Training end datetime")
#     parser.add_argument("--timestamp-col", default="timestamp", help="Name of timestamp column")

#     args = parser.parse_args()

#     run(
#         input_csv_path=args.input,
#         output_csv_path=args.output,
#         train_start=args.train_start,
#         train_end=args.train_end,
#         timestamp_col=args.timestamp_col,
#     )

"""
Multivariate Time Series Anomaly Detection using Isolation Forest.

Detects anomalies in sensor/IoT time series data by training on a known
normal period, scoring all rows, and identifying which features drove
each anomaly. Outputs the original CSV enriched with an Abnormality_score
(0-100) and the top 7 contributing feature names.

Usage:
    python anomaly_detection.py \
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

class DataProcessor:
    """Handles loading, validation, and cleaning of the input CSV.

    Parses timestamps, identifies numeric feature columns, fills missing
    values, and removes columns that carry no information (zero variance).
    """

    def __init__(self, timestamp_col: Optional[str] = None) -> None:
        """Initialise with the name of the timestamp column (if any).

        Args:
            timestamp_col: Name of the datetime column in the CSV.
                           Pass None if the file has no timestamp.
        """
        self.timestamp_col = timestamp_col
        self.feature_cols: list[str] = []

    def load(self, path: str) -> pd.DataFrame:
        """Load the CSV, parse timestamps, and sort by time.

        Args:
            path: Path to the input CSV file.

        Returns:
            A DataFrame sorted by timestamp (if present).
        """
        df = pd.read_csv(path)
        n_cols = df.shape[1]
        print(f"[INFO] Loaded {len(df)} rows, {n_cols} columns from '{path}'")

        if self.timestamp_col and self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Find all numeric columns, excluding the timestamp column.

        Args:
            df: The loaded DataFrame.

        Returns:
            List of column names to use as model features.

        Raises:
            ValueError: If no numeric columns are found.
        """
        exclude = {self.timestamp_col} if self.timestamp_col else set()
        cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cols:
            raise ValueError(
                "No numeric feature columns found in the dataset."
            )

        self.feature_cols = cols
        print(f"[INFO] Feature columns ({len(cols)}): {cols}")
        return cols

    def clean(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Fill missing values and drop constant (zero-variance) columns.

        Missing values are forward-filled first, then back-filled so that
        leading NaNs are also handled. Columns that never change are
        dropped because they add no discriminating information.

        Args:
            df:   The DataFrame to clean.
            cols: The feature column names to inspect.

        Returns:
            Cleaned DataFrame (constant columns removed in-place on self).
        """
        df[cols] = df[cols].ffill().bfill()

        stds = df[cols].std()
        constant_cols = stds[stds == 0].index.tolist()
        if constant_cols:
            print(f"[WARN] Dropping constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)
            self.feature_cols = [
                c for c in self.feature_cols if c not in constant_cols
            ]

        return df


# ---------------------------------------------------------------------------
# Anomaly model (Isolation Forest)
# ---------------------------------------------------------------------------

class AnomalyModel:
    """Isolation Forest wrapper trained exclusively on normal-period data.

    Isolation Forest detects anomalies by measuring how easy it is to
    isolate a data point with random splits. Normal points need many
    splits; outliers are isolated quickly. The decision_function output
    is negated so that higher scores always mean more anomalous.
    """

    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        """Set up the scaler and Isolation Forest model.

        Args:
            contamination: Expected fraction of outliers in the training
                           data. Keep low (0.01) since training on normal
                           data only.
            n_estimators:  Number of isolation trees. More trees = more
                           stable scores, slower training.
            random_state:  Seed for reproducibility.
        """
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray) -> None:
        """Standardise features and fit Isolation Forest on normal data.

        Args:
            X_train: 2-D array of shape (n_samples, n_features) containing
                     only the normal (training) period rows.
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        print(f"[INFO] Model trained on {len(X_train)} normal samples.")

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores for every row (higher = more anomalous).

        Isolation Forest's decision_function gives positive values for
        normal points and negative for anomalies. We negate it so the
        convention is consistent: higher number = more suspicious.

        Args:
            X: 2-D array of shape (n_samples, n_features).

        Returns:
            1-D array of raw anomaly scores, shape (n_samples,).
        """
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)


# ---------------------------------------------------------------------------
# Score smoothing
# ---------------------------------------------------------------------------

class ScoreSmoother:
    """Applies a rolling average to remove sudden score jumps.

    Sensor anomalies develop gradually, so a single bad reading should
    not produce a cliff-edge score change. A short rolling window (3
    time steps by default) blends each score with its neighbours without
    significantly delaying detection.
    """

    def __init__(self, window: int = 3) -> None:
        """Initialise with the desired smoothing window size.

        Args:
            window: Number of consecutive time points to average.
                    Use 1 to disable smoothing.
        """
        self.window = window

    def smooth(self, raw: np.ndarray) -> np.ndarray:
        """Apply a centred rolling mean, keeping edge values intact.

        Args:
            raw: 1-D array of raw scores in time order.

        Returns:
            Smoothed 1-D array of the same length.
        """
        if self.window <= 1:
            return raw.copy()

        series = pd.Series(raw)
        smoothed = (
            series
            .rolling(window=self.window, center=True, min_periods=1)
            .mean()
            .values
        )
        return smoothed


# ---------------------------------------------------------------------------
# Score normalisation (0-100)
# ---------------------------------------------------------------------------

class ScoreNormalizer:
    """Converts smoothed model scores to a human-readable 0-100 scale.

    The calibration anchors training-period scores below 10 by treating
    the 99th-percentile training score as a 'normal ceiling'. Everything
    below that ceiling maps to 0-9; everything above maps to 9-100.
    This means the training window reliably validates at mean < 10.
    """

    def fit(self, train_scores: np.ndarray) -> None:
        """Learn the normal ceiling from training-period scores.

        Args:
            train_scores: 1-D array of smoothed raw scores for the
                          training period only.
        """
        self.normal_ceiling = float(np.percentile(train_scores, 99))
        print(
            f"[INFO] Normal ceiling "
            f"(99th pct of training): {self.normal_ceiling:.4f}"
        )

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """Map raw scores to [0, 100] using the fitted ceiling.

        Scores at or below the ceiling are mapped linearly to [0, 9].
        Scores above the ceiling are percentile-ranked among themselves
        and mapped to (9, 100]. Tiny Gaussian noise prevents ties.

        Args:
            raw: 1-D array of smoothed raw scores for all rows.

        Returns:
            1-D float array of normalised scores in [0, 100].
        """
        n = len(raw)
        scores = np.zeros(n, dtype=float)
        rng = np.random.default_rng(0)

        normal_mask = raw <= self.normal_ceiling
        anomaly_mask = ~normal_mask

        # Normal region: linear stretch to [0, 9]
        if normal_mask.any():
            lo = raw[normal_mask].min()
            hi = self.normal_ceiling
            span = hi - lo if hi > lo else 1.0
            scores[normal_mask] = ((raw[normal_mask] - lo) / span) * 9.0

        # Anomaly region: percentile rank within anomalies -> (9, 100]
        if anomaly_mask.any():
            anom_raw = raw[anomaly_mask]
            ranks = np.argsort(np.argsort(anom_raw))
            m = len(anom_raw)
            scores[anomaly_mask] = 9.0 + (ranks / max(m - 1, 1)) * 91.0

        # Small noise so identical raw scores get slightly different outputs
        scores += rng.normal(0, 0.05, n)
        return np.clip(scores, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Feature attribution
# ---------------------------------------------------------------------------

class FeatureAttributor:
    """Identifies which features drove each anomaly using leave-one-out.

    For every row, each feature is temporarily replaced with its
    training-period mean. The drop in anomaly score tells us how much
    that feature contributed. Larger drop = bigger culprit.
    """

    # Features contributing less than 1% of total are ignored
    MIN_CONTRIBUTION_PCT = 0.01

    def __init__(
        self,
        model: AnomalyModel,
        feature_cols: list[str],
        train_means: np.ndarray,
    ) -> None:
        """Store the trained model, feature names, and training means.

        Args:
            model:        A fitted AnomalyModel instance.
            feature_cols: Ordered list of feature column names.
            train_means:  Per-feature means computed from training data,
                          shape (n_features,).
        """
        self.model = model
        self.feature_cols = feature_cols
        self.train_means = train_means

    def top_features(
        self, X: np.ndarray, raw_scores: np.ndarray, top_k: int = 7
    ) -> list[list[str]]:
        """Return the top contributing feature names for each row.

        Args:
            X:          2-D feature array for all rows, shape (n, p).
            raw_scores: 1-D raw anomaly scores, shape (n,).
            top_k:      Maximum number of features to return per row.

        Returns:
            List of length n. Each element is a list of up to top_k
            feature names, ordered by contribution descending.
        """
        n_rows, n_feat = X.shape
        results: list[list[str]] = []

        print(f"[INFO] Computing feature attribution for {n_rows} rows...")

        for i in range(n_rows):
            row = X[i].copy()
            base_score = raw_scores[i]
            contributions = np.zeros(n_feat)

            for j in range(n_feat):
                # Mask out feature j and see how the score changes
                masked = row.copy()
                masked[j] = self.train_means[j]
                masked_score = self.model.raw_scores(
                    masked.reshape(1, -1)
                )[0]
                # Positive: removing this feature lowered score -> it was bad
                contributions[j] = base_score - masked_score

            total = np.sum(np.abs(contributions))
            if total == 0:
                results.append([])
                continue

            pct = contributions / total
            eligible = [
                (self.feature_cols[j], contributions[j])
                for j in range(n_feat)
                if contributions[j] > 0
                and abs(pct[j]) > self.MIN_CONTRIBUTION_PCT
            ]

            # Sort by contribution descending; break ties alphabetically
            eligible.sort(key=lambda x: (-x[1], x[0]))
            results.append([name for name, _ in eligible[:top_k]])

        return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def add_output_columns(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_features: list[list[str]],
) -> pd.DataFrame:
    """Append the 8 required output columns to the original DataFrame.

    Adds Abnormality_score (0-100) and top_feature_1 through
    top_feature_7. Slots with no contributing feature are left empty.

    Args:
        df:           The original input DataFrame.
        scores:       1-D array of normalised anomaly scores.
        top_features: List of per-row feature name lists.

    Returns:
        A copy of df with 8 new columns appended.
    """
    df = df.copy()
    df["Abnormality_score"] = np.round(scores, 4)

    for k in range(1, 8):
        col_name = f"top_feature_{k}"
        df[col_name] = [
            feats[k - 1] if len(feats) >= k else ""
            for feats in top_features
        ]

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_csv_path: str,
    output_csv_path: str,
    train_start: str = "1/1/2004 0:00",
    train_end: str = "1/5/2004 23:59",
    timestamp_col: str = "Time",
    smooth_window: int = 5,
) -> pd.DataFrame:
    """Run the full anomaly detection pipeline end to end.

    Loads the CSV, trains an Isolation Forest on the specified normal
    period, scores all rows, smooths the scores, normalises to 0-100,
    computes feature attribution, and writes the enriched CSV.

    Args:
        input_csv_path:  Path to the input CSV file.
        output_csv_path: Path where the enriched CSV will be written.
        train_start:     Start datetime of the normal (training) period.
        train_end:       End datetime of the normal (training) period.
        timestamp_col:   Name of the timestamp column in the CSV.
        smooth_window:   Rolling window size for score smoothing (default 5).
                         Set to 1 to disable smoothing.

    Returns:
        DataFrame with all original columns plus 8 new output columns.

    Raises:
        ValueError: If the training period contains fewer than 72 rows,
                    or if no numeric feature columns are found.
    """
    np.random.seed(42)

    # Step 1 - Load and clean the data
    processor = DataProcessor(timestamp_col=timestamp_col)
    df = processor.load(input_csv_path)
    feature_cols = processor.get_feature_columns(df)
    df = processor.clean(df, feature_cols)

    # feature_cols may shrink if constant columns were dropped
    feature_cols = processor.feature_cols
    X_full = df[feature_cols].values.astype(float)

    # Step 2 - Identify the training (normal) window
    if timestamp_col and timestamp_col in df.columns:
        ts = df[timestamp_col]
        train_mask = (
            (ts >= pd.Timestamp(train_start))
            & (ts <= pd.Timestamp(train_end))
        )
    else:
        # No timestamp column - fall back to the first 120 rows
        print("[WARN] No timestamp column; using first 120 rows.")
        train_mask = pd.Series([False] * len(df))
        train_mask.iloc[:120] = True

    n_train = int(train_mask.sum())
    if n_train < 72:
        raise ValueError(
            f"Training period contains only {n_train} rows. "
            "At least 72 hours of normal data are required."
        )
    print(f"[INFO] Training rows: {n_train}  |  Total rows: {len(df)}")

    X_train = X_full[train_mask]
    train_means = X_train.mean(axis=0)

    # Step 3 - Train the Isolation Forest on normal data only
    model = AnomalyModel()
    model.fit(X_train)

    # Step 4 - Score every row and smooth to remove abrupt jumps
    raw = model.raw_scores(X_full)

    smoother = ScoreSmoother(window=smooth_window)
    smoothed_raw = smoother.smooth(raw)

    # Step 5 - Normalise to 0-100 using training period as baseline
    normalizer = ScoreNormalizer()
    normalizer.fit(smoothed_raw[train_mask])
    scores = normalizer.normalize(smoothed_raw)

    # Validate that training-period scores look right
    train_scores = scores[train_mask]
    print(
        f"[VALIDATION] Training-period scores  "
        f"mean={train_scores.mean():.2f}  "
        f"max={train_scores.max():.2f}"
    )
    if train_scores.mean() >= 10:
        print("[WARN] Training-period mean >= 10. Check the training window.")

    # Step 6 - Identify the top contributing features per row
    attributor = FeatureAttributor(model, feature_cols, train_means)
    top_feats = attributor.top_features(X_full, raw)

    # Step 7 - Write the enriched CSV
    df_out = add_output_columns(df, scores, top_feats)
    df_out.to_csv(output_csv_path, index=False)
    print(f"[INFO] Results written to '{output_csv_path}'")

    return df_out


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multivariate Time Series Anomaly Detection"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output", required=True, help="Path for the enriched output CSV"
    )
    parser.add_argument(
        "--train-start",
        default="1/1/2004 0:00",
        help="Start of the normal training period (default: 1/1/2004 0:00)",
    )
    parser.add_argument(
        "--train-end",
        default="1/5/2004 23:59",
        help="End of the normal training period (default: 1/5/2004 23:59)",
    )
    parser.add_argument(
        "--timestamp-col",
        default="Time",
        help="Name of the timestamp column (default: Time)",
    )
    parser.add_argument(
        "--smooth-window",
        default=3,
        type=int,
        help="Rolling window for score smoothing; 1 = off (default: 3)",
    )

    args = parser.parse_args()

    run(
        input_csv_path=args.input,
        output_csv_path=args.output,
        train_start=args.train_start,
        train_end=args.train_end,
        timestamp_col=args.timestamp_col,
        smooth_window=args.smooth_window,
    )