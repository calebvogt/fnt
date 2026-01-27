"""
USV Classifier - Random Forest classifier for USV detection refinement.

This module provides a trainable classifier that learns to distinguish
true USV calls from noise based on acoustic features extracted from
candidate detections.

Typical workflow:
1. Run DSP detector to get candidate detections
2. Manually label candidates in USV Inspector (accept/reject)
3. Export training data
4. Train classifier on labeled data
5. Use classifier to filter future DSP detections

Example usage:
    # Training
    classifier = USVClassifier()
    classifier.train_from_export("path/to/fnt_usv_training_data")
    classifier.save("my_usv_model")

    # Inference
    classifier = USVClassifier.load("my_usv_model")
    predictions = classifier.predict(features_df)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Scikit-learn imports (optional dependency)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_score, recall_score, f1_score, accuracy_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Joblib for model persistence
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@dataclass
class USVClassifierConfig:
    """Configuration for USV Random Forest classifier."""

    # Model hyperparameters
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"  # Handle imbalanced classes
    random_state: int = 42
    n_jobs: int = -1  # Use all cores

    # Feature columns to use (if None, use all numeric columns)
    feature_columns: Optional[List[str]] = None

    # Training metadata
    trained_on: Optional[str] = None
    training_date: Optional[str] = None
    n_training_samples: int = 0
    n_usv_samples: int = 0
    n_noise_samples: int = 0

    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def to_json(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'USVClassifierConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Handle None values and type conversions
        if data.get('feature_columns') == 'null':
            data['feature_columns'] = None
        if data.get('max_depth') == 'null':
            data['max_depth'] = None
        return cls(**data)


# Default feature columns for classification
DEFAULT_FEATURE_COLUMNS = [
    'duration_ms',
    'bandwidth_hz',
    'center_freq_hz',
    'spectral_centroid_hz',
    'spectral_bandwidth_hz',
    'spectral_flatness',
    'rms_power',
    'peak_power_db',
    'freq_modulation_rate',
    'zero_crossing_rate',
]


class USVClassifier:
    """Random Forest classifier for USV detection.

    Learns to distinguish true USV calls from noise based on
    acoustic features extracted from spectrogram regions.
    """

    def __init__(self, config: Optional[USVClassifierConfig] = None):
        """Initialize classifier.

        Args:
            config: Classifier configuration. If None, uses defaults.
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for USVClassifier. "
                "Install with: pip install scikit-learn"
            )

        self.config = config or USVClassifierConfig()
        self.model: Optional[RandomForestClassifier] = None
        self.is_trained = False

    def train(self,
              features: pd.DataFrame,
              labels: pd.Series,
              test_size: float = 0.2,
              cv_folds: int = 5) -> Dict:
        """Train the classifier on labeled features.

        Args:
            features: DataFrame with feature columns
            labels: Series with 'usv' or 'noise' labels
            test_size: Fraction of data for test set
            cv_folds: Number of cross-validation folds

        Returns:
            Dict with training metrics
        """
        # Determine feature columns
        if self.config.feature_columns:
            feature_cols = [c for c in self.config.feature_columns if c in features.columns]
        else:
            # Use default features that exist in the data
            feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in features.columns]

        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found in data")

        self.config.feature_columns = feature_cols

        # Prepare data
        X = features[feature_cols].values
        y = (labels == 'usv').astype(int).values  # 1 = USV, 0 = noise

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state, stratify=y
        )

        # Create and train model
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        self.config.accuracy = float(accuracy_score(y_test, y_pred))
        self.config.precision = float(precision_score(y_test, y_pred))
        self.config.recall = float(recall_score(y_test, y_pred))
        self.config.f1 = float(f1_score(y_test, y_pred))

        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
        self.config.cv_scores = [float(s) for s in cv_scores]

        # Feature importances
        importances = self.model.feature_importances_
        self.config.feature_importances = {
            col: float(imp) for col, imp in zip(feature_cols, importances)
        }

        # Training metadata
        self.config.n_training_samples = len(y)
        self.config.n_usv_samples = int(np.sum(y))
        self.config.n_noise_samples = int(np.sum(1 - y))
        self.config.training_date = datetime.now().isoformat()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': self.config.accuracy,
            'precision': self.config.precision,
            'recall': self.config.recall,
            'f1': self.config.f1,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'confusion_matrix': cm.tolist(),
            'feature_importances': self.config.feature_importances,
            'n_train': len(y_train),
            'n_test': len(y_test),
        }

    def train_from_export(self, export_dir: str) -> Dict:
        """Train from an exported training data folder.

        Args:
            export_dir: Path to fnt_usv_training_* folder

        Returns:
            Dict with training metrics
        """
        export_path = Path(export_dir)
        metadata_path = export_path / "metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.csv found in {export_dir}")

        # Load metadata
        df = pd.read_csv(metadata_path)

        if 'label' not in df.columns:
            raise ValueError("metadata.csv must have 'label' column")

        self.config.trained_on = str(export_path.name)

        return self.train(df, df['label'])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict labels for new detections.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of predicted labels ('usv' or 'noise')
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Classifier must be trained before prediction")

        # Get feature columns
        feature_cols = self.config.feature_columns or DEFAULT_FEATURE_COLUMNS
        feature_cols = [c for c in feature_cols if c in features.columns]

        X = features[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y_pred = self.model.predict(X)

        return np.array(['usv' if p == 1 else 'noise' for p in y_pred])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for new detections.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of shape (n_samples, 2) with [noise_prob, usv_prob]
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Classifier must be trained before prediction")

        feature_cols = self.config.feature_columns or DEFAULT_FEATURE_COLUMNS
        feature_cols = [c for c in feature_cols if c in features.columns]

        X = features[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return self.model.predict_proba(X)

    def save(self, model_dir: str):
        """Save trained model and config to directory.

        Creates:
            model_dir/
            ├── model.joblib      # Trained sklearn model
            ├── config.json       # Model configuration and metrics
            └── README.txt        # Human-readable summary

        Args:
            model_dir: Directory to save model files
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save untrained model")

        if not HAS_JOBLIB:
            raise ImportError("joblib is required to save models. Install with: pip install joblib")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, model_path / "model.joblib")

        # Save config
        self.config.to_json(str(model_path / "config.json"))

        # Create README
        readme = self._create_readme()
        with open(model_path / "README.txt", 'w') as f:
            f.write(readme)

    @classmethod
    def load(cls, model_dir: str) -> 'USVClassifier':
        """Load trained model from directory.

        Args:
            model_dir: Directory containing model.joblib and config.json

        Returns:
            Loaded USVClassifier instance
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for USVClassifier")
        if not HAS_JOBLIB:
            raise ImportError("joblib is required to load models")

        model_path = Path(model_dir)

        # Load config
        config = USVClassifierConfig.from_json(str(model_path / "config.json"))

        # Create classifier
        classifier = cls(config)

        # Load model
        classifier.model = joblib.load(model_path / "model.joblib")
        classifier.is_trained = True

        return classifier

    def _create_readme(self) -> str:
        """Create human-readable model summary."""
        # Sort features by importance
        sorted_features = sorted(
            self.config.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        feature_str = "\n".join([
            f"    {name}: {imp:.4f}" for name, imp in sorted_features
        ])

        return f"""FNT USV Classifier Model
========================
Trained: {self.config.training_date}
Source: {self.config.trained_on or 'Unknown'}

Training Data:
  Total samples: {self.config.n_training_samples}
  USV samples: {self.config.n_usv_samples}
  Noise samples: {self.config.n_noise_samples}

Model: Random Forest
  n_estimators: {self.config.n_estimators}
  max_depth: {self.config.max_depth}
  class_weight: {self.config.class_weight}

Performance (Test Set):
  Accuracy:  {self.config.accuracy:.3f}
  Precision: {self.config.precision:.3f}
  Recall:    {self.config.recall:.3f}
  F1 Score:  {self.config.f1:.3f}

Cross-Validation ({len(self.config.cv_scores)}-fold):
  Mean: {np.mean(self.config.cv_scores):.3f} (+/- {np.std(self.config.cv_scores):.3f})

Feature Importances:
{feature_str}

Usage:
    from fnt.usv.usv_classifier import USVClassifier

    classifier = USVClassifier.load("{self.config.trained_on or 'model_dir'}")
    predictions = classifier.predict(features_df)
    probabilities = classifier.predict_proba(features_df)
"""

    def get_feature_importance_report(self) -> str:
        """Get formatted feature importance report."""
        if not self.config.feature_importances:
            return "No feature importances available (model not trained)"

        sorted_features = sorted(
            self.config.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        lines = ["Feature Importances:", "=" * 40]
        for name, imp in sorted_features:
            bar = "█" * int(imp * 50)
            lines.append(f"{name:25s} {imp:.4f} {bar}")

        return "\n".join(lines)


def extract_features_from_detection(
    audio_data: np.ndarray,
    sample_rate: int,
    start_s: float,
    stop_s: float,
    min_freq: float,
    max_freq: float
) -> Dict:
    """Extract features from a single detection region.

    This is a standalone function that can be used during inference
    without needing the full Inspector UI.

    Args:
        audio_data: Full audio signal
        sample_rate: Sample rate in Hz
        start_s: Detection start time in seconds
        stop_s: Detection stop time in seconds
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz

    Returns:
        Dict of features for classification
    """
    from scipy import signal as scipy_signal

    # Get audio segment
    start_sample = int(start_s * sample_rate)
    stop_sample = int(stop_s * sample_rate)
    segment = audio_data[start_sample:stop_sample]

    if len(segment) < 10:
        segment = np.zeros(100)

    features = {
        'start_seconds': start_s,
        'stop_seconds': stop_s,
        'duration_ms': (stop_s - start_s) * 1000,
        'min_freq_hz': min_freq,
        'max_freq_hz': max_freq,
        'bandwidth_hz': max_freq - min_freq,
        'center_freq_hz': (min_freq + max_freq) / 2,
    }

    # Time-domain features
    features['rms_power'] = float(np.sqrt(np.mean(segment ** 2)))
    features['peak_power_db'] = float(20 * np.log10(np.max(np.abs(segment)) + 1e-10))
    features['zero_crossing_rate'] = float(np.sum(np.diff(np.sign(segment)) != 0) / len(segment))

    # Spectral features
    if len(segment) >= 256:
        n_fft = min(1024, len(segment))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        fft_mag = np.abs(np.fft.rfft(segment[:n_fft]))
        power = fft_mag ** 2

        freq_mask = (freqs >= min_freq * 0.5) & (freqs <= max_freq * 1.5)
        if np.any(freq_mask):
            freqs_roi = freqs[freq_mask]
            power_roi = power[freq_mask]
            power_roi = power_roi / (np.sum(power_roi) + 1e-10)

            features['spectral_centroid_hz'] = float(np.sum(freqs_roi * power_roi))
            centroid = features['spectral_centroid_hz']
            features['spectral_bandwidth_hz'] = float(
                np.sqrt(np.sum(((freqs_roi - centroid) ** 2) * power_roi))
            )

            geo_mean = np.exp(np.mean(np.log(power_roi + 1e-10)))
            arith_mean = np.mean(power_roi)
            features['spectral_flatness'] = float(geo_mean / (arith_mean + 1e-10))
        else:
            features['spectral_centroid_hz'] = features['center_freq_hz']
            features['spectral_bandwidth_hz'] = features['bandwidth_hz'] / 2
            features['spectral_flatness'] = 0.5
    else:
        features['spectral_centroid_hz'] = features['center_freq_hz']
        features['spectral_bandwidth_hz'] = features['bandwidth_hz'] / 2
        features['spectral_flatness'] = 0.5

    # Frequency modulation rate
    try:
        if len(segment) >= 512:
            f, t, Sxx = scipy_signal.spectrogram(
                segment, fs=sample_rate, nperseg=256, noverlap=200
            )
            freq_mask = (f >= min_freq * 0.8) & (f <= max_freq * 1.2)
            if np.any(freq_mask) and Sxx.shape[1] > 1:
                Sxx_roi = Sxx[freq_mask, :]
                f_roi = f[freq_mask]
                peak_freqs = f_roi[np.argmax(Sxx_roi, axis=0)]
                if len(peak_freqs) > 2:
                    slope, _ = np.polyfit(np.arange(len(peak_freqs)), peak_freqs, 1)
                    dt = (stop_s - start_s) / len(peak_freqs)
                    features['freq_modulation_rate'] = float(slope / dt) if dt > 0 else 0.0
                else:
                    features['freq_modulation_rate'] = 0.0
            else:
                features['freq_modulation_rate'] = 0.0
        else:
            features['freq_modulation_rate'] = 0.0
    except:
        features['freq_modulation_rate'] = 0.0

    return features
