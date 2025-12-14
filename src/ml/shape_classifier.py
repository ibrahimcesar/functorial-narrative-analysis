"""
ML Classifier for Narrative Shape Classification

This module addresses the core research question:
    "Are the Six Shapes Universal or Provincial?"

We train classifiers on information-geometric features to:
1. Classify narratives into Reagan's 6 shapes (or our 5 info-geometric shapes)
2. Test cross-cultural generalization: train on Western, test on Japanese
3. Quantify shape universality via transfer learning performance

Key insight: If shapes are universal, a classifier trained on Western literature
should generalize to Japanese literature. If shapes are culturally specific,
cross-cultural transfer will fail.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


# Reagan's 6 shapes mapped to information-geometric signatures
REAGAN_SHAPES = {
    "rags_to_riches": "Steady rise in valence/entropy",
    "tragedy": "Steady fall (geodesic descent)",
    "man_in_hole": "Fall then rise (V-shape)",
    "icarus": "Rise then fall (inverted V)",
    "cinderella": "Rise-fall-rise (W-shape)",
    "oedipus": "Fall-rise-fall (M-shape)",
}

# Our information-geometric shapes
INFO_GEO_SHAPES = {
    "geodesic_tragedy": "Low curvature, smooth descent",
    "high_curvature_mystery": "Sustained high information rate",
    "random_walk_comedy": "High variance, oscillating",
    "compression_progress": "Steady entropy reduction",
    "discontinuous_twist": "Late curvature spike",
}


@dataclass
class FeatureVector:
    """Feature vector for a single work."""
    work_id: str
    title: str
    author: str
    corpus: str  # 'japanese' or 'western'

    # Trajectory features
    mean_surprisal: float = 0.0
    std_surprisal: float = 0.0
    min_surprisal: float = 0.0
    max_surprisal: float = 0.0

    # Curvature features
    mean_curvature: float = 0.0
    max_curvature: float = 0.0
    std_curvature: float = 0.0
    n_peaks: int = 0

    # Shape features
    skewness: float = 0.0
    kurtosis: float = 0.0
    entropy_change: float = 0.0
    arc_length: float = 0.0

    # Temporal features (quartile means)
    q1_mean: float = 0.0
    q2_mean: float = 0.0
    q3_mean: float = 0.0
    q4_mean: float = 0.0

    # Derivative features
    mean_velocity: float = 0.0
    std_velocity: float = 0.0

    # Labels
    shape_label: str = ""  # Primary shape classification

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML."""
        return np.array([
            self.mean_surprisal,
            self.std_surprisal,
            self.min_surprisal,
            self.max_surprisal,
            self.mean_curvature,
            self.max_curvature,
            self.std_curvature,
            self.n_peaks,
            self.skewness,
            self.kurtosis,
            self.entropy_change,
            self.arc_length,
            self.q1_mean,
            self.q2_mean,
            self.q3_mean,
            self.q4_mean,
            self.mean_velocity,
            self.std_velocity,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "mean_surprisal", "std_surprisal", "min_surprisal", "max_surprisal",
            "mean_curvature", "max_curvature", "std_curvature", "n_peaks",
            "skewness", "kurtosis", "entropy_change", "arc_length",
            "q1_mean", "q2_mean", "q3_mean", "q4_mean",
            "mean_velocity", "std_velocity",
        ]


@dataclass
class ClassifierResults:
    """Results from classifier evaluation."""
    classifier_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    cohen_kappa: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    feature_importances: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "classifier_name": self.classifier_name,
            "accuracy": float(self.accuracy),
            "f1_macro": float(self.f1_macro),
            "f1_weighted": float(self.f1_weighted),
            "cohen_kappa": float(self.cohen_kappa),
            "cv_scores": [float(s) for s in self.cv_scores],
            "cv_mean": float(self.cv_mean),
            "cv_std": float(self.cv_std),
            "confusion_matrix": self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            "classification_report": self.classification_report,
            "feature_importances": self.feature_importances,
        }


@dataclass
class UniversalityTest:
    """Results from cross-cultural universality test."""
    train_corpus: str
    test_corpus: str
    train_accuracy: float
    test_accuracy: float
    transfer_gap: float  # train_acc - test_acc
    test_f1: float
    test_kappa: float
    conclusion: str  # "universal", "partially_universal", "culture_specific"

    def to_dict(self) -> dict:
        return asdict(self)


def extract_features_from_trajectory(
    values: np.ndarray,
    positions: np.ndarray,
    work_id: str = "",
    title: str = "",
    author: str = "",
    corpus: str = "",
) -> FeatureVector:
    """
    Extract feature vector from a surprisal trajectory.
    """
    from scipy.stats import skew, kurtosis
    from scipy.signal import find_peaks

    n = len(values)

    # Basic statistics
    mean_surp = float(np.mean(values))
    std_surp = float(np.std(values))
    min_surp = float(np.min(values))
    max_surp = float(np.max(values))

    # Curvature (approximate)
    if n > 2:
        velocity = np.gradient(values, positions)
        acceleration = np.gradient(velocity, positions)
        curvature = np.abs(acceleration) / np.power(1 + velocity**2, 1.5)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

        mean_curv = float(np.mean(curvature))
        max_curv = float(np.max(curvature))
        std_curv = float(np.std(curvature))
        mean_vel = float(np.mean(velocity))
        std_vel = float(np.std(velocity))
    else:
        mean_curv = max_curv = std_curv = mean_vel = std_vel = 0.0

    # Peaks
    peaks, _ = find_peaks(values, prominence=0.5 * std_surp)
    n_peaks = len(peaks)

    # Shape moments
    skewness = float(skew(values))
    kurt = float(kurtosis(values))

    # Entropy change (first quarter vs last quarter)
    q_size = n // 4
    if q_size > 0:
        entropy_change = float(np.mean(values[-q_size:]) - np.mean(values[:q_size]))
        q1_mean = float(np.mean(values[:q_size]))
        q2_mean = float(np.mean(values[q_size:2*q_size]))
        q3_mean = float(np.mean(values[2*q_size:3*q_size]))
        q4_mean = float(np.mean(values[3*q_size:]))
    else:
        entropy_change = q1_mean = q2_mean = q3_mean = q4_mean = 0.0

    # Arc length
    if n > 1:
        diffs = np.diff(values)
        pos_diffs = np.diff(positions)
        arc_length = float(np.sum(np.sqrt(diffs**2 + pos_diffs**2)))
    else:
        arc_length = 0.0

    return FeatureVector(
        work_id=work_id,
        title=title,
        author=author,
        corpus=corpus,
        mean_surprisal=mean_surp,
        std_surprisal=std_surp,
        min_surprisal=min_surp,
        max_surprisal=max_surp,
        mean_curvature=mean_curv,
        max_curvature=max_curv,
        std_curvature=std_curv,
        n_peaks=n_peaks,
        skewness=skewness,
        kurtosis=kurt,
        entropy_change=entropy_change,
        arc_length=arc_length,
        q1_mean=q1_mean,
        q2_mean=q2_mean,
        q3_mean=q3_mean,
        q4_mean=q4_mean,
        mean_velocity=mean_vel,
        std_velocity=std_vel,
    )


def assign_shape_label(features: FeatureVector) -> str:
    """
    Assign a shape label based on trajectory features.

    Uses Reagan-inspired categories mapped to information-geometric signatures:
    - rags_to_riches: Positive entropy change, positive skew
    - tragedy: Negative entropy change, negative skew
    - man_in_hole: V-shape (low middle, high ends)
    - icarus: Inverted V (high middle, low ends)
    - cinderella: W-shape (q4 > q2, oscillating)
    - oedipus: M-shape (q2 > q4, oscillating)
    """
    ec = features.entropy_change
    sk = features.skewness

    # Quartile pattern
    q_pattern = [features.q1_mean, features.q2_mean, features.q3_mean, features.q4_mean]
    q_mid = (features.q2_mean + features.q3_mean) / 2
    q_ends = (features.q1_mean + features.q4_mean) / 2

    # High variance = likely oscillating (comedy/complex)
    cv = features.std_surprisal / features.mean_surprisal if features.mean_surprisal > 0 else 0

    # Classification logic
    if ec > 0.1 and sk > 0.3:
        return "rags_to_riches"
    elif ec < -0.1 and sk < -0.3:
        return "tragedy"
    elif q_mid < q_ends * 0.9:  # Low middle = V-shape
        return "man_in_hole"
    elif q_mid > q_ends * 1.1:  # High middle = inverted V
        return "icarus"
    elif features.n_peaks >= 3 and features.q4_mean > features.q2_mean:
        return "cinderella"
    elif features.n_peaks >= 3 and features.q2_mean > features.q4_mean:
        return "oedipus"
    elif cv > 0.25:
        return "random_walk"  # Comedy/episodic
    else:
        return "other"


class ShapeClassifier:
    """
    ML classifier for narrative shapes.

    Addresses: "Are Six Shapes Universal or Provincial?"
    """

    def __init__(self, classifier_type: str = "random_forest"):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False

        # Initialize classifier
        if classifier_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
            )
        elif classifier_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif classifier_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
        elif classifier_type == "svm":
            self.model = SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def fit(self, features: List[FeatureVector], labels: Optional[List[str]] = None):
        """
        Fit classifier on feature vectors.

        If labels not provided, uses auto-assigned shape labels.
        """
        # Extract arrays
        X = np.array([f.to_array() for f in features])

        if labels is None:
            labels = [f.shape_label if f.shape_label else assign_shape_label(f) for f in features]

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, features: List[FeatureVector]) -> List[str]:
        """Predict shape labels."""
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X = np.array([f.to_array() for f in features])
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        return self.label_encoder.inverse_transform(y_pred).tolist()

    def evaluate(
        self,
        features: List[FeatureVector],
        labels: Optional[List[str]] = None,
        cv_folds: int = 5,
    ) -> ClassifierResults:
        """
        Evaluate classifier with cross-validation.
        """
        X = np.array([f.to_array() for f in features])

        if labels is None:
            labels = [f.shape_label if f.shape_label else assign_shape_label(f) for f in features]

        y = self.label_encoder.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        cv = StratifiedKFold(n_splits=min(cv_folds, len(set(y))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')

        # Full fit for final metrics
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        y_pred = self.model.predict(X_scaled)

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Classification report
        report = classification_report(
            y, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0,
        )

        # Feature importances (if available)
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = dict(zip(
                FeatureVector.feature_names(),
                self.model.feature_importances_.tolist()
            ))

        return ClassifierResults(
            classifier_name=self.classifier_type,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            cohen_kappa=kappa,
            cv_scores=cv_scores.tolist(),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            confusion_matrix=cm,
            classification_report=report,
            feature_importances=importances,
        )


def test_universality(
    western_features: List[FeatureVector],
    japanese_features: List[FeatureVector],
    classifier_type: str = "random_forest",
) -> UniversalityTest:
    """
    Test the universality hypothesis:
    Train on Western corpus, test on Japanese corpus.

    If shapes are universal:
        - Transfer accuracy should be similar to training accuracy
        - Transfer gap should be small (<10%)

    If shapes are culture-specific:
        - Transfer accuracy will be much lower
        - Transfer gap will be large (>20%)
    """
    # Assign labels
    for f in western_features:
        if not f.shape_label:
            f.shape_label = assign_shape_label(f)
    for f in japanese_features:
        if not f.shape_label:
            f.shape_label = assign_shape_label(f)

    # Get common labels
    western_labels = [f.shape_label for f in western_features]
    japanese_labels = [f.shape_label for f in japanese_features]

    all_labels = list(set(western_labels + japanese_labels))

    # Create classifier
    clf = ShapeClassifier(classifier_type)

    # Train on Western
    clf.fit(western_features, western_labels)

    # Evaluate on training set
    train_pred = clf.predict(western_features)
    train_acc = accuracy_score(
        clf.label_encoder.transform(western_labels),
        clf.label_encoder.transform(train_pred)
    )

    # Test on Japanese
    test_pred = clf.predict(japanese_features)

    # Handle unseen labels
    valid_mask = [l in clf.label_encoder.classes_ for l in japanese_labels]
    if not all(valid_mask):
        # Some labels in Japanese not seen in Western
        japanese_labels_filtered = [l for l, v in zip(japanese_labels, valid_mask) if v]
        test_pred_filtered = [p for p, v in zip(test_pred, valid_mask) if v]

        if len(japanese_labels_filtered) > 0:
            test_acc = accuracy_score(
                clf.label_encoder.transform(japanese_labels_filtered),
                clf.label_encoder.transform(test_pred_filtered)
            )
            test_f1 = f1_score(
                clf.label_encoder.transform(japanese_labels_filtered),
                clf.label_encoder.transform(test_pred_filtered),
                average='weighted',
                zero_division=0,
            )
            test_kappa = cohen_kappa_score(
                clf.label_encoder.transform(japanese_labels_filtered),
                clf.label_encoder.transform(test_pred_filtered),
            )
        else:
            test_acc = test_f1 = test_kappa = 0.0
    else:
        test_acc = accuracy_score(
            clf.label_encoder.transform(japanese_labels),
            clf.label_encoder.transform(test_pred)
        )
        test_f1 = f1_score(
            clf.label_encoder.transform(japanese_labels),
            clf.label_encoder.transform(test_pred),
            average='weighted',
            zero_division=0,
        )
        test_kappa = cohen_kappa_score(
            clf.label_encoder.transform(japanese_labels),
            clf.label_encoder.transform(test_pred),
        )

    transfer_gap = train_acc - test_acc

    # Conclusion
    if transfer_gap < 0.10:
        conclusion = "universal"
    elif transfer_gap < 0.20:
        conclusion = "partially_universal"
    else:
        conclusion = "culture_specific"

    return UniversalityTest(
        train_corpus="Western",
        test_corpus="Japanese",
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        transfer_gap=float(transfer_gap),
        test_f1=float(test_f1),
        test_kappa=float(test_kappa),
        conclusion=conclusion,
    )


def run_full_analysis(
    japanese_dir: Path,
    western_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Run complete shape classification and universality analysis.
    """
    from src.geometry.surprisal import SurprisalExtractor

    extractor = SurprisalExtractor(method='entropy', window_size=200)

    # Extract features from Japanese corpus
    print("Extracting features from Japanese corpus...")
    japanese_features = []
    for f in sorted(japanese_dir.glob("*.json")):
        with open(f) as file:
            data = json.load(file)

        text = data.get('text', '')
        if len(text) < 5000:
            continue

        try:
            trajectory = extractor.extract(text)
            features = extract_features_from_trajectory(
                trajectory.values,
                trajectory.positions,
                work_id=str(data.get('book_id', f.stem)),
                title=data.get('title', 'Unknown'),
                author=data.get('author', 'Unknown'),
                corpus='japanese',
            )
            features.shape_label = assign_shape_label(features)
            japanese_features.append(features)
        except Exception as e:
            print(f"  Error processing {data.get('title')}: {e}")

    print(f"  Extracted {len(japanese_features)} Japanese works")

    # Extract features from Western corpus
    print("Extracting features from Western corpus...")
    western_features = []
    for f in sorted(western_dir.glob("*.json")):
        with open(f) as file:
            data = json.load(file)

        text = data.get('text', '')
        if len(text) < 5000:
            continue

        try:
            trajectory = extractor.extract(text)
            features = extract_features_from_trajectory(
                trajectory.values,
                trajectory.positions,
                work_id=str(data.get('id', f.stem)),
                title=data.get('title', 'Unknown'),
                author=data.get('author', 'Unknown'),
                corpus='western',
            )
            features.shape_label = assign_shape_label(features)
            western_features.append(features)
        except Exception as e:
            print(f"  Error processing {data.get('title')}: {e}")

    print(f"  Extracted {len(western_features)} Western works")

    # Shape distribution
    jp_shapes = Counter(f.shape_label for f in japanese_features)
    west_shapes = Counter(f.shape_label for f in western_features)

    print("\n=== SHAPE DISTRIBUTION ===")
    print(f"{'Shape':<20} {'Japanese':<15} {'Western':<15}")
    print("-" * 50)
    all_shapes = sorted(set(jp_shapes.keys()) | set(west_shapes.keys()))
    for shape in all_shapes:
        jp_pct = jp_shapes.get(shape, 0) / len(japanese_features) * 100 if japanese_features else 0
        west_pct = west_shapes.get(shape, 0) / len(western_features) * 100 if western_features else 0
        print(f"{shape:<20} {jp_pct:>5.1f}%          {west_pct:>5.1f}%")

    # Train and evaluate classifiers
    print("\n=== CLASSIFIER EVALUATION ===")
    all_features = japanese_features + western_features

    results = {}
    for clf_type in ["random_forest", "logistic", "gradient_boosting"]:
        clf = ShapeClassifier(clf_type)
        eval_results = clf.evaluate(all_features)
        results[clf_type] = eval_results.to_dict()

        print(f"\n{clf_type.upper()}:")
        print(f"  Accuracy: {eval_results.accuracy:.3f}")
        print(f"  F1 (macro): {eval_results.f1_macro:.3f}")
        print(f"  CV Mean: {eval_results.cv_mean:.3f} ± {eval_results.cv_std:.3f}")

    # Universality test
    print("\n=== UNIVERSALITY TEST ===")
    print("Training on Western, testing on Japanese...")

    universality = test_universality(western_features, japanese_features)

    print(f"  Train accuracy (Western): {universality.train_accuracy:.3f}")
    print(f"  Test accuracy (Japanese): {universality.test_accuracy:.3f}")
    print(f"  Transfer gap: {universality.transfer_gap:.3f}")
    print(f"  Conclusion: {universality.conclusion.upper()}")

    # Compile results
    output = {
        "corpus_sizes": {
            "japanese": len(japanese_features),
            "western": len(western_features),
        },
        "shape_distribution": {
            "japanese": dict(jp_shapes),
            "western": dict(west_shapes),
        },
        "classifier_results": results,
        "universality_test": universality.to_dict(),
        "feature_names": FeatureVector.feature_names(),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    # Run analysis on expanded corpus
    japanese_dir = Path("data/raw/corpus_expanded/japanese")
    western_dir = Path("data/raw/corpus_expanded/western")
    output_path = Path("data/results/ml_shape_analysis.json")

    results = run_full_analysis(japanese_dir, western_dir, output_path)
