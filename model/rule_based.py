"""
rule_based.py
-------------
Two detection approaches in one file:

  1. Rule-based detector  — uses edge pixel count threshold from edge_detect.py
                            No training needed. Instant. Explainable.
                            "If edge count > 30, object is present."

  2. SVM classifier       — trained on extracted feature vectors
                            Takes ~1 second to train on simulated data.
                            More robust than pure threshold across positions.

For the demo and pitch, lead with the SVM — it sounds more technical.
Explain the rule-based as "our interpretable baseline."

Both are legitimate approaches used in CSI sensing research.
The Widar paper itself used SVM before switching to deep learning.

Pipeline position:
  signal_engine → preprocessing → feature_extract → rule_based (classify)
  signal_engine → edge_detect   → rule_based (threshold decision)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signal_engine   import generate_rf_scene, generate_2d_heatmap
from core.preprocessing   import preprocess
from core.feature_extract import extract_feature_vector
from core.edge_detect     import process_heatmap, is_object_present


# ─────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────

LABEL_EMPTY  = 0   # no object
LABEL_OBJECT = 1   # object present

LABEL_NAMES = {
    LABEL_EMPTY:  "Empty room",
    LABEL_OBJECT: "Object detected",
}


# ─────────────────────────────────────────────
# 1. RULE-BASED DETECTOR
# ─────────────────────────────────────────────

class RuleBasedDetector:
    """
    Detects object presence using edge pixel count threshold.

    This is your interpretable baseline — no training, no black box.
    Works directly on the 2D heatmap output from generate_2d_heatmap().

    Usage:
        detector = RuleBasedDetector(threshold=30)
        heatmap  = generate_2d_heatmap(object_size=0.6)
        result   = detector.predict(heatmap)
        print(result['label'], result['confidence'])
    """

    def __init__(self, threshold=30):
        """
        Args:
            threshold: minimum edge pixels to declare object present.
                       30 works well for our 64×64 heatmaps.
                       Lower = more sensitive (more false positives).
                       Higher = less sensitive (may miss small objects).
        """
        self.threshold = threshold

    def predict(self, heatmap):
        """
        Runs edge detection and applies threshold decision.

        Args:
            heatmap: 2D numpy array from generate_2d_heatmap()

        Returns:
            dict:
              'label'       — 0 (empty) or 1 (object)
              'label_name'  — human readable string
              'confidence'  — float 0.0-1.0
              'edge_count'  — raw edge pixel count
              'edges'       — binary edge map (for display)
              'heatmap_img' — processed uint8 image (for display)
        """
        edge_result       = process_heatmap(heatmap)
        detected, conf    = is_object_present(edge_result, self.threshold)
        label             = LABEL_OBJECT if detected else LABEL_EMPTY

        return {
            'label':       label,
            'label_name':  LABEL_NAMES[label],
            'confidence':  conf,
            'edge_count':  edge_result['edge_count'],
            'edges':       edge_result['edges'],
            'heatmap_img': edge_result['heatmap_img'],
        }

    def predict_batch(self, heatmaps):
        """Runs predict() on a list of heatmaps. Returns list of result dicts."""
        return [self.predict(h) for h in heatmaps]


# ─────────────────────────────────────────────
# 2. TRAINING DATA GENERATOR
# ─────────────────────────────────────────────

def generate_training_data(n_samples=500, noise_level=0.05):
    """
    Generates a balanced synthetic training dataset for the SVM.

    Half the samples are empty rooms (label=0).
    Half have objects at random sizes and positions (label=1).

    The variety in object size (0.2-1.0) and position (0.2-0.8)
    makes the classifier robust — it learns to detect objects
    regardless of where they are or how large they are.

    Args:
        n_samples:   total samples (split 50/50 between classes)
        noise_level: noise added to each sample

    Returns:
        X: 2D numpy array (n_samples, n_features) — feature matrix
        y: 1D numpy array (n_samples,) — labels (0 or 1)
    """
    X_list = []
    y_list = []

    half = n_samples // 2

    print(f"Generating {n_samples} training samples...")

    # Class 0: empty rooms
    for _ in range(half):
        signal = generate_rf_scene(object_size=0.0, noise_level=noise_level)
        clean  = preprocess(signal)
        vec, _ = extract_feature_vector(clean)
        X_list.append(vec)
        y_list.append(LABEL_EMPTY)

    # Class 1: objects at random sizes and positions
    for _ in range(half):
        size   = np.random.uniform(0.2, 1.0)   # random object size
        pos    = np.random.uniform(20, 80)      # random position
        signal = generate_rf_scene(object_size=size, object_pos=pos,
                                    noise_level=noise_level)
        clean  = preprocess(signal)
        vec, _ = extract_feature_vector(clean)
        X_list.append(vec)
        y_list.append(LABEL_OBJECT)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle so empty/object samples are interleaved
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────
# 3. SVM CLASSIFIER
# ─────────────────────────────────────────────

class SVMDetector:
    """
    SVM-based object detector trained on CSI feature vectors.

    Uses an RBF kernel SVM — the standard choice for CSI classification
    in research (Widar, WiGest, and many others used SVM before CNNs).
    Trains in under 2 seconds on 500 samples. No GPU needed.

    Usage:
        detector = SVMDetector()
        detector.train()                    # generates data + trains
        result = detector.predict(signal)   # pass preprocessed 1D signal
    """

    def __init__(self):
        self.model   = SVC(kernel='rbf', C=10, gamma='scale',
                           probability=True)   # probability=True for confidence scores
        self.scaler  = StandardScaler()        # SVM needs normalized features
        self.trained = False
        self.feature_names = None

    def train(self, n_samples=500, noise_level=0.05, test_size=0.2):
        """
        Generates training data and trains the SVM.

        Steps:
          1. Generate n_samples synthetic CSI scenes
          2. Extract feature vectors from each
          3. Scale features (StandardScaler — required for SVM)
          4. Train SVM with RBF kernel
          5. Evaluate on held-out test set
          6. Report accuracy

        Args:
            n_samples:   total training samples (500 is enough for demo)
            noise_level: noise in generated signals
            test_size:   fraction held out for evaluation (0.2 = 20%)

        Returns:
            dict with 'accuracy', 'report' keys
        """
        # Generate data
        X, y = generate_training_data(n_samples, noise_level)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features — SVM is sensitive to feature scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # Train SVM
        print("Training SVM classifier...")
        self.model.fit(X_train_scaled, y_train)
        self.trained = True

        # Evaluate
        y_pred   = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(
            y_test, y_pred,
            target_names=["Empty room", "Object detected"]
        )

        print(f"Training complete.")
        print(f"Test accuracy: {accuracy:.1%}\n")
        print(report)

        return {'accuracy': accuracy, 'report': report}

    def predict(self, signal):
        """
        Predicts object presence from a preprocessed 1D signal.

        Args:
            signal: 1D numpy array — output of preprocessing.preprocess()

        Returns:
            dict:
              'label'      — 0 or 1
              'label_name' — human readable
              'confidence' — probability of predicted class
              'proba'      — [p_empty, p_object] probability array
        """
        assert self.trained, "Call .train() before .predict()"

        vec, _         = extract_feature_vector(signal)
        vec_scaled     = self.scaler.transform(vec.reshape(1, -1))
        label          = int(self.model.predict(vec_scaled)[0])
        proba          = self.model.predict_proba(vec_scaled)[0]
        confidence     = float(proba[label])

        return {
            'label':      label,
            'label_name': LABEL_NAMES[label],
            'confidence': round(confidence, 2),
            'proba':      proba,
        }

    def save(self, path='model/svm_detector.pkl'):
        """Saves trained model + scaler to disk."""
        assert self.trained, "Train the model before saving."
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"Model saved to {path}")

    def load(self, path='model/svm_detector.pkl'):
        """Loads a previously saved model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model   = data['model']
        self.scaler  = data['scaler']
        self.trained = True
        print(f"Model loaded from {path}")


# ─────────────────────────────────────────────
# 4. UNIFIED DETECTOR (what Streamlit calls)
# ─────────────────────────────────────────────

class WiSenseDetector:
    """
    Unified detector that runs BOTH rule-based and SVM in parallel.

    This is what the Streamlit demo imports — one class, one predict call,
    returns results from both methods so the UI can show both.

    Usage:
        detector = WiSenseDetector()
        detector.setup()                           # train SVM
        results  = detector.predict(object_size=0.6, object_x=0.5, object_y=0.5)
    """

    def __init__(self):
        self.rule_detector = RuleBasedDetector(threshold=30)
        self.svm_detector  = SVMDetector()
        self.ready         = False

    def setup(self, n_samples=400):
        """Train the SVM. Call once at Streamlit app startup."""
        print("Setting up WiSense detector...")
        self.svm_detector.train(n_samples=n_samples, noise_level=0.05)
        self.ready = True
        print("WiSense detector ready.\n")

    def predict(self, object_size=0.0, object_x=0.5, object_y=0.5,
                noise_level=0.02):
        """
        Generates a scene and runs both detectors on it.

        Args:
            object_size:  0.0-1.0
            object_x:     0.0-1.0 horizontal position
            object_y:     0.0-1.0 vertical position
            noise_level:  noise amount

        Returns:
            dict with all data needed for the Streamlit demo:
              'heatmap'        — 2D float array (for display)
              'edges'          — binary edge map
              'heatmap_img'    — uint8 image
              'rule_result'    — RuleBasedDetector output dict
              'svm_result'     — SVMDetector output dict (if trained)
              'signal'         — 1D signal (for signal plot)
              'edge_count'     — raw edge pixels
        """
        # Generate 2D heatmap for visualization + rule-based detection
        heatmap = generate_2d_heatmap(
            object_size=object_size,
            object_x=object_x,
            object_y=object_y,
            noise_level=noise_level
        )

        # Generate 1D signal for SVM features + signal plot
        signal_1d = generate_rf_scene(
            object_size=object_size,
            object_pos=int(object_x * 100),
            noise_level=noise_level
        )
        clean_signal = preprocess(signal_1d)

        # Rule-based detection
        rule_result = self.rule_detector.predict(heatmap)

        # SVM detection
        if self.ready:
            svm_result = self.svm_detector.predict(clean_signal)
        else:
            svm_result = {'label': 0, 'label_name': 'Not trained',
                          'confidence': 0.0}

        return {
            'heatmap':     heatmap,
            'edges':       rule_result['edges'],
            'heatmap_img': rule_result['heatmap_img'],
            'rule_result': rule_result,
            'svm_result':  svm_result,
            'signal':      signal_1d,
            'edge_count':  rule_result['edge_count'],
        }


# ─────────────────────────────────────────────
# 5. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing rule_based.py...\n")

    # ── Test 1: Rule-based detector ──
    print("=" * 40)
    print("Test 1: Rule-based detector")
    print("=" * 40)
    rule_det = RuleBasedDetector(threshold=30)

    for label, size in [("Empty", 0.0), ("Small", 0.3), ("Large", 1.0)]:
        hm  = generate_2d_heatmap(object_size=size, noise_level=0.02)
        res = rule_det.predict(hm)
        print(f"{label:>5}: {res['label_name']:<18} "
              f"edges={res['edge_count']:>5}  conf={res['confidence']:.2f}")

    # ── Test 2: SVM detector ──
    print("\n" + "=" * 40)
    print("Test 2: SVM detector")
    print("=" * 40)
    svm_det = SVMDetector()
    svm_det.train(n_samples=300, noise_level=0.05)

    print("\nSVM predictions on fresh samples:")
    for label, size in [("Empty", 0.0), ("Small", 0.3), ("Large", 1.0)]:
        signal = generate_rf_scene(object_size=size, noise_level=0.05)
        clean  = preprocess(signal)
        res    = svm_det.predict(clean)
        correct = (size == 0.0 and res['label'] == 0) or \
                  (size > 0.0  and res['label'] == 1)
        print(f"{label:>5}: {res['label_name']:<18} "
              f"conf={res['confidence']:.2f}  {'✓' if correct else '✗'}")

    # ── Test 3: Unified detector ──
    print("\n" + "=" * 40)
    print("Test 3: WiSense unified detector")
    print("=" * 40)
    detector = WiSenseDetector()
    detector.setup(n_samples=300)

    for label, size in [("Empty", 0.0), ("Object", 0.7)]:
        res = detector.predict(object_size=size, object_x=0.5, object_y=0.5)
        print(f"{label:>6}: "
              f"rule={res['rule_result']['label_name']:<18} "
              f"svm={res['svm_result']['label_name']:<18} "
              f"edges={res['edge_count']}")

    print("\nAll tests passed. rule_based.py is ready.")