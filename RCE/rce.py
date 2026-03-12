"""
Standard Restricted Coulomb Energy (RCE) classifier.

Prototype layer with spherical influence fields (Euclidean distance).
Training: prototype commitment + threshold modification.
Vectorised with numpy for practical speed on pixel-level data.
"""

import numpy as np
from scipy.spatial.distance import cdist


class RCE:
    """
    RCE classifier following the standard formulation:
      - Each prototype: (center, radius, label).
      - Fires when d(x, center) < radius.
      - Training: commit new prototypes; shrink wrong-class radii.
    """

    def __init__(self, R_max=100.0, default_label="background"):
        self.R_max = float(R_max)
        self.default_label = default_label
        self.centers_ = None   # (P, F) float64
        self.radii_ = None     # (P,)   float64
        self.labels_ = None    # (P,)   object array

    @property
    def prototypes_(self):
        """Legacy access: list of (center, radius, label) tuples."""
        if self.centers_ is None:
            return []
        return list(zip(self.centers_, self.radii_, self.labels_))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, f = X.shape

        # Precompute per-class indices and nearest-opposite distances
        unique_labels = np.unique(y)
        class_indices = {lbl: np.where(y == lbl)[0] for lbl in unique_labels}

        # For each sample, distance to nearest opposite-class sample (precomputed)
        nearest_opp = np.full(n, self.R_max, dtype=np.float64)
        for lbl in unique_labels:
            own = class_indices[lbl]
            opp_idx = np.where(y != lbl)[0]
            if len(opp_idx) == 0:
                continue
            # Chunked cdist to avoid huge memory
            chunk = 5000
            for start in range(0, len(own), chunk):
                batch = own[start:start + chunk]
                D = cdist(X[batch], X[opp_idx], metric="euclidean")
                nearest_opp[batch] = np.minimum(nearest_opp[batch], D.min(axis=1))

        # Preallocate prototype storage (grows by doubling)
        cap = min(n, 4096)
        P_centers = np.empty((cap, f), dtype=np.float64)
        P_radii = np.empty(cap, dtype=np.float64)
        P_labels = np.empty(cap, dtype=y.dtype)
        n_proto = 0

        for i in range(n):
            x = X[i]
            label = y[i]

            if n_proto > 0:
                dists = np.linalg.norm(
                    P_centers[:n_proto] - x, axis=1
                )  # (n_proto,)
                fired = dists < P_radii[:n_proto]
                L = P_labels[:n_proto]

                if np.any((L == label) & fired):
                    continue

                wrong_mask = (L != label) & fired
                if np.any(wrong_mask):
                    wrong_idx = np.where(wrong_mask)[0]
                    P_radii[wrong_idx] = dists[wrong_idx]

            # Commit new prototype
            r0 = min(float(nearest_opp[i]), self.R_max)
            r0 = max(r0, 1e-6)
            if n_proto >= cap:
                cap = cap * 2
                P_centers = np.resize(P_centers, (cap, f))
                P_radii = np.resize(P_radii, cap)
                P_labels = np.resize(P_labels, cap)
            P_centers[n_proto] = x
            P_radii[n_proto] = r0
            P_labels[n_proto] = label
            n_proto += 1

        self.centers_ = P_centers[:n_proto].copy()
        self.radii_ = P_radii[:n_proto].copy()
        self.labels_ = P_labels[:n_proto].copy()
        return self

    def predict(self, X):
        """
        Vectorised prediction. For each input, find activated prototypes
        (d < radius); pick the one with smallest distance and return its label.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = len(X)
        if self.centers_ is None or len(self.centers_) == 0:
            return np.full(n, self.default_label)

        C = self.centers_  # (P, F)
        R = self.radii_    # (P,)
        L = self.labels_   # (P,)
        default = self.default_label

        chunk = 10000
        results = []
        for start in range(0, n, chunk):
            Xb = X[start:start + chunk]   # (B, F)
            D = cdist(Xb, C, metric="euclidean")  # (B, P)
            # Mask distances >= radius (not activated)
            D_masked = np.where(D < R[np.newaxis, :], D, np.inf)
            best_idx = np.argmin(D_masked, axis=1)  # (B,)
            best_d = D_masked[np.arange(len(Xb)), best_idx]
            preds = np.where(best_d < np.inf, L[best_idx], default)
            results.append(preds)
        return np.concatenate(results)

    def predict_proba(self, X, sigma=0.1):
        """
        Output probability mode: p_i = exp(-sigma * d_i), summed by class,
        normalised.  Returns (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.centers_ is None or len(self.centers_) == 0:
            return np.ones((len(X), 1))

        C = self.centers_
        L = self.labels_
        unique = []
        for l in L:
            if l not in unique:
                unique.append(l)

        chunk = 10000
        probs_list = []
        for start in range(0, len(X), chunk):
            Xb = X[start:start + chunk]
            D = cdist(Xb, C, metric="euclidean")  # (B, P)
            E = np.exp(-sigma * D)                 # (B, P)
            class_probs = np.zeros((len(Xb), len(unique)))
            for j, lbl in enumerate(unique):
                mask = L == lbl
                class_probs[:, j] = E[:, mask].sum(axis=1)
            s = class_probs.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            probs_list.append(class_probs / s)
        return np.vstack(probs_list)
