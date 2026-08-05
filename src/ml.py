"""
ml.py -- Self-contained NumPy implementations of the regression models used in
this study. No scikit-learn / torch dependency, so the full pipeline is
reproducible from a bare Python + NumPy environment.

Models
------
RidgeRegressor            closed-form L2-penalised linear regression
KNNRegressor              k-nearest-neighbour regression (Euclidean, standardised)
HistRandomForest          histogram-based random forest (bagging + feature subsampling)
HistGradientBoosting      histogram-based gradient boosting (squared-error)
MLPRegressor              1-hidden-layer -> 2-hidden-layer MLP, Adam, early stopping
HybridCNNDAFMRBF          1D-CNN feature extractor -> Dynamic Attention Fusion
                          Module -> RBF-activated regression head (the architecture
                          proposed in this manuscript)

All models expose .fit(X, y) / .predict(X) and take a numpy RandomState-style seed.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Linear
# ----------------------------------------------------------------------------

class RidgeRegressor:
    """Closed-form ridge regression. Exact, no optimisation error."""

    def __init__(self, alpha=1.0, **kw):
        self.alpha = alpha

    def fit(self, X, y):
        n, p = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        A = Xb.T @ Xb
        # do not penalise the intercept
        pen = self.alpha * np.eye(p + 1)
        pen[0, 0] = 0.0
        self.coef_ = np.linalg.solve(A + pen, Xb.T @ y)
        return self

    def predict(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return Xb @ self.coef_


# ----------------------------------------------------------------------------
# k-NN
# ----------------------------------------------------------------------------

class KNNRegressor:
    """Plain k-NN regression. Included because it is maximally sensitive to
    grouped/spatial leakage and therefore a useful diagnostic."""

    def __init__(self, k=10, **kw):
        self.k = k

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, block=512):
        out = np.empty(X.shape[0], dtype=float)
        tr_sq = (self.X_ ** 2).sum(1)
        for s in range(0, X.shape[0], block):
            e = min(s + block, X.shape[0])
            Q = X[s:e]
            d = (Q ** 2).sum(1)[:, None] + tr_sq[None, :] - 2.0 * Q @ self.X_.T
            k = min(self.k, self.X_.shape[0])
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            out[s:e] = self.y_[idx].mean(1)
        return out


# ----------------------------------------------------------------------------
# Histogram-based decision tree (shared by RF and GBM)
# ----------------------------------------------------------------------------

def _bin_features(X, n_bins, edges=None):
    """Quantile-bin each column into at most n_bins integer codes."""
    if edges is None:
        edges = []
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            e = np.unique(np.quantile(X[:, j], qs))
            edges.append(e)
    B = np.empty(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        B[:, j] = np.searchsorted(edges[j], X[:, j], side='left')
    return B, edges


class _HistTree:
    """Regression tree grown on pre-binned features using per-bin histograms.

    Splits are evaluated for every feature simultaneously via np.bincount and
    cumulative sums, which keeps the whole fit vectorised.
    """

    __slots__ = ('max_depth', 'min_samples_leaf', 'n_bins', 'max_features',
                 'rng', 'feat', 'thr', 'left', 'right', 'value', 'lam')

    def __init__(self, max_depth=8, min_samples_leaf=20, n_bins=64,
                 max_features=None, rng=None, lam=0.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.lam = lam

    def fit(self, B, g, n_feat_total):
        # node arrays
        self.feat, self.thr = [], []
        self.left, self.right, self.value = [], [], []
        idx_all = np.arange(B.shape[0])
        self._build(B, g, idx_all, 0, n_feat_total)
        self.feat = np.array(self.feat, dtype=np.int32)
        self.thr = np.array(self.thr, dtype=np.int32)
        self.left = np.array(self.left, dtype=np.int32)
        self.right = np.array(self.right, dtype=np.int32)
        self.value = np.array(self.value, dtype=float)
        return self

    def _new_node(self, val):
        self.feat.append(-1); self.thr.append(-1)
        self.left.append(-1); self.right.append(-1)
        self.value.append(val)
        return len(self.value) - 1

    def _build(self, B, g, idx, depth, n_feat_total):
        node = self._new_node(g[idx].mean() if len(idx) else 0.0)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return node

        gi = g[idx]
        n = len(idx)
        parent_sse = gi.sum() ** 2 / n

        if self.max_features is None or self.max_features >= n_feat_total:
            feats = np.arange(n_feat_total)
        else:
            feats = self.rng.choice(n_feat_total, self.max_features, replace=False)

        # Evaluate every candidate feature in one pass: offset each feature's
        # bin codes into a disjoint range and use a single bincount, then take
        # cumulative sums along the bin axis to score all split points at once.
        nb = self.n_bins + 1
        Bsub = B[np.ix_(idx, feats)]                       # (n, F)
        F = len(feats)
        offs = (np.arange(F) * nb)[None, :]
        flat = (Bsub + offs).ravel()
        cnt = np.bincount(flat, minlength=F * nb).astype(float).reshape(F, nb)
        ssum = np.bincount(flat, weights=np.repeat(gi, F),
                           minlength=F * nb).reshape(F, nb)
        cl = np.cumsum(cnt, axis=1)[:, :-1]
        sl = np.cumsum(ssum, axis=1)[:, :-1]
        cr = n - cl
        sr = gi.sum() - sl
        ok = (cl >= self.min_samples_leaf) & (cr >= self.min_samples_leaf)
        gain = np.where(ok,
                        sl ** 2 / np.maximum(cl, 1e-12)
                        + sr ** 2 / np.maximum(cr, 1e-12)
                        - parent_sse,
                        -np.inf)
        if not np.isfinite(gain).any():
            return node
        pos = int(np.argmax(gain))
        fi, b = divmod(pos, nb - 1)
        if gain[fi, b] <= 0.0:
            return node
        j = int(feats[fi])
        mask = B[idx, j] <= b
        li, ri = idx[mask], idx[~mask]
        if len(li) < self.min_samples_leaf or len(ri) < self.min_samples_leaf:
            return node

        self.feat[node] = j
        self.thr[node] = b
        self.left[node] = self._build(B, g, li, depth + 1, n_feat_total)
        self.right[node] = self._build(B, g, ri, depth + 1, n_feat_total)
        return node

    def predict(self, B):
        out = np.empty(B.shape[0], dtype=float)
        node = np.zeros(B.shape[0], dtype=np.int32)
        active = np.ones(B.shape[0], dtype=bool)
        while active.any():
            cur = node[active]
            leaf = self.feat[cur] < 0
            ai = np.where(active)[0]
            if leaf.any():
                out[ai[leaf]] = self.value[cur[leaf]]
                active[ai[leaf]] = False
            nl = ~leaf
            if nl.any():
                ii = ai[nl]
                cc = node[ii]
                go_left = B[ii, self.feat[cc]] <= self.thr[cc]
                node[ii] = np.where(go_left, self.left[cc], self.right[cc])
        return out


class HistRandomForest:
    def __init__(self, n_estimators=100, max_depth=12, min_samples_leaf=5,
                 max_features=0.5, n_bins=64, seed=0, **kw):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_bins = n_bins
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        B, self.edges_ = _bin_features(X, self.n_bins)
        n, p = X.shape
        mf = max(1, int(round(self.max_features * p)))
        self.trees_ = []
        for t in range(self.n_estimators):
            boot = rng.integers(0, n, n)
            tr = _HistTree(self.max_depth, self.min_samples_leaf, self.n_bins,
                           mf, np.random.default_rng(self.seed + t + 1))
            tr.fit(B[boot], y[boot], p)
            self.trees_.append(tr)
        return self

    def predict(self, X):
        B, _ = _bin_features(X, self.n_bins, self.edges_)
        acc = np.zeros(X.shape[0])
        for tr in self.trees_:
            acc += tr.predict(B)
        return acc / len(self.trees_)


class HistGradientBoosting:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=4,
                 min_samples_leaf=20, subsample=0.8, n_bins=64, seed=0, **kw):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.n_bins = n_bins
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        B, self.edges_ = _bin_features(X, self.n_bins)
        n, p = X.shape
        self.base_ = y.mean()
        resid = y - self.base_
        self.trees_ = []
        ns = max(1, int(self.subsample * n))
        for t in range(self.n_estimators):
            sub = rng.choice(n, ns, replace=False) if self.subsample < 1 else np.arange(n)
            tr = _HistTree(self.max_depth, self.min_samples_leaf, self.n_bins,
                           None, np.random.default_rng(self.seed + t + 1))
            tr.fit(B[sub], resid[sub], p)
            resid = resid - self.learning_rate * tr.predict(B)
            self.trees_.append(tr)
        return self

    def predict(self, X):
        B, _ = _bin_features(X, self.n_bins, self.edges_)
        out = np.full(X.shape[0], self.base_)
        for tr in self.trees_:
            out += self.learning_rate * tr.predict(B)
        return out


# ----------------------------------------------------------------------------
# Adam optimiser (shared by the neural models)
# ----------------------------------------------------------------------------

class _Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(v) for v in params]
        self.v = [np.zeros_like(v) for v in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------

class MLPRegressor:
    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=200, batch=128,
                 l2=1e-4, patience=20, seed=0, **kw):
        self.hidden = hidden
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.l2, self.patience, self.seed = l2, patience, seed

    def _init(self, p):
        rng = np.random.default_rng(self.seed)
        dims = [p] + list(self.hidden) + [1]
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]

    def _forward(self, X):
        a = X
        cache = [a]
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(z, 0)
            cache.append(a)
        out = (a @ self.W[-1] + self.b[-1]).ravel()
        return out, cache

    def fit(self, X, y, X_val=None, y_val=None):
        n, p = X.shape
        self._init(p)
        opt = _Adam(self.W + self.b, lr=self.lr)
        rng = np.random.default_rng(self.seed)
        best, best_w, bad = np.inf, None, 0
        self.history_ = []
        for ep in range(self.epochs):
            run_loss, nb = 0.0, 0
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                xb, yb = X[idx], y[idx]
                out, cache = self._forward(xb)
                run_loss += float(np.mean((out - yb) ** 2))
                nb += 1
                m = len(idx)
                d = (2.0 / m) * (out - yb)[:, None]
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                gW[-1] = cache[-1].T @ d + self.l2 * self.W[-1]
                gb[-1] = d.sum(0)
                da = d @ self.W[-1].T
                for i in range(len(self.W) - 2, -1, -1):
                    dz = da * (cache[i + 1] > 0)
                    gW[i] = cache[i].T @ dz + self.l2 * self.W[i]
                    gb[i] = dz.sum(0)
                    da = dz @ self.W[i].T
                opt.step(gW + gb)
            tr = run_loss / max(nb, 1)
            if X_val is not None:
                v = float(np.mean((self._forward(X_val)[0] - y_val) ** 2))
                self.history_.append((ep, tr, v))
                if v < best - 1e-9:
                    best, bad = v, 0
                    best_w = ([w.copy() for w in self.W], [c.copy() for c in self.b])
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                self.history_.append((ep, tr, float('nan')))
        if X_val is not None and best_w is not None:
            self.W, self.b = best_w
        return self

    def predict(self, X):
        return self._forward(X)[0]


# ----------------------------------------------------------------------------
# Hybrid 1D-CNN -> DAFM -> RBF head  (the architecture proposed in this study)
# ----------------------------------------------------------------------------

class HybridCNNDAFMRBF:
    """1D convolution over the feature vector -> Dynamic Attention Fusion
    Module -> RBF-activated regression head.

    Attention is computed over the p FEATURE POSITIONS, not over latent
    channels. Position j is the centre of the convolution window on input
    variable j, so the attention weight alpha_j is attributable to input
    variable j (up to the kernel neighbourhood, width K). This is what makes
    .attention() interpretable at the input level; a formulation that attends
    over latent channels would not permit that mapping.

    Shapes (batch n, features p, kernel K, channels C, attention dim A,
    RBF units R):
        U     (n, p, K)   unfolded input windows
        H     (n, p, C)   conv + ReLU
        T     (n, p, A)   attention hidden state
        alpha (n, p)      softmax over feature positions, sums to 1
        z     (n, C)      attention-weighted context vector
        Phi   (n, R)      RBF activations
        out   (n,)        prediction
    """

    def __init__(self, n_channels=32, kernel=3, att_dim=32, n_rbf=16,
                 lr=1e-3, epochs=150, batch=128, l2=1e-5, patience=20,
                 seed=0, **kw):
        self.C, self.K, self.A, self.R = n_channels, kernel, att_dim, n_rbf
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.l2, self.patience, self.seed = l2, patience, seed

    # -- conv as an unfolded matrix multiply (features are 1-D, single input channel)
    def _unfold(self, X):
        n, p = X.shape
        pad = self.K // 2
        Xp = np.pad(X, ((0, 0), (pad, pad)))
        idx = np.arange(p)[:, None] + np.arange(self.K)[None, :]
        return Xp[:, idx]            # (n, p, K)

    def _init(self, p):
        rng = np.random.default_rng(self.seed)
        self.Wc = rng.normal(0, np.sqrt(2.0 / self.K), (self.K, self.C))
        self.bc = np.zeros(self.C)
        self.Wa = rng.normal(0, np.sqrt(2.0 / self.C), (self.C, self.A))
        self.ba = np.zeros(self.A)
        self.va = rng.normal(0, np.sqrt(2.0 / self.A), self.A)
        self.Wr = rng.normal(0, np.sqrt(2.0 / self.C), (self.C, self.R))
        self.br = np.zeros(self.R)
        self.log_gamma = np.zeros(self.R)
        self.Wo = rng.normal(0, np.sqrt(2.0 / self.R), self.R)
        self.bo = np.zeros(1)
        self.params = [self.Wc, self.bc, self.Wa, self.ba, self.va,
                       self.Wr, self.br, self.log_gamma, self.Wo, self.bo]

    def _forward(self, X, keep=False, U=None):
        # All contractions are expressed as matmuls so they dispatch to BLAS;
        # the equivalent einsum form is ~4x slower here.
        if U is None:
            U = self._unfold(X)                              # (n, p, K)
        Z = U @ self.Wc + self.bc                            # (n, p, C)
        H = np.maximum(Z, 0)
        # --- DAFM: additive attention over the p feature positions
        T = np.tanh(H @ self.Wa + self.ba)                   # (n, p, A)
        e = T @ self.va                                      # (n, p)
        e = e - e.max(1, keepdims=True)
        Aw = np.exp(e)
        Aw /= Aw.sum(1, keepdims=True)                       # (n, p) sums to 1
        z = np.matmul(Aw[:, None, :], H)[:, 0, :]            # (n, C)
        # --- RBF head
        P = z @ self.Wr + self.br                            # (n, R)
        g = np.exp(self.log_gamma)
        Phi = np.exp(-g * P ** 2)
        out = Phi @ self.Wo + self.bo
        if keep:
            self._c = dict(U=U, Z=Z, H=H, T=T, Aw=Aw, z=z, P=P, Phi=Phi, g=g)
        return out.ravel()

    def _backward(self, y):
        c = self._c
        n = len(y)
        d_out = (2.0 / n) * (self._last - y)                 # (n,)
        gWo = c['Phi'].T @ d_out + self.l2 * self.Wo
        gbo = np.array([d_out.sum()])
        dPhi = d_out[:, None] * self.Wo[None, :]             # (n, R)
        dP = dPhi * c['Phi'] * (-c['g']) * 2.0 * c['P']
        glog = (dPhi * c['Phi'] * (-c['g'] * c['P'] ** 2)).sum(0)
        gWr = c['z'].T @ dP + self.l2 * self.Wr
        gbr = dP.sum(0)
        dz = dP @ self.Wr.T                                  # (n, C)

        # z = sum_p alpha_p * H_p
        dAw = np.matmul(c['H'], dz[:, :, None])[:, :, 0]     # (n, p)
        dH = c['Aw'][:, :, None] * dz[:, None, :]            # (n, p, C)

        # softmax over positions
        de = c['Aw'] * (dAw - (dAw * c['Aw']).sum(1, keepdims=True))

        T = c['T']
        nA = T.shape[2]
        gva = T.reshape(-1, nA).T @ de.reshape(-1)
        dT = de[:, :, None] * self.va[None, None, :] * (1 - T ** 2)
        H2 = c['H'].reshape(-1, c['H'].shape[2])
        gWa = H2.T @ dT.reshape(-1, nA) + self.l2 * self.Wa
        gba = dT.sum((0, 1))
        dH = dH + dT @ self.Wa.T

        dZ = dH * (c['Z'] > 0)
        nC = dZ.shape[2]
        gWc = c['U'].reshape(-1, self.K).T @ dZ.reshape(-1, nC) + self.l2 * self.Wc
        gbc = dZ.sum((0, 1))
        return [gWc, gbc, gWa, gba, gva, gWr, gbr, glog, gWo, gbo]

    def fit(self, X, y, X_val=None, y_val=None):
        self._init(X.shape[1])
        opt = _Adam(self.params, lr=self.lr)
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        # unfold once rather than per minibatch / per epoch
        U_all = self._unfold(X)
        U_val = self._unfold(X_val) if X_val is not None else None
        best, best_p, bad = np.inf, None, 0
        self.history_ = []
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            run_loss, nb = 0.0, 0
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                self._last = self._forward(None, keep=True, U=U_all[idx])
                run_loss += float(np.mean((self._last - y[idx]) ** 2))
                nb += 1
                opt.step(self._backward(y[idx]))
            # running mean of minibatch losses, so no extra full-train pass
            tr = run_loss / max(nb, 1)
            if X_val is not None:
                v = float(np.mean((self._forward(None, U=U_val) - y_val) ** 2))
                self.history_.append((ep, tr, v))
                if v < best - 1e-9:
                    best, bad = v, 0
                    best_p = [p.copy() for p in self.params]
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                self.history_.append((ep, tr, float('nan')))
        if X_val is not None and best_p is not None:
            for p, q in zip(self.params, best_p):
                p[...] = q
        return self

    def predict(self, X):
        return self._forward(X)

    def attention(self, X):
        """Mean attention weight per input feature position, shape (p,).

        Sums to 1 across features. Attributable to input variables because the
        DAFM attends over feature positions (see class docstring).
        """
        self._forward(X, keep=True)
        return self._c['Aw'].mean(0)


MODELS = {
    'Ridge': RidgeRegressor,
    'KNN': KNNRegressor,
    'RandomForest': HistRandomForest,
    'GradientBoosting': HistGradientBoosting,
    'MLP': MLPRegressor,
    'HybridCNN': HybridCNNDAFMRBF,
}
