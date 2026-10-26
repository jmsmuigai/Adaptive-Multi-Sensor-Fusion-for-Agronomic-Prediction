"""
classical.py -- library implementations (scikit-learn, LightGBM, Cubist) of the
benchmark models, plus memory-based learning (local PLS), which has no
maintained Python package and is implemented here following Ramirez-Lopez et
al. (2013) / the 'resemble' approach (neighbours in PC space, weighted average
of local PLS predictions over a range of components).

Each entry: (constructor(params) -> estimator, list_of_param_dicts, search)
search = 'kfold' (3 group-disjoint inner folds) or 'split' (one inner
group-disjoint validation split), chosen by cost.
"""
import os
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

warnings.filterwarnings('ignore')


class PLS:
    def __init__(self, n_components=10):
        self.k = n_components

    def fit(self, X, y):
        self.m = PLSRegression(n_components=min(self.k, X.shape[1], len(X) - 1),
                               scale=False).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X).ravel()


class MBL:
    """Memory-based learning: for each query, the k nearest training spectra
    (Mahalanobis distance in the space of the first PCs, i.e. Euclidean on
    whitened scores) are used to fit local PLS models with 3..max_comp
    components; predictions are averaged with weights inversely proportional to
    each local model's in-sample RMSE x number of components ('wapls')."""

    def __init__(self, k=100, max_comp=12, n_pc=20):
        self.k, self.max_comp, self.n_pc = k, max_comp, n_pc

    def fit(self, X, y):
        self.X, self.y = X, np.asarray(y, float)
        self.pca = PCA(n_components=min(self.n_pc, X.shape[1]), whiten=True).fit(X)
        self.S = self.pca.transform(X)
        return self

    def predict(self, Xq):
        Sq = self.pca.transform(Xq)
        out = np.empty(len(Xq))
        k = min(self.k, len(self.X))
        for i, s in enumerate(Sq):
            d = ((self.S - s) ** 2).sum(1)
            nn_ = np.argpartition(d, k - 1)[:k]
            Xl, yl = self.X[nn_], self.y[nn_]
            xm, ym = Xl.mean(0), yl.mean()
            Xc, yc = Xl - xm, yl - ym
            # NIPALS PLS1, collect predictions for 1..max_comp
            E, f = Xc.copy(), yc.copy()
            W, P, Q = [], [], []
            preds, wts = [], []
            xq = Xq[i] - xm
            for a in range(self.max_comp):
                w = E.T @ f
                nw = np.linalg.norm(w)
                if nw < 1e-12:
                    break
                w /= nw
                t = E @ w
                tt = t @ t
                if tt < 1e-12:
                    break
                p = E.T @ t / tt
                q = (f @ t) / tt
                E -= np.outer(t, p)
                f = f - q * t
                W.append(w); P.append(p); Q.append(q)
                if a + 1 >= 3:
                    Wm, Pm, Qm = np.array(W).T, np.array(P).T, np.array(Q)
                    B = Wm @ np.linalg.pinv(Pm.T @ Wm) @ Qm
                    rmse = np.sqrt(np.mean(f ** 2)) + 1e-9
                    preds.append(xq @ B + ym)
                    wts.append(1.0 / (rmse * (a + 1)))
            out[i] = np.average(preds, weights=wts) if preds else ym
        return out


class CubistWrap:
    def __init__(self, n_committees=5, neighbors=None, n_pc=30):
        self.c, self.nb, self.n_pc = n_committees, neighbors, n_pc

    def fit(self, X, y):
        from cubist import Cubist
        # Cubist on the leading principal components, the common practice for
        # high-dimensional spectra
        self.pca = PCA(n_components=min(self.n_pc, X.shape[1])).fit(X)
        self.m = Cubist(n_committees=self.c, neighbors=self.nb, verbose=False)
        self.m.fit(self.pca.transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.pca.transform(X))


class GPR:
    def fit(self, X, y):
        L = X.shape[1]
        k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=np.sqrt(L),
                                                    length_scale_bounds=(1, 1e4)) \
            + WhiteKernel(0.1, (1e-4, 1.0))
        self.m = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0,
                                          random_state=0).fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.m.predict(X, return_std=return_std)


class LGBM:
    def __init__(self, **p):
        self.p = p

    def fit(self, X, y, Xv=None, yv=None):
        import lightgbm as lgb
        p = dict(n_estimators=2000, learning_rate=0.03, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.3, min_child_samples=10,
                 verbose=-1, n_jobs=1, random_state=0)
        p.update(self.p)
        self.m = lgb.LGBMRegressor(**p)
        if Xv is not None:
            self.m.fit(X, y, eval_set=[(Xv, yv)],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
            self.best_iter = self.m.best_iteration_
        else:
            self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def grid(**kw):
    import itertools
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


L_HINT = 425
MODELS = {
    'PLSR': (lambda p: PLS(**p), grid(n_components=[4, 8, 12, 16, 20, 25, 30]), 'kfold'),
    'Ridge': (lambda p: Ridge(**p), grid(alpha=list(np.logspace(-1, 4, 11))), 'kfold'),
    'SVR': (lambda p: SVR(kernel='rbf', **p),
            grid(C=[1, 10, 100], gamma=[0.3 / L_HINT, 1 / L_HINT, 3 / L_HINT], epsilon=[0.1]), 'kfold'),
    'KRR': (lambda p: KernelRidge(kernel='rbf', **p),
            grid(alpha=[0.01, 0.1, 1.0], gamma=[0.3 / L_HINT, 1 / L_HINT, 3 / L_HINT]), 'kfold'),
    'GPR': (lambda p: GPR(), [{}], 'none'),
    'Cubist': (lambda p: CubistWrap(**p), grid(n_committees=[5, 20], neighbors=[None, 5]), 'split'),
    'MBL': (lambda p: MBL(**p), grid(k=[75, 150, 300]), 'split'),
    'RF': (lambda p: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=0,
                                           min_samples_leaf=3, **p),
           grid(max_features=[0.33]), 'none'),
    'LightGBM': (lambda p: LGBM(**p), grid(num_leaves=[7, 31]), 'split'),
}
