"""
ml_dual.py -- Dual-branch attention architecture for soil nutrient prediction.

This is the architecture the manuscript proposes, in the form its abstract
always described but the earlier implementation did not deliver:

    spectral branch : 1D-CNN over the MIR absorbance sequence
                      (wavenumbers ARE ordered and locally correlated, so a
                       convolution is the appropriate inductive bias here --
                       unlike an unordered tabular covariate vector)

    covariate branch: dense encoder over environmental covariates
                      (satellite, terrain, climate)

    DAFM            : attention over spectral positions produces a spectral
                      context vector; a second, gate-level attention weighs
                      the two branches against each other per sample

    RBF head        : radial-basis-activated regression head, fully
                      differentiable, trained end-to-end

Every gradient in this file is verified against central finite differences by
tests at the bottom (`python3 ml_dual.py`).

Shapes (batch n, spectral length S, spectral channels C, covariates P,
covariate embedding D, attention dim A, RBF units R):
    Xs    (n, S)      spectral input
    H     (n, S', C)  conv feature map  (S' = S after striding/pooling)
    alpha (n, S')     attention over spectral positions, sums to 1
    zs    (n, C)      spectral context
    zc    (n, D)      covariate embedding
    g     (n, 2)      branch gate, sums to 1
    z     (n, C+D)    gated fusion
    Phi   (n, R)      RBF activations
    out   (n,)
"""

import numpy as np

from ml import _Adam           # reuse the verified Adam implementation


# ----------------------------------------------------------------- helpers

def _conv1d_forward(X, W, b, stride):
    """X (n, L, Cin) -> (n, Lout, Cout) via im2col + matmul.

    Returns the output and the unfolded matrix, which the backward pass needs.
    """
    n, L, cin = X.shape
    k, _, cout = W.shape
    Lout = (L - k) // stride + 1
    idx = (np.arange(Lout)[:, None] * stride + np.arange(k)[None, :])   # (Lout, k)
    cols = X[:, idx, :]                                                 # (n,Lout,k,cin)
    cols = cols.reshape(n, Lout, k * cin)
    out = cols @ W.reshape(k * cin, cout) + b
    return out, cols, idx


def _conv1d_backward(dOut, cols, idx, W, X_shape, stride):
    """Gradients for _conv1d_forward."""
    n, L, cin = X_shape
    k, _, cout = W.shape
    Lout = dOut.shape[1]
    gW = (cols.reshape(-1, k * cin).T @ dOut.reshape(-1, cout)).reshape(k, cin, cout)
    gb = dOut.sum((0, 1))
    dcols = (dOut.reshape(-1, cout) @ W.reshape(k * cin, cout).T)
    dcols = dcols.reshape(n, Lout, k, cin)
    dX = np.zeros((n, L, cin))
    # scatter-add the overlapping windows back onto the input
    for j in range(k):
        np.add.at(dX, (slice(None), idx[:, j], slice(None)), dcols[:, :, j, :])
    return gW, gb, dX


def _maxpool_forward(X, size):
    n, L, c = X.shape
    Lout = L // size
    Xt = X[:, :Lout * size, :].reshape(n, Lout, size, c)
    am = Xt.argmax(2)
    out = np.take_along_axis(Xt, am[:, :, None, :], axis=2)[:, :, 0, :]
    return out, am, L


def _maxpool_backward(dOut, am, size, L, c):
    n, Lout, _ = dOut.shape
    dXt = np.zeros((n, Lout, size, c))
    np.put_along_axis(dXt, am[:, :, None, :], dOut[:, :, None, :], axis=2)
    dX = np.zeros((n, L, c))
    dX[:, :Lout * size, :] = dXt.reshape(n, Lout * size, c)
    return dX


# ------------------------------------------------------------------- model

class DualBranchDAFM:
    def __init__(self, n_spec, n_cov,
                 conv_channels=(16, 32), kernels=(9, 7), strides=(3, 1),
                 pool=2, cov_hidden=32, att_dim=32, n_rbf=16,
                 lr=1e-3, epochs=120, batch=64, l2=1e-5, patience=15,
                 seed=0, **kw):
        self.S, self.P = n_spec, n_cov
        self.CC, self.KK, self.SS = conv_channels, kernels, strides
        self.pool, self.D, self.A, self.R = cov_hidden, cov_hidden, att_dim, n_rbf
        self.pool = pool
        self.Dh = cov_hidden
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.l2, self.patience, self.seed = l2, patience, seed
        self._init()

    def _init(self):
        r = np.random.default_rng(self.seed)
        k1, k2 = self.KK
        c1, c2 = self.CC
        self.W1 = r.normal(0, np.sqrt(2.0 / (k1 * 1)), (k1, 1, c1))
        self.b1 = np.zeros(c1)
        self.W2 = r.normal(0, np.sqrt(2.0 / (k2 * c1)), (k2, c1, c2))
        self.b2 = np.zeros(c2)
        # DAFM over spectral positions
        self.Wa = r.normal(0, np.sqrt(2.0 / c2), (c2, self.A))
        self.ba = np.zeros(self.A)
        self.va = r.normal(0, np.sqrt(2.0 / self.A), self.A)
        # covariate branch
        self.Wc = r.normal(0, np.sqrt(2.0 / max(self.P, 1)), (max(self.P, 1), self.Dh))
        self.bc = np.zeros(self.Dh)
        # branch gate: scores the spectral context and the covariate embedding
        self.Wg = r.normal(0, np.sqrt(2.0 / (c2 + self.Dh)), (c2 + self.Dh, 2))
        self.bg = np.zeros(2)
        # RBF head
        F = c2 + self.Dh
        self.Wr = r.normal(0, np.sqrt(2.0 / F), (F, self.R))
        self.br = np.zeros(self.R)
        self.log_gamma = np.zeros(self.R)
        self.Wo = r.normal(0, np.sqrt(2.0 / self.R), self.R)
        self.bo = np.zeros(1)
        self.params = [self.W1, self.b1, self.W2, self.b2,
                       self.Wa, self.ba, self.va,
                       self.Wc, self.bc, self.Wg, self.bg,
                       self.Wr, self.br, self.log_gamma, self.Wo, self.bo]

    # ------------------------------------------------------------ forward
    def _forward(self, Xs, Xc, keep=False):
        n = Xs.shape[0]
        A0 = Xs[:, :, None]                                   # (n, S, 1)
        Z1, cols1, idx1 = _conv1d_forward(A0, self.W1, self.b1, self.SS[0])
        H1 = np.maximum(Z1, 0)
        P1, am1, L1 = _maxpool_forward(H1, self.pool)
        Z2, cols2, idx2 = _conv1d_forward(P1, self.W2, self.b2, self.SS[1])
        H2 = np.maximum(Z2, 0)                                # (n, S', C)

        # DAFM over spectral positions
        T = np.tanh(H2 @ self.Wa + self.ba)                   # (n, S', A)
        e = T @ self.va
        e = e - e.max(1, keepdims=True)
        Aw = np.exp(e)
        Aw /= Aw.sum(1, keepdims=True)                        # (n, S')
        zs = np.matmul(Aw[:, None, :], H2)[:, 0, :]           # (n, C)

        # covariate branch
        zc = np.maximum(Xc @ self.Wc + self.bc, 0)            # (n, D)

        # branch gate
        cat = np.concatenate([zs, zc], axis=1)                # (n, C+D)
        gs = cat @ self.Wg + self.bg                          # (n, 2)
        gs = gs - gs.max(1, keepdims=True)
        G = np.exp(gs)
        G /= G.sum(1, keepdims=True)
        C = zs.shape[1]
        z = np.concatenate([zs * G[:, :1], zc * G[:, 1:]], axis=1)

        Pp = z @ self.Wr + self.br
        g = np.exp(self.log_gamma)
        Phi = np.exp(-g * Pp ** 2)
        out = (Phi @ self.Wo + self.bo).ravel()
        if keep:
            self._c = dict(A0=A0, Z1=Z1, H1=H1, cols1=cols1, idx1=idx1, L1=L1,
                           P1=P1, am1=am1, Z2=Z2, H2=H2, cols2=cols2, idx2=idx2,
                           T=T, Aw=Aw, zs=zs, zc=zc, Xc=Xc, cat=cat, G=G,
                           z=z, Pp=Pp, Phi=Phi, gam=g, C=C)
        return out

    # ----------------------------------------------------------- backward
    def _backward(self, y):
        c = self._c
        n = len(y)
        d_out = (2.0 / n) * (self._last - y)

        gWo = c['Phi'].T @ d_out + self.l2 * self.Wo
        gbo = np.array([d_out.sum()])
        dPhi = d_out[:, None] * self.Wo[None, :]
        dPp = dPhi * c['Phi'] * (-c['gam']) * 2.0 * c['Pp']
        glog = (dPhi * c['Phi'] * (-c['gam'] * c['Pp'] ** 2)).sum(0)
        gWr = c['z'].T @ dPp + self.l2 * self.Wr
        gbr = dPp.sum(0)
        dz = dPp @ self.Wr.T                                   # (n, C+D)

        C = c['C']
        dzs = dz[:, :C] * c['G'][:, :1]
        dzc = dz[:, C:] * c['G'][:, 1:]
        dG = np.stack([(dz[:, :C] * c['zs']).sum(1),
                       (dz[:, C:] * c['zc']).sum(1)], axis=1)   # (n, 2)
        # softmax backward for the gate
        dgs = c['G'] * (dG - (dG * c['G']).sum(1, keepdims=True))
        gWg = c['cat'].T @ dgs + self.l2 * self.Wg
        gbg = dgs.sum(0)
        dcat = dgs @ self.Wg.T
        dzs = dzs + dcat[:, :C]
        dzc = dzc + dcat[:, C:]

        # covariate branch
        dpre_c = dzc * (c['zc'] > 0)
        gWc = c['Xc'].T @ dpre_c + self.l2 * self.Wc
        gbc = dpre_c.sum(0)

        # DAFM backward
        H2 = c['H2']
        dAw = np.matmul(H2, dzs[:, :, None])[:, :, 0]          # (n, S')
        dH2 = c['Aw'][:, :, None] * dzs[:, None, :]
        de = c['Aw'] * (dAw - (dAw * c['Aw']).sum(1, keepdims=True))
        T = c['T']
        nA = T.shape[2]
        gva = T.reshape(-1, nA).T @ de.reshape(-1)
        dT = de[:, :, None] * self.va[None, None, :] * (1 - T ** 2)
        gWa = H2.reshape(-1, H2.shape[2]).T @ dT.reshape(-1, nA) + self.l2 * self.Wa
        gba = dT.sum((0, 1))
        dH2 = dH2 + dT @ self.Wa.T

        # conv2
        dZ2 = dH2 * (c['Z2'] > 0)
        gW2, gb2, dP1 = _conv1d_backward(dZ2, c['cols2'], c['idx2'], self.W2,
                                         c['P1'].shape, self.SS[1])
        gW2 = gW2 + self.l2 * self.W2
        # pool
        dH1 = _maxpool_backward(dP1, c['am1'], self.pool, c['L1'], c['H1'].shape[2])
        # conv1
        dZ1 = dH1 * (c['Z1'] > 0)
        gW1, gb1, _ = _conv1d_backward(dZ1, c['cols1'], c['idx1'], self.W1,
                                       c['A0'].shape, self.SS[0])
        gW1 = gW1 + self.l2 * self.W1

        return [gW1, gb1, gW2, gb2, gWa, gba, gva,
                gWc, gbc, gWg, gbg, gWr, gbr, glog, gWo, gbo]

    # ---------------------------------------------------------------- fit
    def fit(self, Xs, Xc, y, Xs_val=None, Xc_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n = Xs.shape[0]
        opt = _Adam(self.params, lr=self.lr)
        best, best_p, bad = np.inf, None, 0
        self.history_ = []
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            run, nb = 0.0, 0
            for s in range(0, n, self.batch):
                i = perm[s:s + self.batch]
                self._last = self._forward(Xs[i], Xc[i], keep=True)
                run += float(np.mean((self._last - y[i]) ** 2)); nb += 1
                opt.step(self._backward(y[i]))
            tr = run / max(nb, 1)
            if Xs_val is not None:
                v = float(np.mean((self._forward(Xs_val, Xc_val) - y_val) ** 2))
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
        if best_p is not None:
            for p, q in zip(self.params, best_p):
                p[...] = q
        return self

    def predict(self, Xs, Xc):
        return self._forward(Xs, Xc)

    def spectral_attention(self, Xs, Xc):
        """Mean attention over spectral positions, shape (S',). Maps back to
        wavenumber ranges, so it is interpretable as which absorbance regions
        the model relies on."""
        self._forward(Xs, Xc, keep=True)
        return self._c['Aw'].mean(0)

    def branch_gate(self, Xs, Xc):
        """Mean weight given to [spectral, covariate] branches. Quantifies how
        much of the prediction each modality actually supplies."""
        self._forward(Xs, Xc, keep=True)
        return self._c['G'].mean(0)


# ------------------------------------------------------------------ tests
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n, S, P = 20, 60, 5
    Xs = rng.normal(size=(n, S))
    Xc = rng.normal(size=(n, P))
    y = rng.normal(size=n)

    m = DualBranchDAFM(S, P, conv_channels=(4, 6), kernels=(5, 3),
                       strides=(2, 1), pool=2, cov_hidden=4, att_dim=4,
                       n_rbf=3, l2=0.0, seed=1)
    m._last = m._forward(Xs, Xc, keep=True)
    grads = m._backward(y)

    def loss():
        return float(np.mean((m._forward(Xs, Xc) - y) ** 2))

    names = ['W1', 'b1', 'W2', 'b2', 'Wa', 'ba', 'va', 'Wc', 'bc',
             'Wg', 'bg', 'Wr', 'br', 'log_gamma', 'Wo', 'bo']
    worst, bad = 0.0, []
    for p, g, nm in zip(m.params, grads, names):
        f, gf = p.ravel(), np.asarray(g).ravel()
        for i in rng.choice(f.size, min(5, f.size), replace=False):
            o = f[i]
            f[i] = o + 1e-6; lp = loss()
            f[i] = o - 1e-6; lm = loss()
            f[i] = o
            num = (lp - lm) / 2e-6
            rel = abs(num - gf[i]) / max(1e-8, abs(num), abs(gf[i]))
            worst = max(worst, rel)
            if rel > 1e-4:
                bad.append((nm, i, num, gf[i]))
    print(f'gradient check: worst relative error {worst:.2e} -> '
          f'{"PASS" if not bad else "FAIL"}')
    for b in bad[:8]:
        print('   mismatch', b)

    a = m.spectral_attention(Xs, Xc)
    gsum = m.branch_gate(Xs, Xc)
    print(f'spectral attention shape {a.shape}, sums to {a.sum():.6f}')
    print(f'branch gate [spectral, covariate] = {gsum.round(3)}, '
          f'sums to {gsum.sum():.6f}')
