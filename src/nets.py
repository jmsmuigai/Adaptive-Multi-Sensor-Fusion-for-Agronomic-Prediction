"""
nets.py -- PyTorch implementations of the neural models used in the revision.

HybridNet is the proposed architecture with switchable components so that
every ablation requested by the reviewer is the SAME code path with exactly
one component replaced:

    pool  : 'attn' (DAFM additive attention over spectral positions) | 'mean' | 'max'
    fuse  : 'gate' (learned softmax modality gate) | 'concat' | 'none' (spectra only)
    head  : 'rbf'   RBF layer with LEARNED CENTRES  phi_i = exp(-gamma_i ||u - c_i||^2)
            'gauss' round-1 head: Gaussian activation centred at zero exp(-gamma p^2)
            'linear' a single linear output layer (identical network otherwise)
            'mlp'   dense ReLU layer + linear output
"""
import copy

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


class HybridNet(nn.Module):
    def __init__(self, L, P, c1=16, c2=32, k1=11, s1=2, k2=7, pool_size=2,
                 att_dim=32, D=8, R=16, d_rbf=8, dropout=0.0,
                 pool='attn', fuse='gate', head='rbf', n_out=1, pos=True, bn=False, heads=1, zdim=64, **kw):
        super().__init__()
        self.use_pos = pos
        self.pool, self.fuse, self.head = pool, fuse, head
        if bn:
            self.conv = nn.Sequential(
                nn.Conv1d(1, c1, k1, stride=s1), nn.BatchNorm1d(c1), nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Conv1d(c1, c2, k2), nn.BatchNorm1d(c2), nn.ReLU())
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(1, c1, k1, stride=s1), nn.ReLU(), nn.MaxPool1d(pool_size),
                nn.Conv1d(c1, c2, k2), nn.ReLU())
        self.rf = k1 + (pool_size - 1) * s1 + (k2 - 1) * s1 * pool_size  # input bins
        self.jump = s1 * pool_size
        with torch.no_grad():
            p_len = self.conv(torch.zeros(1, 1, L)).shape[-1]
        self.p_len = p_len
        # learned positional embedding: global pooling / attention over
        # positions is otherwise permutation-invariant and blind to WHERE in
        # the spectrum a feature occurs
        if pos:
            self.pe = nn.Parameter(torch.randn(p_len, c2) * 0.1)
        self.heads = heads if pool in ('attn', 'reweight') else 1
        if pool in ('attn', 'reweight'):
            self.Wa = nn.Linear(c2, att_dim)
            self.va = nn.Linear(att_dim, self.heads, bias=False)
        C2 = c2 * self.heads
        if pool in ('reweight', 'flatten'):
            # attention RE-WEIGHTS positions (no pooling); the weighted map is
            # flattened and projected, so positional detail is retained
            self.zproj = nn.Sequential(nn.Linear(p_len * c2 * self.heads, zdim), nn.ReLU())
            C2 = zdim
        F = C2
        if fuse != 'none':
            self.enc = nn.Sequential(nn.Linear(P, D), nn.ReLU())
            F = C2 + D
            if fuse == 'gate':
                self.gate = nn.Linear(C2 + D, 2)
        self.drop = nn.Dropout(dropout)
        if head == 'rbf':
            self.proj = nn.Linear(F, d_rbf)
            self.centres = nn.Parameter(torch.randn(R, d_rbf) * 0.5)
            self.log_gamma = nn.Parameter(torch.zeros(R))
            self.out = nn.Linear(R, n_out)
        elif head == 'gauss':
            self.proj = nn.Linear(F, R)
            self.log_gamma = nn.Parameter(torch.zeros(R))
            self.out = nn.Linear(R, n_out)
        elif head == 'linear':
            self.out = nn.Linear(F, n_out)
        elif head == 'mlp':
            self.hid = nn.Sequential(nn.Linear(F, R), nn.ReLU())
            self.out = nn.Linear(R, n_out)
        self._store = False

    def forward(self, x, a):
        H = self.conv(x[:, None, :]).transpose(1, 2)          # (n, p, C)
        if self.use_pos:
            H = H + self.pe[None]
        if self.pool == 'attn':
            e = self.va(torch.tanh(self.Wa(H)))               # (n, p, heads)
            al = torch.softmax(e, 1)
            zs = torch.einsum('nph,npc->nhc', al, H).flatten(1)  # (n, heads*C)
            alpha = al.mean(2)
        elif self.pool == 'reweight':
            e = self.va(torch.tanh(self.Wa(H)))
            al = torch.softmax(e, 1)                            # (n, p, heads)
            Hw = torch.einsum('nph,npc->nphc', al * H.shape[1], H).flatten(1)
            zs = self.zproj(Hw)
            alpha = al.mean(2)
        elif self.pool == 'flatten':
            alpha = None; zs = self.zproj(H.flatten(1))
        elif self.pool == 'mean':
            alpha = None; zs = H.mean(1)
        else:
            alpha = None; zs = H.max(1).values
        G = None
        if self.fuse == 'none':
            z = zs
        else:
            zc = self.enc(a)
            if self.fuse == 'gate':
                G = torch.softmax(self.gate(torch.cat([zs, zc], 1)), 1)
                z = torch.cat([zs * G[:, :1], zc * G[:, 1:]], 1)
            else:
                z = torch.cat([zs, zc], 1)
        z = self.drop(z)
        if self.head == 'rbf':
            u = self.proj(z)
            d2 = ((u[:, None, :] - self.centres[None]) ** 2).sum(-1)
            phi = torch.exp(-torch.exp(self.log_gamma) * d2)
            y = self.out(phi)
        elif self.head == 'gauss':
            p = self.proj(z)
            phi = torch.exp(-torch.exp(self.log_gamma) * p ** 2)
            y = self.out(phi)
        elif self.head == 'linear':
            phi = z; y = self.out(z)
        else:
            phi = self.hid(z); y = self.out(phi)
        if self._store:
            self.cache = dict(alpha=alpha, G=G, phi=phi, z=z)
        return y if y.shape[1] > 1 else y[:, 0]

    def wo_l1(self):
        return float(self.out.weight.abs().sum().item())


class MLPNet(nn.Module):
    def __init__(self, L, P, h1=128, h2=64, dropout=0.1, n_out=1, **kw):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(L + P, h1), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, n_out))

    def forward(self, x, a):
        y = self.f(torch.cat([x, a], 1))
        return y if y.shape[1] > 1 else y[:, 0]


class CNNNet(nn.Module):
    """Plain 1D-CNN in the style of Padarian et al. (2019) / Ng et al. (2019):
    conv-pool blocks, flatten, dense layer; depth concatenated before the dense
    layer. n_out > 1 gives the multi-output variant."""

    def __init__(self, L, P, c1=16, c2=32, k1=11, s1=2, k2=7, h=64, dropout=0.2,
                 n_out=1, **kw):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, k1, stride=s1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, k2), nn.ReLU(), nn.MaxPool1d(2))
        with torch.no_grad():
            nf = self.conv(torch.zeros(1, 1, L)).numel()
        self.head = nn.Sequential(nn.Linear(nf + P, h), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(h, n_out))

    def forward(self, x, a):
        f = self.conv(x[:, None, :]).flatten(1)
        y = self.head(torch.cat([f, a], 1))
        return y if y.shape[1] > 1 else y[:, 0]


ARCH = {'Hybrid': HybridNet, 'MLP': MLPNet, 'CNN': CNNNet, 'MOCNN': CNNNet}


def train_net(kind, cfg, Xtr, Atr, ytr, Xva, Ava, yva, seed, max_epochs=200,
              patience=15, batch=64, min_epochs=25):
    """Adam + early stopping on a GROUP-AWARE inner validation split.
    y may be (n,) or (n, T) with NaN for missing (multi-output, masked loss)."""
    torch.manual_seed(seed); np.random.seed(seed)
    L, P = Xtr.shape[1], Atr.shape[1]
    n_out = 1 if ytr.ndim == 1 else ytr.shape[1]
    net = ARCH[kind](L, P, n_out=n_out, **cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get('lr', 1e-3),
                           weight_decay=cfg.get('wd', 1e-4))
    T = lambda v: torch.as_tensor(v, dtype=torch.float32)
    Xt, At, yt = T(Xtr), T(Atr), T(ytr)
    Xv, Av, yv = T(Xva), T(Ava), T(yva)

    def loss_fn(pred, y):
        if y.ndim == 1:
            return ((pred - y) ** 2).mean()
        m = ~torch.isnan(y)
        return ((torch.nan_to_num(pred - y) ** 2) * m).sum() / m.sum()

    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, ep_best = np.inf, None, 0, 0
    n = len(Xt)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            opt.zero_grad()
            l = loss_fn(net(Xt[i], At[i]), yt[i])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xv, Av), yv))
        if v < best - 1e-6:
            best, bad, ep_best = v, 0, ep
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience and ep >= min_epochs:
                break
    net.load_state_dict(best_state)
    net.eval()
    net.val_loss = best
    net.best_epoch = ep_best
    return net


def predict(net, X, A):
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float32),
                   torch.as_tensor(A, dtype=torch.float32)).numpy()
