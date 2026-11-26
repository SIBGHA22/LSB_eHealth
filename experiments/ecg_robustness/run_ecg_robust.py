# Standalone ECG robustness study: robust aggregators x stronger attacks.
# Sections: (1) setup = notebook cells 1-2, (2) model/utilities = notebook cell 3 unchanged,
# (3) new cells A-E.

# ======================= (1) setup =======================
# Local (non-Colab) version of cells 1-2 of LSB_EHEALTH_MEDICAL_TRACK_FINAL.ipynb.
# Same data, preprocessing, split and settings; only the Colab/Drive lines are replaced.
import os, sys, json, math, time, random, zipfile, urllib.request
from pathlib import Path
from collections import OrderedDict
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

def display(*objs, **kw):
    for o in objs:
        if isinstance(o, pd.DataFrame):
            with pd.option_context("display.max_columns", 50, "display.width", 220):
                print(o.to_string())
        else:
            print(o)

SMOKE = os.environ.get("LSB_SMOKE", "0") == "1"
RESULTS = Path(os.environ.get("LSB_RESULTS", Path(__file__).resolve().parent.parent / "results"))
DRIVE = RESULTS / "ecg_robustness"; DRIVE.mkdir(parents=True, exist_ok=True)   # stands in for MyDrive
CACHE = Path(os.environ.get("LSB_DATA_CACHE", RESULTS / "data_cache")) / "physionet2017"; CACHE.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEV = torch.device("cuda"); print("GPU:", torch.cuda.get_device_name(0), flush=True)
else:
    DEV = torch.device("cpu"); print("WARNING: no CUDA GPU found - this will be very slow.", flush=True)

TASK         = 'binary_af'
SEEDS        = [42, 43, 44, 45, 46, 47]
FS           = 100
MAX_LEN      = 6000
CROP         = 1000
EVAL_CROPS   = 5
N_CLIENTS    = 10
ROUNDS       = 40
LOCAL_EPOCHS = 2
BATCH        = 128
LR           = 2e-3
WD           = 1e-4
SERVER_MOM   = 0.5
DIRICHLET_A  = 0.5
BASE_EVAL_EVERY = 5   # evaluation schedule used for the paper's tables
EVAL_EVERY   = int(os.environ.get("ECG_EVAL_EVERY", str(BASE_EVAL_EVERY)))   # 1 = log every round (convergence figure)
# Extra evaluations (EVAL_EVERY < 5) run under a forked RNG, so they do not change training or the final results.
CLIP_NORM    = 5.0
WEIGHT_CAP   = 2.0
SPARSIFY     = 0.30
RESIDUAL_CAP = 1.0
FILTER_ON_CLEAN = False
ADV_FRACTION = 0.20
POISON_RATE  = 0.30
FGSM_EPS     = 0.01
if SMOKE:
    SEEDS, ROUNDS, EVAL_EVERY = [42], 2, 1

SYNTH = os.environ.get("LSB_SYNTHETIC_DATA", "0") == "1"   # developer self-test only, never for results
LABELS = ['N', 'A', 'O', '~']
if SYNTH:
    _r = np.random.default_rng(0); n = 3000
    SIG = _r.normal(size=(n, 1500)).astype(np.float32); LEN = np.full(n, 1500, np.int32)
    Y4 = np.where(_r.random(n) < 0.12, 1, 0).astype(np.int64)
    SIG[Y4 == 1] += np.sin(np.arange(1500) / 3)[None, :].astype(np.float32)
    MAX_LEN = 1500
else:
    ZIP_URL  = 'https://physionet.org/files/challenge-2017/1.0.0/training2017.zip'
    ZIP_PATH = CACHE/'training2017.zip'
    RAW_DIR  = CACHE/'training2017'
    if not RAW_DIR.exists():
        if not ZIP_PATH.exists():
            print('downloading PhysioNet 2017 (one time)...', flush=True); t0 = time.perf_counter()
            urllib.request.urlretrieve(ZIP_URL, ZIP_PATH); print(f'  {time.perf_counter()-t0:.0f}s')
        with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(CACHE)
    NPZ = CACHE/f'physio2017_fs{FS}_max{MAX_LEN}.npz'
    if NPZ.exists():
        d = np.load(NPZ); SIG, LEN, Y4 = d['SIG'], d['LEN'], d['Y']; print('cached:', SIG.shape)
    else:
        from scipy.io import loadmat
        from scipy.signal import resample_poly
        ref = pd.read_csv(RAW_DIR/'REFERENCE.csv', header=None, names=['record', 'label'])
        SIG = np.zeros((len(ref), MAX_LEN), dtype=np.float32)
        LEN = np.zeros(len(ref), dtype=np.int32); Y4 = np.zeros(len(ref), dtype=np.int64)
        t0 = time.perf_counter()
        for i, row in ref.iterrows():
            s = loadmat(RAW_DIR/f"{row['record']}.mat")['val'].astype(np.float32).ravel()
            s = resample_poly(s, FS, 300).astype(np.float32)
            s = (s - np.median(s)) / (np.percentile(np.abs(s-np.median(s)), 95) + 1e-6)
            s = np.clip(s, -8, 8)
            if len(s) < CROP: s = np.tile(s, int(np.ceil(CROP/len(s))))
            n = min(len(s), MAX_LEN)
            SIG[i, :n] = s[:n]; LEN[i] = n; Y4[i] = LABELS.index(row['label'])
            if i % 1500 == 0: print(f'  {i}/{len(ref)}  {time.perf_counter()-t0:.0f}s', flush=True)
        np.savez_compressed(NPZ, SIG=SIG, LEN=LEN, Y=Y4); print('saved:', SIG.shape)

if TASK == 'binary_af':
    Y = (Y4 == (1 if SYNTH else LABELS.index('A'))).astype(np.int64); N_CLASS = 2
    CLASS_NAMES = ['non-AF', 'AF']
else:
    Y = Y4.copy(); N_CLASS = 4; CLASS_NAMES = ['Normal', 'AF', 'Other', 'Noisy']

r0 = np.random.default_rng(0); perm = r0.permutation(len(SIG))
n_test = int(0.20*len(SIG)); n_val = int(0.10*len(SIG))
test_idx = perm[:n_test]; val_idx = perm[n_test:n_test+n_val]; train_idx = perm[n_test+n_val:]
SIG_t = torch.tensor(SIG); LEN_t = torch.tensor(LEN); Y_t = torch.tensor(Y)
tr_idx, te_idx, va_idx = (torch.tensor(train_idx), torch.tensor(test_idx), torch.tensor(val_idx))

cnt = np.bincount(Y[train_idx], minlength=N_CLASS).astype(np.float32)
CLASS_W = torch.tensor(np.sqrt(cnt.sum()/(N_CLASS*np.maximum(cnt, 1)))).to(DEV)
print('classes:', dict(zip(CLASS_NAMES, cnt.astype(int).tolist())))
print('class weights:', [round(float(w), 3) for w in CLASS_W])
print(f'train {len(train_idx)} | test {len(test_idx)}', flush=True)

def make_crops(idx, seed=None, n_crops=1, train=True):
    g = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_crops):
        for i in idx.tolist():
            n = int(LEN_t[i])
            if train:
                st = int(g.integers(0, max(1, n-CROP+1)))
            else:
                st = 0 if n <= CROP else int((n-CROP)//2)
            x = SIG_t[i, st:st+CROP]
            if len(x) < CROP: x = torch.cat([x, torch.zeros(CROP-len(x))])
            xs.append(x); ys.append(int(Y_t[i]))
    return torch.stack(xs).unsqueeze(1), torch.tensor(ys)

def eval_tensors(which):
    xs, ys = [], []
    for c in range(EVAL_CROPS):
        for i in which.tolist():
            n = int(LEN_t[i])
            st = 0 if n <= CROP else int(min(n-CROP, c*(n-CROP)//max(1, EVAL_CROPS-1)))
            x = SIG_t[i, st:st+CROP]
            if len(x) < CROP: x = torch.cat([x, torch.zeros(CROP-len(x))])
            xs.append(x); ys.append(int(Y_t[i]))
    return torch.stack(xs).unsqueeze(1).to(DEV), torch.tensor(ys).to(DEV)

XE, YE = eval_tensors(te_idx)
XV, YV = eval_tensors(va_idx)
print('test tensor:', tuple(XE.shape), '| val tensor:', tuple(XV.shape), flush=True)

# ======================= (2) notebook cell 3: model, FL utilities =======================
class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        g = lambda c: nn.GroupNorm(min(8, c), c)
        self.c1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False); self.n1 = g(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False);               self.n2 = g(cout)
        self.sc = (nn.Identity() if stride == 1 and cin == cout
                   else nn.Conv1d(cin, cout, 1, stride=stride, bias=False))
        self.do = nn.Dropout(0.1)
    def forward(self, x):
        s = self.sc(x)
        h = F.silu(self.n1(self.c1(x)))
        h = self.do(h)
        h = self.n2(self.c2(h))
        return F.silu(h + s)

class ECGNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
                                  nn.GroupNorm(8, 32), nn.SiLU())
        self.body = nn.Sequential(ResBlock1D(32,48), ResBlock1D(48,64),
                                  ResBlock1D(64,96), ResBlock1D(96,128), ResBlock1D(128,128))
        self.head = nn.Linear(256, n_class)
    def forward(self, x):
        h = self.body(self.stem(x))
        return self.head(torch.cat([F.adaptive_avg_pool1d(h,1).flatten(1),
                                    F.adaptive_max_pool1d(h,1).flatten(1)], 1))

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def dirichlet_partition(labels, n_clients, alpha, seed):
    r = np.random.default_rng(seed); idx = [[] for _ in range(n_clients)]
    for c in range(N_CLASS):
        ids = np.where(labels == c)[0]; r.shuffle(ids)
        p = r.dirichlet([alpha]*n_clients)
        cuts = (np.cumsum(p)*len(ids)).astype(int)[:-1]
        for k, part in enumerate(np.split(ids, cuts)): idx[k].extend(part.tolist())
    return [np.array(sorted(i), dtype=int) for i in idx]

def _macro_f1(pred, y):
    f1 = []
    for c in range(N_CLASS):
        tp = float(((pred==c)&(y==c)).sum()); fp = float(((pred==c)&(y!=c)).sum())
        fn = float(((pred!=c)&(y==c)).sum())
        f1.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(f1))

@torch.no_grad()
def _pooled_logits(m, X, n_rec):
    out = []
    for b in range(0, len(X), 512): out.append(m(X[b:b+512]))
    return torch.cat(out).view(EVAL_CROPS, n_rec, N_CLASS).mean(0)

@torch.no_grad()
def evaluate(state):
    m = ECGNet(N_CLASS).to(DEV); m.load_state_dict(state); m.eval()
    lg_te = _pooled_logits(m, XE, len(te_idx)); y_te = YE.view(EVAL_CROPS, len(te_idx))[0]
    if N_CLASS == 2:
        lg_va = _pooled_logits(m, XV, len(va_idx)); y_va = YV.view(EVAL_CROPS, len(va_idx))[0]
        p_va = torch.softmax(lg_va, 1)[:, 1]
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 37):
            f = _macro_f1((p_va > float(thr)).long(), y_va)
            if f > best_f1: best_f1, best_thr = f, float(thr)
        p_te = torch.softmax(lg_te, 1)[:, 1]
        pred = (p_te > best_thr).long()
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y_te.cpu().numpy(), p_te.cpu().numpy()))
    else:
        pred = lg_te.argmax(1); auroc = float('nan')
    return float((pred == y_te).float().mean()), _macro_f1(pred, y_te), auroc

def fgsm_batch(model, xb, yb, eps):
    xb = xb.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xb), yb, weight=CLASS_W).backward()
    return (xb + eps*xb.grad.sign()).detach()

def local_update(gstate, ids, seed, scenario, adversarial, rnd):
    m = ECGNet(N_CLASS).to(DEV); m.load_state_dict(gstate); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    xb_all, yb_all = make_crops(torch.tensor(ids), seed=seed+rnd, n_crops=1, train=True)
    xb_all, yb_all = xb_all.to(DEV), yb_all.to(DEV)
    if adversarial and scenario == 'DATA_POISON':
        gg = torch.Generator().manual_seed(int(seed))
        k = int(POISON_RATE*len(yb_all))
        fl = torch.randperm(len(yb_all), generator=gg)[:k].to(DEV)
        yb_all = yb_all.clone(); yb_all[fl] = (yb_all[fl] + 1) % N_CLASS
    gen = torch.Generator().manual_seed(int(seed))
    for _ in range(LOCAL_EPOCHS):
        order = torch.randperm(len(yb_all), generator=gen)
        for b in range(0, len(order), BATCH):
            sel = order[b:b+BATCH].to(DEV)
            xb, yb = xb_all[sel], yb_all[sel]
            if scenario == 'FGSM':
                xb = fgsm_batch(m, xb, yb, FGSM_EPS*(4 if adversarial else 1))
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(xb), yb, weight=CLASS_W).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
    new = m.state_dict()
    upd = OrderedDict((k, (new[k].float()-gstate[k].float()).detach()) for k in gstate)
    if adversarial and scenario == 'MODEL_POISON':
        upd = OrderedDict((k, -6.0*v) for k, v in upd.items())
    return upd

def update_norm(u): return math.sqrt(sum(float((v**2).sum()) for v in u.values()))
def clip_update(u, mx):
    n = update_norm(u); f = min(1.0, mx/(n+1e-12))
    return OrderedDict((k, v*f) for k, v in u.items()), n
def top_k_sparsify(u, frac):
    flat = torch.cat([v.flatten() for v in u.values()]); k = max(1, int(frac*flat.numel()))
    thr = flat.abs().kthvalue(flat.numel()-k+1).values
    return OrderedDict((kk, torch.where(v.abs() >= thr, v, torch.zeros_like(v))) for kk, v in u.items())
def capped_weights(raw):
    w = np.asarray(raw, float); w = w/w.sum()
    cap = WEIGHT_CAP/len(w); w = np.minimum(w, cap); return w/w.sum()
def aggregate(ups, ws, keys):
    return OrderedDict((k, sum(u[k]*float(w) for u, w in zip(ups, ws))) for k in keys)
def mad_filter(norms, k=3.0, floor=0.25):
    n = np.asarray(norms, float); med = float(np.median(n))
    mad = max(float(np.median(np.abs(n-med))), floor*abs(med), 1e-9)
    return np.where(np.abs(n-med) <= k*mad)[0]

# ======================= (3) extended robustness study =======================
# %% [CELL A] ------------------------------------------------------------------
# ============================================================
# 4 — Extended robustness study: configuration
#     Arms: clipped FedAvg (= the ECG "EPP" arm), coordinate median,
#           trimmed mean, Multi-Krum, FLTrust, LSB eHealth.
#     Attacks: none, -6 sign flip (paper), ALIE, IPM, norm-matched flip
#              (adaptive against norm-only MAD screening), 30% label flip.
# ============================================================
from scipy.stats import norm as _norm

EXT_OUT = DRIVE/('SMOKE_TEST' if SMOKE or SYNTH else 'robust_baselines'); EXT_OUT.mkdir(parents=True, exist_ok=True)
EXT_CSV  = EXT_OUT/'ecg_robust_runs.csv'
EXT_HIST = EXT_OUT/'ecg_robust_histories.json'

EXT_ARMS    = ['FEDAVG_CLIP', 'MEDIAN', 'TRIMMED_MEAN', 'MULTI_KRUM', 'FLTRUST', 'LSB', 'LSB_V2']
# LSB_V2 = LSB eHealth with two fixes, both fixed in advance (no tuning on results):
#   (1) the screen is ALWAYS on (the original only screened when an attack was known to be running);
#   (2) a direction check is added to the norm (MAD) check: an update is excluded if its cosine
#       similarity to the coordinate-wise median of all updates is below DIR_TAU.
DIR_TAU = 0.0
EXT_ATTACKS = ['NONE', 'SIGNFLIP6', 'NORM_MATCHED_FLIP', 'ALIE', 'IPM', 'LABEL_FLIP30']
EXT_SEEDS   = SEEDS                       # 42..47, paired across arms
# Run order: the most informative cells first, so partial results are already usable.
PRIORITY = ['SIGNFLIP6', 'NORM_MATCHED_FLIP', 'ALIE', 'NONE', 'IPM', 'LABEL_FLIP30']
if os.environ.get('ECG_ARMS'):    EXT_ARMS = os.environ['ECG_ARMS'].split(',')
if os.environ.get('ECG_ATTACKS'): PRIORITY = os.environ['ECG_ATTACKS'].split(',')
if SMOKE: PRIORITY = ['NONE', 'NORM_MATCHED_FLIP']

N_ADV      = max(1, int(ADV_FRACTION*N_CLIENTS))   # 2 of 10, same as the paper
IPM_EPS    = 0.5                                    # Xie et al. 2020, small-epsilon IPM
ROOT_SIZE  = 200                                    # FLTrust server root set
# ALIE z_max (Baruch et al., NeurIPS 2019): s = floor(n/2 + 1) - f ; z = Phi^-1((n - s)/n)
_s = int(N_CLIENTS/2 + 1) - N_ADV
ALIE_Z = float(_norm.ppf((N_CLIENTS - _s)/N_CLIENTS))
print(f'adversaries={N_ADV}/{N_CLIENTS} | ALIE z={ALIE_Z:.3f} | IPM eps={IPM_EPS}')

# FLTrust root set: a small, stratified slice of the validation split.
# NOTE for the paper: this gives FLTrust clean server-side data that the other arms do not get,
# which favours FLTrust; state this in the setup section.
_rr = np.random.default_rng(1234)
_va = va_idx.numpy()
_pos = _va[Y[_va] == 1]; _neg = _va[Y[_va] == 0]
_npos = max(1, int(round(ROOT_SIZE*len(_pos)/len(_va))))
ROOT_IDX = np.concatenate([_rr.choice(_pos, _npos, replace=False),
                           _rr.choice(_neg, ROOT_SIZE-_npos, replace=False)])
print('FLTrust root set:', len(ROOT_IDX), 'records,', int(Y[ROOT_IDX].sum()), 'AF')


# %% [CELL B] ------------------------------------------------------------------
# ============================================================
# 5 — Robust aggregators and attacks (all operate on flattened updates)
# ============================================================
def flat(u):   return torch.cat([v.reshape(-1).float() for v in u.values()])
def unflat(vec, like):
    out, i = OrderedDict(), 0
    for k, v in like.items():
        n = v.numel(); out[k] = vec[i:i+n].view_as(v).to(v.dtype); i += n
    return out

def agg_median(V):
    S = torch.sort(V, dim=0).values; n = V.shape[0]
    return S[n//2] if n % 2 else 0.5*(S[n//2-1] + S[n//2])

def agg_trimmed(V, b):
    S = torch.sort(V, dim=0).values
    return S[b:V.shape[0]-b].mean(0)

def agg_multikrum(V, f, m=None):
    n = V.shape[0]; m = n - f if m is None else m
    D = torch.cdist(V, V)**2
    k = max(1, n - f - 2)
    scores = torch.stack([torch.sort(D[i][torch.arange(n, device=V.device) != i]).values[:k].sum()
                          for i in range(n)])
    keep = torch.argsort(scores)[:m]
    return V[keep].mean(0), keep.tolist()

def agg_fltrust(V, g0):
    g0n = g0.norm() + 1e-12
    cos = (V @ g0) / (V.norm(dim=1) * g0n + 1e-12)
    ts = torch.relu(cos)
    if float(ts.sum()) <= 0: return torch.zeros_like(g0), ts.tolist()
    Vn = V * (g0n / (V.norm(dim=1, keepdim=True) + 1e-12))
    return (ts[:, None] * Vn).sum(0) / ts.sum(), ts.tolist()

def craft(attack, benign_vecs, own_vec):
    """Return the malicious raw update (flattened). Omniscient adversary: it sees
    the benign updates of the current round, the standard assumption for ALIE/IPM."""
    B = torch.stack(benign_vecs); mu = B.mean(0)
    if attack == 'SIGNFLIP6':
        return -6.0*own_vec
    if attack == 'ALIE':
        return mu - ALIE_Z*B.std(0, unbiased=True)
    if attack == 'IPM':
        return -IPM_EPS*mu
    if attack == 'NORM_MATCHED_FLIP':
        # opposite of the benign mean, rescaled to the median benign norm, so that a
        # norm-only screen (Eq. 15-17) cannot separate it from honest clients
        med = torch.median(B.norm(dim=1))
        return -mu * (med / (mu.norm() + 1e-12))
    raise ValueError(attack)


# %% [CELL C] ------------------------------------------------------------------
# ============================================================
# 6 — run_ext(): one federated run for any (arm, attack, seed)
# ============================================================
def run_ext(arm, attack, seed, verbose=False):
    seed_all(seed)
    parts = [train_idx[p] for p in dirichlet_partition(Y[train_idx], N_CLIENTS, DIRICHLET_A, seed)]
    n_adv = 0 if attack == 'NONE' else N_ADV
    adv = set(np.random.default_rng(seed).choice(N_CLIENTS, n_adv, replace=False).tolist())

    gstate = ECGNet(N_CLASS).to(DEV).state_dict(); keys = list(gstate)
    momentum = OrderedDict((k, torch.zeros_like(v, dtype=torch.float32)) for k, v in gstate.items())
    residual = {c: None for c in range(N_CLIENTS)}
    hist = {'round': [], 'accuracy': [], 'macro_f1': []}
    adv_excl = hon_excl = 0; t0 = time.perf_counter()

    for r in range(ROUNDS):
        lr_scale = 0.5*(1 + math.cos(math.pi*r/ROUNDS))
        # 1) every client trains honestly (label-flip clients train on poisoned labels)
        raw = []
        for ci in range(N_CLIENTS):
            sc = 'DATA_POISON' if (attack == 'LABEL_FLIP30' and ci in adv) else 'CLEAN'
            u = local_update(gstate, parts[ci], seed*1000+ci, sc, ci in adv, r)
            raw.append(OrderedDict((k, v*lr_scale) for k, v in u.items()))
        # 2) model-poisoning clients replace their update
        if attack not in ('NONE', 'LABEL_FLIP30') and adv:
            benign = [flat(raw[i]) for i in range(N_CLIENTS) if i not in adv]
            for ci in adv:
                raw[ci] = unflat(craft(attack, benign, flat(raw[ci])), raw[ci])
        # 3) client-side pipeline: LSB = residual + top-k + clip ; others = clip
        ups, sizes, norms = [], [], []
        for ci in range(N_CLIENTS):
            u = raw[ci]
            if arm in ('LSB', 'LSB_V2'):
                if residual[ci] is not None: u = OrderedDict((k, u[k]+residual[ci][k]) for k in u)
                sp = top_k_sparsify(u, SPARSIFY)
                left = OrderedDict((k, (u[k]-sp[k]).detach()) for k in u)
                ln = math.sqrt(sum(float((v**2).sum()) for v in left.values()))
                residual[ci] = OrderedDict((k, v*min(1.0, RESIDUAL_CAP*CLIP_NORM/(ln+1e-12)))
                                           for k, v in left.items())
                u = sp
            u, rn = clip_update(u, CLIP_NORM)
            ups.append(u); sizes.append(len(parts[ci])); norms.append(rn)

        # 4) server-side aggregation
        if arm in ('LSB', 'LSB_V2'):
            keep = list(range(N_CLIENTS))
            if arm == 'LSB_V2':
                V = torch.stack([flat(u) for u in ups]); med = agg_median(V)
                cos = (V @ med) / (V.norm(dim=1) * med.norm() + 1e-12)
                norm_ok = set(mad_filter(norms).tolist())
                keep = [i for i in range(N_CLIENTS) if i in norm_ok and float(cos[i]) >= DIR_TAU]
                if len(keep) < 2: keep = list(range(N_CLIENTS))      # self-denial guard (Alg. 5)
            elif attack != 'NONE' or FILTER_ON_CLEAN:
                keep = mad_filter(norms).tolist()
                if len(keep) < 2: keep = list(range(N_CLIENTS))      # self-denial guard (Alg. 5)
            groups = [keep[i::3] for i in range(3)]
            cl, sz = [], []
            for gr in groups:
                if not gr: continue
                cl.append(aggregate([ups[i] for i in gr], capped_weights([sizes[i] for i in gr]), keys))
                sz.append(sum(sizes[i] for i in gr))
            w = np.asarray(sz, float); agg = aggregate(cl, w/w.sum(), keys)
        elif arm == 'FEDAVG_CLIP':                       # identical to the ECG "EPP" arm
            keep = list(range(N_CLIENTS))
            agg = aggregate(ups, capped_weights(sizes), keys)
        else:
            V = torch.stack([flat(u) for u in ups])
            if arm == 'MEDIAN':
                vec, keep = agg_median(V), list(range(N_CLIENTS))
            elif arm == 'TRIMMED_MEAN':
                vec, keep = agg_trimmed(V, N_ADV), list(range(N_CLIENTS))
            elif arm == 'MULTI_KRUM':
                vec, keep = agg_multikrum(V, N_ADV)
            elif arm == 'FLTRUST':
                g = local_update(gstate, ROOT_IDX, seed*1000+999, 'CLEAN', False, r)
                g = OrderedDict((k, v*lr_scale) for k, v in g.items())
                g, _ = clip_update(g, CLIP_NORM)
                vec, ts = agg_fltrust(V, flat(g))
                keep = [i for i, t in enumerate(ts) if t > 0]
            else:
                raise ValueError(arm)
            agg = unflat(vec, ups[0])
        adv_excl += sum(1 for i in adv if i not in keep)
        hon_excl += sum(1 for i in range(N_CLIENTS) if i not in adv and i not in keep)

        momentum = OrderedDict((k, SERVER_MOM*momentum[k] + agg[k]) for k in keys)
        gstate = OrderedDict((k, gstate[k].float() + momentum[k]) for k in keys)

        if (r+1) % EVAL_EVERY == 0 or r == ROUNDS-1:
            if (r+1) % BASE_EVAL_EVERY == 0 or r == ROUNDS-1:
                a, f, _ = evaluate(gstate)
            else:
                with torch.random.fork_rng(devices=[DEV] if torch.device(DEV).type == 'cuda' else []):
                    a, f, _ = evaluate(gstate)
            hist['round'].append(r+1); hist['accuracy'].append(round(a,4)); hist['macro_f1'].append(round(f,4))
            if verbose: print(f'  r{r+1:02d} acc={a:.4f} F1={f:.4f}')

    acc, mf1, auroc = evaluate(gstate)
    R = ROUNDS
    selects = arm in ('LSB', 'LSB_V2', 'MULTI_KRUM', 'FLTRUST')
    return {'arm': arm, 'attack': attack, 'seed': seed,
            'test_accuracy': round(acc,4), 'macro_f1': round(mf1,4),
            'auroc': None if math.isnan(auroc) else round(auroc,4),
            # selection rates per round (only for arms that select clients: LSB, Multi-Krum, FLTrust)
            'adv_exclusion_rate': (round(adv_excl/(R*len(adv)),4) if (adv and selects) else None),
            'honest_exclusion_rate': (round(hon_excl/(R*(N_CLIENTS-len(adv))),4) if selects else None),
            'wall_seconds': round(time.perf_counter()-t0,1)}, hist


# %% [CELL D] ------------------------------------------------------------------
# ============================================================
# 7 — Full matrix, resumable: saved after every run
# ============================================================
ext = pd.read_csv(EXT_CSV) if EXT_CSV.exists() else pd.DataFrame()
ext_hist = json.loads(EXT_HIST.read_text()) if EXT_HIST.exists() else {}
def _done(a, at, s):
    return (not ext.empty) and ((ext.arm==a)&(ext.attack==at)&(ext.seed==s)).any()

plan = [(at, s, a) for at in PRIORITY if at in EXT_ATTACKS for s in EXT_SEEDS for a in EXT_ARMS]
print(f'{len(plan)} runs planned, {sum(_done(a,at,s) for at,s,a in plan)} already saved')
for i, (at, s, a) in enumerate(plan, 1):
    if _done(a, at, s): continue
    row, h = run_ext(a, at, s)
    ext = pd.concat([ext, pd.DataFrame([row])], ignore_index=True); ext.to_csv(EXT_CSV, index=False)
    ext_hist[f'{a}|{at}|{s}'] = h; EXT_HIST.write_text(json.dumps(ext_hist))
    print(f'[{i}/{len(plan)}] {at:17s} seed{s} {a:12s} acc={row["test_accuracy"]:.4f} '
          f'F1={row["macro_f1"]:.4f} advExcl={row["adv_exclusion_rate"]} ({row["wall_seconds"]:.0f}s)')


# %% [CELL E] ------------------------------------------------------------------
# ============================================================
# 8 — Tables: mean ± 95% CI, worst seed, paired Wilcoxon of each arm vs LSB
# ============================================================
from scipy import stats
ext = pd.read_csv(EXT_CSV).drop_duplicates(['arm','attack','seed'], keep='last')

def ci95(v):
    v = np.asarray(v, float)
    return 0.0 if len(v) < 2 else float(stats.t.ppf(0.975, len(v)-1)*v.std(ddof=1)/math.sqrt(len(v)))

rows = []
for (at, a), g in ext.groupby(['attack','arm']):
    REF = os.environ.get('ECG_REF_ARM', 'LSB')
    l = ext[(ext.attack==at)&(ext.arm==REF)].set_index('seed')
    g = g.set_index('seed'); common = sorted(set(g.index) & set(l.index))
    rec = {'attack': at, 'arm': a, 'seeds': len(g),
           'macro_f1': f"{g.macro_f1.mean():.4f} ± {ci95(g.macro_f1):.4f}",
           'accuracy': f"{g.test_accuracy.mean():.4f} ± {ci95(g.test_accuracy):.4f}",
           'worst_F1': round(g.macro_f1.min(),4), 'worst_acc': round(g.test_accuracy.min(),4),
           'adv_excluded': None if g.adv_exclusion_rate.isna().all() else round(g.adv_exclusion_rate.mean(),3),
           'honest_excluded': None if g.honest_exclusion_rate.isna().all() else round(g.honest_exclusion_rate.mean(),3)}
    if a != REF and len(common) >= 5:
        d = l.loc[common,'macro_f1'].values - g.loc[common,'macro_f1'].values
        try: p = stats.wilcoxon(d).pvalue
        except ValueError: p = float('nan')
        rec.update({f'{REF}_minus_arm_F1': round(float(d.mean()),4), f'{REF}_wins': int((d>0).sum()),
                    'p_wilcoxon': round(float(p),4)})
    rows.append(rec)
table = pd.DataFrame(rows).sort_values(['attack','arm'])
table.to_csv(EXT_OUT/f'table_robust_baselines_vs_{REF}.csv', index=False)
(EXT_OUT/f'table_robust_baselines_vs_{REF}.tex').write_text(table.to_latex(index=False, escape=True))
display(table)
print('With six paired seeds the smallest attainable two-sided Wilcoxon p is 0.031; '
      'report wins/6 and worst-case alongside p.')

print('DONE: ECG robust baselines ->', EXT_OUT, flush=True)
