# AUTO-GENERATED from LSB_EHEALTH_COMPLETE_BASE_PAPERS_COLAB.ipynb for local (non-Colab) runs.
# Only the Colab/Drive bootstrap was replaced; the experiment code is the notebook's own.
import os, sys, json, time, shutil, zipfile
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
RESULTS = Path(os.environ.get("LSB_RESULTS", Path(__file__).resolve().parent.parent / "results"))
SMOKE = os.environ.get("LSB_SMOKE", "0") == "1"
MASTER_ROOT = RESULTS / 'cifar_epp'
MASTER_ROOT.mkdir(parents=True, exist_ok=True)
print("MASTER ROOT:", MASTER_ROOT, "| smoke test:" , SMOKE, flush=True)
def display(*objs, **kw):
    import pandas as _pd
    for o in objs:
        if isinstance(o, _pd.DataFrame):
            with _pd.option_context("display.max_columns", 50, "display.width", 200):
                print(o.head(200).to_string())
        else:
            print(o)

# ===================== notebook cell 16 =====================
SEED_PLAN='MATCHED_LOCAL'   # defined in Part A of the original notebook
import copy, hashlib, json, math, os, random, shutil, time, warnings, zipfile
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
warnings.filterwarnings('default')

def progress(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}",flush=True)
# ===================== notebook cell 17 =====================

# B0 — EPP resource-aligned control panel
PROJECT_ID='EPP_BCFL_REPRO_COLAB_V1'
PROFILE='RESOURCE_ALIGNED'

SEEDS=tuple(int(x) for x in os.environ.get('EPP_SEEDS','42').split(','))
CLIENT_GRID=(10,)
REPRESENTATIVE_CLIENTS=10

GLOBAL_ROUNDS=2 if SMOKE else 50
LOCAL_EPOCHS=4            # MATCHED to the LSB CIFAR arm (was 2)
CLIENT_FRACTION=1.0
CLASSES_PER_CLIENT=8      # MATCHED to the LSB CIFAR arm (was 5)

LOCAL_BATCH_SIZE=64
LEARNING_RATE=0.05
MIN_LEARNING_RATE_FACTOR=0.02
MOMENTUM=0.9
WEIGHT_DECAY=5e-4

EARLY_STOP_PATIENCE=10**9  # early stopping OFF (LSB arm has none)
EARLY_STOP_MIN_ROUNDS=30
BEST_MIN_DELTA=1e-4

TARGET_EPSILONS=(0.5,)
DELTA=1e-5
CLIENT_CLIP_NORM=25.0
CLIP_CLEAN_UPDATES=False

MAX_AGG_WEIGHT_FACTOR=2.0
ADVERSARIAL_CLIENT_FRACTION=0.20
DATA_POISON_FRACTION=0.30
MODEL_POISON_SCALE=-5.0
FGSM_EPSILON=8/255
ANOMALY_MAD_THRESHOLD=3.5
POS_BFT_BYZANTINE_FRACTION=0.20
VALIDATION_FRACTION=0.10
NUM_WORKERS=0

# Drive-safe checkpoint cadence.
RESUME_EVERY_ROUNDS=10

ROOT=MASTER_ROOT/'epp_bcfl'
DIRS={n:ROOT/n for n in ('datasets','checkpoints','models','metrics','plots','logs','manifests')}
for p in DIRS.values():
    p.mkdir(parents=True,exist_ok=True)

print('EPP ROOT:',ROOT,flush=True)

# ===================== notebook cell 18 =====================
SCIENTIFIC_CONFIG={
 'project_id':PROJECT_ID,
 'seed_plan':SEED_PLAN,'seeds':SEEDS,'client_grid':CLIENT_GRID,
 'representative_clients':REPRESENTATIVE_CLIENTS,
 'global_rounds':GLOBAL_ROUNDS,'local_epochs':LOCAL_EPOCHS,
 'client_fraction':CLIENT_FRACTION,
 'classes_per_client':CLASSES_PER_CLIENT,
 'local_batch_size':LOCAL_BATCH_SIZE,
 'optimizer':['SGD',LEARNING_RATE,MOMENTUM,WEIGHT_DECAY,'cosine',MIN_LEARNING_RATE_FACTOR],
 'validation_fraction':VALIDATION_FRACTION,
 'architecture':'WideResNet-16-4 GroupNorm reconstruction',
 'best_checkpoint':['validation_accuracy',EARLY_STOP_PATIENCE,EARLY_STOP_MIN_ROUNDS,BEST_MIN_DELTA],
 'clean_clipping':CLIP_CLEAN_UPDATES,
 'dp':[TARGET_EPSILONS,DELTA,CLIENT_CLIP_NORM],
 'attacks':[ADVERSARIAL_CLIENT_FRACTION,DATA_POISON_FRACTION,MODEL_POISON_SCALE,FGSM_EPSILON],
}
EXPERIMENT_FINGERPRINT=hashlib.sha256(
    json.dumps(SCIENTIFIC_CONFIG,sort_keys=True,default=str).encode()
).hexdigest()[:16]


manifest_cfg=DIRS['manifests']/ 'scientific_config.json'
if manifest_cfg.exists():
    old=json.loads(manifest_cfg.read_text())
    old_fp=old.get('fingerprint')
    if old_fp and old_fp!=EXPERIMENT_FINGERPRINT:
        raise RuntimeError(
            f'EPP fingerprint mismatch existing={old_fp} active={EXPERIMENT_FINGERPRINT}. '
            'Use a fresh Drive folder for a changed scientific configuration.'
        )
manifest_cfg.write_text(json.dumps({
    'project_id':PROJECT_ID,
    'fingerprint':EXPERIMENT_FINGERPRINT,
    'config':SCIENTIFIC_CONFIG
},indent=2,default=str))
print('EPP experiment fingerprint:',EXPERIMENT_FINGERPRINT,flush=True)

# ===================== notebook cell 19 =====================
def detect_device():
    if not torch.cuda.is_available(): return torch.device('cpu'),{'reason':'CUDA unavailable'}
    try:
        major,minor=torch.cuda.get_device_capability(0); arch=f'sm_{major}{minor}'; supported=list(torch.cuda.get_arch_list())
        # arch check disabled: it wrongly rejected sm_89 (RTX 4000 Ada), which current PyTorch supports
        if False: return torch.device('cpu'),{'gpu':torch.cuda.get_device_name(0),'reason':f'{arch} unsupported'}
        x=torch.ones(1,device='cuda')+1; torch.cuda.synchronize(); del x
        return torch.device('cuda'),{'gpu':torch.cuda.get_device_name(0),'arch':arch}
    except Exception as exc: return torch.device('cpu'),{'reason':str(exc)}
DEVICE,DEVICE_DETAILS=detect_device(); CUDA_USABLE=DEVICE.type=='cuda'
print('Selected device:',DEVICE,DEVICE_DETAILS,flush=True)
if not CUDA_USABLE: raise RuntimeError('Use a compatible T4/L4 GPU for this batched EPP notebook.')

def seed_everything(seed):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
seed_everything(SEEDS[0])
# ===================== notebook cell 20 =====================
@dataclass
class EPPConfig:
    seed: int=42
    force_rerun: bool=False
    num_workers: int=NUM_WORKERS
    learning_rate: float=LEARNING_RATE
    min_learning_rate_factor: float=MIN_LEARNING_RATE_FACTOR
    momentum: float=MOMENTUM
    weight_decay: float=WEIGHT_DECAY
    local_batch_size: int=LOCAL_BATCH_SIZE
    local_epochs: int=LOCAL_EPOCHS
    global_rounds: int=GLOBAL_ROUNDS
    client_fraction: float=CLIENT_FRACTION
    non_iid_method: str='LIMITED_CLASSES'
    classes_per_client: int=CLASSES_PER_CLIENT
    max_aggregation_weight_factor: float=MAX_AGG_WEIGHT_FACTOR
    delta: float=DELTA
    client_clip_norm: float=CLIENT_CLIP_NORM
    clip_clean_updates: bool=CLIP_CLEAN_UPDATES
    early_stop_patience: int=EARLY_STOP_PATIENCE
    early_stop_min_rounds: int=EARLY_STOP_MIN_ROUNDS
    best_min_delta: float=BEST_MIN_DELTA
    adversarial_client_fraction: float=ADVERSARIAL_CLIENT_FRACTION
    data_poison_fraction: float=DATA_POISON_FRACTION
    model_poison_scale: float=MODEL_POISON_SCALE
    anomaly_mad_threshold: float=ANOMALY_MAD_THRESHOLD
    pos_bft_byzantine_fraction: float=POS_BFT_BYZANTINE_FRACTION
    simulated_network_rtt_ms: float=5.0
    simulated_message_processing_ms: float=0.01

CFG=EPPConfig()
CFG.client_grid=CLIENT_GRID

def stage_checkpoint(name):
    return DIRS['checkpoints']/f'{name}.pt'

def stage_csv(name):
    return DIRS['metrics']/f'{name}.csv'

def validate_checkpoint_fingerprint(state,name):
    if state.get('fingerprint')!=EXPERIMENT_FINGERPRINT:
        raise RuntimeError(f'Checkpoint fingerprint mismatch: {name}')

def load_pipeline_checkpoint(path,map_location='cpu'):
    p=Path(path).resolve()
    if ROOT.resolve() not in p.parents and p!=ROOT.resolve():
        raise RuntimeError(f'Unsafe checkpoint path {p}')
    progress(f'CHECKPOINT LOAD → {p.name}')
    try:
        return torch.load(p,map_location=map_location,weights_only=False)
    except TypeError:
        return torch.load(p,map_location=map_location)

def write_resume_bundle():
    bundle=MASTER_ROOT/'EPP_BCFL_REPRO_V4_Resume.zip'
    tmp=bundle.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for folder_name in ('checkpoints','models','metrics','manifests','logs','plots'):
            for f in DIRS[folder_name].rglob('*'):
                if f.is_file():
                    z.write(f,f.relative_to(ROOT))
    tmp.replace(bundle)
    return bundle

def write_slim_results_bundle():
    bundle=MASTER_ROOT/'EPP_BCFL_REPRO_V4_RESULTS_SLIM.zip'
    tmp=bundle.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for folder_name in ('metrics','manifests','logs','plots'):
            for f in DIRS[folder_name].rglob('*'):
                if f.is_file():
                    z.write(f,f.relative_to(ROOT))
        for f in DIRS['models'].glob('epp_clean_n*_seed*.pt'):
            z.write(f,f.relative_to(ROOT))
    tmp.replace(bundle)
    return bundle

def maybe_write_resume_bundle(round_number):
    if round_number%RESUME_EVERY_ROUNDS==0 or round_number==GLOBAL_ROUNDS:
        p=write_resume_bundle()
        progress(f'RESUME SNAPSHOT → {p} | {p.stat().st_size/1024**2:.1f} MB')

print(
    'EPP V4 config validated | '
    f'batch={CFG.local_batch_size} | classes/client={CFG.classes_per_client} | '
    f'clean_clip={CFG.clip_clean_updates} | best-checkpoint=ON',
    flush=True,
)

# ===================== notebook cell 21 =====================
# Context-only paper reference; measured reconstruction remains primary.
PUBLISHED_EPP=pd.DataFrame([
 [10,88.5,87.2,85.6,86.4,20],
 [20,90.1,88.9,87.5,88.2,18],
 [30,92.3,90.5,89.8,90.2,15],
 [40,93.8,92.1,91.4,91.7,12],
 [50,95.2,94.0,93.3,93.6,10],
],columns=['num_clients','paper_accuracy','paper_precision','paper_recall','paper_f1','paper_convergence_epochs'])
PUBLISHED_EPP.to_csv(DIRS['metrics']/ 'epp_published_reference.csv',index=False)
display(PUBLISHED_EPP)
# ===================== notebook cell 22 =====================
# Reuse an attached CIFAR cache if available, otherwise torchvision downloads it.
def detect_cifar_root():
    root=MASTER_ROOT/'datasets'/'cifar10'
    root.mkdir(parents=True,exist_ok=True)
    return str(root)
CIFAR_ROOT=detect_cifar_root()
MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
train_transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
eval_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
train_aug=datasets.CIFAR10(CIFAR_ROOT,train=True,download=True,transform=train_transform)
train_eval=datasets.CIFAR10(CIFAR_ROOT,train=True,download=True,transform=eval_transform)
test_eval=datasets.CIFAR10(CIFAR_ROOT,train=False,download=True,transform=eval_transform)
all_idx=np.arange(len(train_eval))
all_labels=np.asarray(train_eval.targets)
train_indices,val_indices=train_test_split(
    all_idx,
    test_size=VALIDATION_FRACTION,
    random_state=42,
    stratify=all_labels,
)
train_indices=np.asarray(train_indices,dtype=int)
val_indices=np.asarray(val_indices,dtype=int)
test_indices=np.arange(len(test_eval))
VALSET=Subset(train_eval,list(map(int,val_indices)))
TESTSET=Subset(test_eval,list(map(int,test_indices)))
print({
    'official_train':len(train_eval),
    'federated_train':len(train_indices),
    'validation':len(val_indices),
    'official_test':len(test_eval),
    'note':'validation is used for best-round selection; official test remains untouched',
},flush=True)

# ===================== notebook cell 23 =====================
# Shared reconstruction CNN: CIFAR WideResNet-16-4 with GroupNorm.
# The paper states CNN but does not disclose its exact architecture.

class WideBasic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=16):
        super().__init__()
        g1 = min(groups, in_channels)
        while in_channels % g1 != 0 and g1 > 1:
            g1 -= 1
        g2 = min(groups, out_channels)
        while out_channels % g2 != 0 and g2 > 1:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x):
        out = F.silu(self.norm1(x))
        shortcut = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else x)
        out = self.conv1(out)
        out = self.conv2(F.silu(self.norm2(out)))
        return out + shortcut


class CifarWideResNet16x4(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        # WRN-16-4: 2 residual blocks per group, widths 64/128/256.
        self.layer1 = nn.Sequential(
            WideBasic(16, 64, 1),
            WideBasic(64, 64, 1),
        )
        self.layer2 = nn.Sequential(
            WideBasic(64, 128, 2),
            WideBasic(128, 128, 1),
        )
        self.layer3 = nn.Sequential(
            WideBasic(128, 256, 2),
            WideBasic(256, 256, 1),
        )

        self.norm = nn.GroupNorm(16, 256)
        self.fc = nn.Linear(256, classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.silu(self.norm(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def make_model():
    return CifarWideResNet16x4().to(DEVICE)


def finite_model(model):
    return all(
        not torch.is_floating_point(v) or torch.isfinite(v).all().item()
        for v in model.state_dict().values()
    )


def assert_finite_model(model, context="model"):
    bad = [
        k for k, v in model.state_dict().items()
        if torch.is_floating_point(v) and not torch.isfinite(v).all()
    ]
    if bad:
        raise FloatingPointError(f"{context} has non-finite tensors: {bad[:10]}")


@torch.no_grad()
def evaluate(model, dataset, adversarial_epsilon=0.0):
    model = model.to(DEVICE).eval()
    assert_finite_model(model, "evaluation model")
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=CUDA_USABLE,
    )
    losses, truth, predicted = [], [], []

    for batch_id, (x, y) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=CUDA_USABLE)
        y = y.to(DEVICE, non_blocking=CUDA_USABLE)

        if adversarial_epsilon > 0:
            with torch.enable_grad():
                x_adv = x.detach().clone().requires_grad_(True)
                loss_adv = F.cross_entropy(model(x_adv), y)
                gradient = torch.autograd.grad(loss_adv, x_adv)[0]
                std = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)
                mean = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
                epsilon_normalized = adversarial_epsilon / std
                lower = (0.0 - mean) / std
                upper = (1.0 - mean) / std
                x = torch.maximum(
                    torch.minimum(
                        x_adv.detach() + epsilon_normalized * gradient.sign(),
                        upper,
                    ),
                    lower,
                )

        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits in evaluation batch {batch_id}")

        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
        truth.extend(y.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(truth, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(truth, predicted).tolist(),
    }

preview = make_model()
print('Model: WideResNet-16-4 GroupNorm')
print('Trainable parameters:', sum(p.numel() for p in preview.parameters() if p.requires_grad), flush=True)
del preview

# ===================== notebook cell 24 =====================
# B3–B4 Non-IID limited-class client partitions and heterogeneity audit

def client_partitions(n_clients: int, seed: int):
    if CFG.non_iid_method != "LIMITED_CLASSES":
        raise NotImplementedError(
            "This notebook locks the paper-aligned LIMITED_CLASSES reconstruction."
        )

    rng = np.random.default_rng(seed)
    indices = np.asarray(train_indices, dtype=int)
    labels = np.asarray(train_eval.targets)[indices]
    classes = np.unique(labels)

    # Assign exactly classes_per_client class slots per client while keeping
    # every class represented across the population.
    slots = np.resize(classes, n_clients * CFG.classes_per_client)
    rng.shuffle(slots)
    client_classes = {
        client: set(map(int, slots[
            client * CFG.classes_per_client:(client + 1) * CFG.classes_per_client
        ]))
        for client in range(n_clients)
    }

    # Guarantee unique class count when repeated slots land on the same class.
    for client in range(n_clients):
        while len(client_classes[client]) < CFG.classes_per_client:
            client_classes[client].add(int(rng.choice(classes)))

    output = {client: [] for client in range(n_clients)}
    for label in classes:
        class_indices = indices[labels == label].copy()
        rng.shuffle(class_indices)
        eligible = [c for c in range(n_clients) if int(label) in client_classes[c]]
        if not eligible:
            eligible = [int(label) % n_clients]
            client_classes[eligible[0]].add(int(label))
        for client, chunk in zip(eligible, np.array_split(class_indices, len(eligible))):
            output[client].extend(map(int, chunk))

    # A client may be empty only if the source sample is too small.
    empty = [c for c, values in output.items() if not values]
    if empty:
        raise RuntimeError(f"Empty clients after non-IID partition: {empty}")

    flat = [idx for values in output.values() for idx in values]
    if len(flat) != len(set(flat)) or set(flat) != set(map(int, train_indices)):
        raise AssertionError("Client partition duplicated or lost training samples.")

    audit_rows = []
    for client, values in output.items():
        local_labels = np.asarray(train_eval.targets)[values]
        audit_rows.append({
            "num_clients": n_clients,
            "seed": seed,
            "client_id": client,
            "samples": len(values),
            "class_count": len(np.unique(local_labels)),
            "classes": ",".join(map(str, sorted(np.unique(local_labels).tolist()))),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        DIRS["metrics"] / f"client_partition_n{n_clients}_seed{seed}.csv",
        index=False
    )
    return output

# Visual audit for every client grid using the first seed.
partition_audit = []
for n_clients in CFG.client_grid:
    parts = client_partitions(n_clients, CFG.seed)
    for client, values in parts.items():
        local_labels = np.asarray(train_eval.targets)[values]
        partition_audit.append({
            "num_clients": n_clients,
            "client_id": client,
            "samples": len(values),
            "class_count": len(np.unique(local_labels)),
        })
PARTITION_AUDIT = pd.DataFrame(partition_audit)
PARTITION_AUDIT.to_csv(DIRS["metrics"] / "client_partition_audit.csv", index=False)
display(PARTITION_AUDIT.groupby("num_clients")[["samples", "class_count"]].describe())
# ===================== notebook cell 25 =====================
def blockchain_energy(payload_bytes,messages,cpu_ms):
    return payload_bytes*5e-7 + messages*2e-4 + cpu_ms*2e-5
# ===================== notebook cell 26 =====================
# C2 Trainable-parameter updates, clipping and RDP-accounted aggregate noise

def trainable_keys(model):
    return set(dict(model.named_parameters()))


def state_delta(local_state, global_state, keys):
    output = OrderedDict((k, local_state[k].float() - global_state[k].float()) for k in keys)
    assert_finite_update(output, "raw update")
    return output


def assert_finite_update(update, context="update"):
    bad = [k for k, v in update.items() if not torch.isfinite(v).all()]
    if bad:
        raise FloatingPointError(f"{context} has non-finite tensors: {bad[:10]}")


def update_norm(update):
    value = math.sqrt(sum(float((v.float() ** 2).sum()) for v in update.values()))
    if not math.isfinite(value):
        raise FloatingPointError("Update norm is non-finite")
    return value


def clip_update(update, max_norm):
    norm = update_norm(update)
    factor = min(1.0, max_norm / (norm + 1e-12))
    clipped = OrderedDict((k, v.float() * factor) for k, v in update.items())
    return clipped, norm


def normalize_capped_weights(raw_weights):
    weights = np.asarray(raw_weights, dtype=np.float64)
    if len(weights) == 0 or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid aggregation weights")
    weights = weights / weights.sum()
    cap = CFG.max_aggregation_weight_factor / len(weights)
    weights = np.minimum(weights, cap)
    weights = weights / weights.sum()
    return weights


def weighted_average(updates, raw_weights):
    weights = normalize_capped_weights(raw_weights)
    keys = list(updates[0])
    result = OrderedDict()
    for key in keys:
        accumulator = torch.zeros_like(updates[0][key], dtype=torch.float32)
        for update, weight in zip(updates, weights):
            accumulator += update[key].float() * float(weight)
        result[key] = accumulator
    assert_finite_update(result, "weighted aggregate")
    return result, weights


def rdp_epsilon(noise_multiplier, sample_rate, steps, delta):
    """Exact RDP composition for a full-participation Gaussian mechanism.

    The current research configuration uses sample_rate=1.0. Subsampled-client
    accounting is intentionally rejected rather than approximated silently.
    """
    if not math.isclose(sample_rate, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise NotImplementedError(
            "This notebook currently supports formal RDP accounting only for "
            "client_fraction=1.0. Keep full client participation or add a "
            "validated subsampled-Gaussian accountant."
        )
    if noise_multiplier <= 0:
        return math.inf
    orders = np.concatenate([
        np.arange(1.1, 10.0, 0.1),
        np.arange(10.0, 64.0, 1.0),
        np.asarray([64.0, 96.0, 128.0, 256.0]),
    ])
    rdp = steps * orders / (2.0 * noise_multiplier ** 2)
    epsilons = rdp + math.log(1.0 / delta) / (orders - 1.0)
    return float(np.min(epsilons))


def noise_multiplier_for_target(target_epsilon, sample_rate, steps, delta):
    if math.isinf(target_epsilon):
        return 0.0
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    low, high = 1e-4, 1.0
    while rdp_epsilon(high, sample_rate, steps, delta) > target_epsilon:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("Unable to calibrate a finite Gaussian noise multiplier")
    for _ in range(100):
        mid = (low + high) / 2.0
        if rdp_epsilon(mid, sample_rate, steps, delta) > target_epsilon:
            low = mid
        else:
            high = mid
    return high


def add_secure_aggregate_noise(aggregate, normalized_weights, target_epsilon, sample_rate, steps, seed, max_effective_weight=None):
    if math.isinf(target_epsilon):
        return aggregate, {"noise_multiplier": 0.0, "epsilon": math.inf, "noise_std": 0.0}
    sigma = noise_multiplier_for_target(target_epsilon, sample_rate, steps, CFG.delta)
    # Replacement sensitivity of a capped weighted aggregate is bounded by 2*C*max(weight).
    effective_weight = float(np.max(normalized_weights)) if max_effective_weight is None else float(max_effective_weight)
    sensitivity = 2.0 * CFG.client_clip_norm * effective_weight
    std = sigma * sensitivity
    generator = torch.Generator(device="cpu").manual_seed(seed)
    private = OrderedDict()
    for key, value in aggregate.items():
        noise = torch.normal(0.0, std, size=value.shape, generator=generator, dtype=torch.float32)
        private[key] = value.float() + noise
    assert_finite_update(private, "private aggregate")
    return private, {"noise_multiplier": sigma, "epsilon": rdp_epsilon(sigma, sample_rate, steps, CFG.delta), "noise_std": std}


def apply_delta(global_state, delta):
    output = OrderedDict((k, v.detach().cpu().clone()) for k, v in global_state.items())
    for key, value in delta.items():
        output[key] = output[key].float() + value.float()
    if any(torch.is_floating_point(v) and not torch.isfinite(v).all() for v in output.values()):
        raise FloatingPointError("Updated global model is non-finite")
    return output


def update_bytes(update):
    return sum(v.numel() * v.element_size() for v in update.values())
# ===================== notebook cell 27 =====================
# C3 Local training, AMA weights, anomaly filtering and PoS+BFT simulation
@dataclass
class ClientHistory:
    trust: float = 1.0
    previous_norm: float | None = None


def train_local(global_model, indices, seed, label_poison=False, fgsm_poison=False, learning_rate=None):
    seed_everything(seed)
    model = copy.deepcopy(global_model).to(DEVICE)
    lr = CFG.learning_rate if learning_rate is None else float(learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=CFG.momentum, weight_decay=CFG.weight_decay)
    loader = DataLoader(Subset(train_aug, list(map(int, indices))), batch_size=CFG.local_batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=CUDA_USABLE, generator=torch.Generator().manual_seed(seed))
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(CFG.local_epochs):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=CUDA_USABLE)
            y = y.to(DEVICE, non_blocking=CUDA_USABLE)
            if label_poison:
                mask = torch.rand(len(y), device=DEVICE) < CFG.data_poison_fraction
                y = torch.where(mask, (y + 1) % 10, y)
            # Persistent malicious-client FGSM: perturb local training inputs.
            if fgsm_poison:
                x_seed = x.detach().clone().requires_grad_(True)
                seed_loss = F.cross_entropy(model(x_seed), y)
                grad = torch.autograd.grad(seed_loss, x_seed, retain_graph=False, create_graph=False)[0]
                mean_t = torch.tensor(MEAN, device=DEVICE, dtype=x.dtype).view(1,3,1,1)
                std_t = torch.tensor(STD, device=DEVICE, dtype=x.dtype).view(1,3,1,1)
                eps_t = (FGSM_EPSILON / std_t)
                lo = (torch.zeros_like(mean_t) - mean_t) / std_t
                hi = (torch.ones_like(mean_t) - mean_t) / std_t
                x = torch.maximum(torch.minimum(x_seed.detach() + eps_t * grad.sign(), hi), lo)

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Local loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
    assert_finite_model(model, "local model")
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items()), {
        "loss": float(np.mean(losses)),
        "samples": len(indices),
        "seconds": time.perf_counter() - started,
    }


def ama_weight(samples, local_loss, norm, history):
    """
    Paper-aligned clean reconstruction.

    The paper explicitly gives sample-count weighting |Di|/sum|Dj| for the
    global update. It describes additional AMA factors (quality, diversity,
    representativeness, trust) but does not disclose an executable equation.

    To avoid inventing a strong undocumented weighting rule, clean trusted
    clients use sample-count weighting. Trust becomes active only when later
    robustness phases change a client's trust state.
    """
    return max(1e-9, float(samples) * float(history.trust))

def sampled_update_vector(update, maximum=4096):
    vector = np.concatenate([v.detach().cpu().float().reshape(-1).numpy() for v in update.values()])
    if len(vector) > maximum:
        step = max(1, len(vector) // maximum)
        vector = vector[::step][:maximum]
    return vector


def robust_acceptance(updates, enabled):
    if not enabled or len(updates) < 4:
        return np.ones(len(updates), dtype=bool)
    norms = np.asarray([update_norm(u) for u in updates])
    median = np.median(norms)
    mad = np.median(np.abs(norms - median)) + 1e-12
    norm_ok = np.abs(norms - median) / (1.4826 * mad) <= CFG.anomaly_mad_threshold
    vectors = np.vstack([sampled_update_vector(u) for u in updates])
    center = np.median(vectors, axis=0)
    center_norm = np.linalg.norm(center) + 1e-12
    cosine = (vectors @ center) / ((np.linalg.norm(vectors, axis=1) + 1e-12) * center_norm)
    cosine_ok = cosine > -0.10
    return norm_ok & cosine_ok


def pos_bft_verify(client_updates, trusts, update_hashes, seed):
    """Simulated PoS+BFT integrity validation over actual received updates."""
    client_ids = list(client_updates)
    if not client_ids:
        return set(), 0, 0.0, 0, 0

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    validator_count = min(5, len(client_ids))
    stake_values = np.asarray(
        [max(1e-9, trusts[client_id].trust) for client_id in client_ids],
        dtype=float,
    )
    validators = rng.choice(
        client_ids,
        size=validator_count,
        replace=False,
        p=stake_values / stake_values.sum(),
    ).tolist()

    byzantine_count = min(
        validator_count,
        int(round(
            validator_count * CFG.pos_bft_byzantine_fraction
        )),
    )
    byzantine_validators = set(
        rng.choice(
            validators,
            size=byzantine_count,
            replace=False,
        ).tolist()
    ) if byzantine_count else set()

    tolerated = (validator_count - 1) // 3
    threshold = 2 * tolerated + 1
    accepted = set()

    for client_id, update in client_updates.items():
        expected_digest = hash_update(update)
        received_digest = update_hashes.get(client_id, "")
        syntactically_valid = (
            len(received_digest) == 64
            and all(character in "0123456789abcdef" for character in received_digest)
        )
        integrity_valid = (
            syntactically_valid
            and expected_digest == received_digest
        )

        yes_votes = 0
        for validator in validators:
            honest_vote = integrity_valid
            vote = (
                not honest_vote
                if validator in byzantine_validators
                else honest_vote
            )
            yes_votes += int(vote)

        if yes_votes >= threshold:
            accepted.add(client_id)

    messages = 2 * validator_count * max(0, validator_count - 1)
    latency_ms = (
        (time.perf_counter() - started) * 1000
        + 3 * CFG.simulated_network_rtt_ms
        + messages * CFG.simulated_message_processing_ms
    )
    return (
        accepted,
        messages,
        latency_ms,
        validator_count,
        byzantine_count,
    )


def hash_update(update):
    h = hashlib.sha256()
    for key, value in update.items():
        h.update(key.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()
# ===================== notebook cell 28 =====================
# Best-checkpoint, cosine-LR, resumable EPP runner

def global_round_lr(round_index):
    frac=round_index/max(1,CFG.global_rounds-1)
    cosine=0.5*(1.0+math.cos(math.pi*frac))
    factor=CFG.min_learning_rate_factor+(1.0-CFG.min_learning_rate_factor)*cosine
    return CFG.learning_rate*factor

def run_epp_experiment(run_name,n_clients,target_epsilon,seed,attack="NONE",aggregation_mode="AMA"):
    progress(
        f"EPP RUN START → {run_name} | clients={n_clients} | "
        f"epsilon={target_epsilon} | attack={attack} | aggregation={aggregation_mode}"
    )
    seed_everything(seed)
    final_model_path=DIRS["models"]/f"{run_name}.pt"
    final_metrics_path=stage_csv(run_name)
    partial_path=stage_checkpoint(run_name)

    if final_model_path.exists() and final_metrics_path.exists() and not CFG.force_rerun:
        checkpoint=load_pipeline_checkpoint(final_model_path,map_location="cpu")
        validate_checkpoint_fingerprint(checkpoint,final_model_path.name)
        model=make_model()
        model.load_state_dict(checkpoint["state_dict"])
        existing_log=pd.read_csv(final_metrics_path)
        progress(
            f"EPP RUN REUSED → {run_name} | best_round={checkpoint.get('best_round')} | "
            f"logged_rounds={len(existing_log)}"
        )
        return model.cpu(),existing_log

    partitions=client_partitions(n_clients,seed)
    model=make_model()
    histories={cid:ClientHistory() for cid in partitions}

    logs=[]
    start_round=0
    best_state=None
    best_round=0
    best_val_acc=-math.inf
    best_val_loss=math.inf
    rounds_without_improvement=0

    if partial_path.exists() and not CFG.force_rerun:
        state=load_pipeline_checkpoint(partial_path,map_location="cpu")
        validate_checkpoint_fingerprint(state,partial_path.name)
        model.load_state_dict(state["model"])
        histories={int(k):ClientHistory(**v) for k,v in state["histories"].items()}
        logs=state.get("logs",[])
        start_round=int(state.get("next_round",0))
        best_state=state.get("best_state")
        best_round=int(state.get("best_round",0))
        best_val_acc=float(state.get("best_val_acc",-math.inf))
        best_val_loss=float(state.get("best_val_loss",math.inf))
        rounds_without_improvement=int(state.get("rounds_without_improvement",0))
        progress(
            f"EPP RESUME → {run_name} | next={start_round+1}/{CFG.global_rounds} | "
            f"best={best_val_acc:.4f}@{best_round}"
        )

    trainable=trainable_keys(model)
    sample_rate=CFG.client_fraction
    malicious_count=int(round(n_clients*CFG.adversarial_client_fraction))
    malicious_rng=np.random.default_rng(seed+700)
    malicious=set(
        malicious_rng.choice(n_clients,size=malicious_count,replace=False).tolist()
    ) if malicious_count else set()

    round_times=[]

    for round_index in range(start_round,CFG.global_rounds):
        round_started=time.perf_counter()
        current_lr=global_round_lr(round_index)
        progress(
            f"EPP ROUND START → {run_name} | {round_index+1}/{CFG.global_rounds} | lr={current_lr:.6f}"
        )

        round_rng=np.random.default_rng(seed+round_index*100003)
        selected_count=max(1,math.ceil(n_clients*CFG.client_fraction))
        selected=round_rng.choice(n_clients,size=selected_count,replace=False).tolist()
        global_state=OrderedDict(
            (k,v.detach().cpu().clone()) for k,v in model.state_dict().items()
        )

        updates,weights,metadata,hashes=[],[],[],{}
        client_progress_step=max(1,len(selected)//4)

        for client_position,client_id in enumerate(selected,start=1):
            local_state,info=train_local(
                model,
                partitions[client_id],
                seed+round_index*10000+client_id,
                label_poison=client_id in malicious and attack=="DATA_POISON",
                fgsm_poison=client_id in malicious and attack=="FGSM_TRAIN",
                learning_rate=current_lr,
            )
            raw_update=state_delta(local_state,global_state,trainable)

            if client_id in malicious and attack=="MODEL_POISON":
                raw_update=OrderedDict(
                    (k,v*CFG.model_poison_scale) for k,v in raw_update.items()
                )

            original_norm=update_norm(raw_update)
            should_clip=(not math.isinf(target_epsilon)) or CFG.clip_clean_updates

            if should_clip:
                update,_=clip_update(raw_update,CFG.client_clip_norm)
                was_clipped=original_norm>CFG.client_clip_norm
            else:
                update=raw_update
                was_clipped=False

            if aggregation_mode=="FEDAVG":
                weight=float(info["samples"])
            elif aggregation_mode=="AMA":
                weight=ama_weight(
                    info["samples"],info["loss"],update_norm(update),histories[client_id]
                )
            else:
                raise ValueError(f"Unknown aggregation_mode={aggregation_mode}")

            updates.append(update)
            weights.append(weight)
            hashes[client_id]=hash_update(update)
            metadata.append({
                "client_id":client_id,
                "malicious":client_id in malicious,
                "original_norm":original_norm,
                "was_clipped":was_clipped,
                **info,
            })

            if (
                client_position==1
                or client_position%client_progress_step==0
                or client_position==len(selected)
            ):
                progress(
                    f"  local clients → {client_position}/{len(selected)} "
                    f"for {run_name} round {round_index+1}"
                )

        detection_started=time.perf_counter()
        accepted_mask=robust_acceptance(updates,enabled=attack!="NONE")
        detection_seconds=time.perf_counter()-detection_started

        accepted_updates=[u for u,ok in zip(updates,accepted_mask) if ok]
        accepted_weights=[w for w,ok in zip(weights,accepted_mask) if ok]
        accepted_client_ids=[
            m["client_id"] for m,ok in zip(metadata,accepted_mask) if ok
        ]
        accepted_update_map={
            cid:update for cid,update in zip(accepted_client_ids,accepted_updates)
        }

        (
            verified,consensus_messages,consensus_latency_ms,
            consensus_validators,consensus_byzantine,
        )=pos_bft_verify(
            accepted_update_map,
            histories,
            {cid:hashes[cid] for cid in accepted_client_ids},
            seed+round_index,
        )

        positions=[
            j for j,cid in enumerate(accepted_client_ids)
            if cid in verified
        ]
        if not positions:
            raise RuntimeError(
                f"{run_name}: no client updates accepted at round {round_index+1}"
            )

        accepted_updates=[accepted_updates[j] for j in positions]
        accepted_weights=[accepted_weights[j] for j in positions]
        accepted_client_ids=[accepted_client_ids[j] for j in positions]

        aggregate,normalized_weights=weighted_average(
            accepted_updates,accepted_weights
        )
        aggregate,privacy=add_secure_aggregate_noise(
            aggregate,normalized_weights,target_epsilon,sample_rate,
            CFG.global_rounds,seed+900000+round_index,
        )
        model.load_state_dict(apply_delta(global_state,aggregate),strict=True)
        assert_finite_model(model,f"{run_name} round {round_index+1}")

        for client_id,item in zip(selected,metadata):
            histories[client_id].previous_norm=item["original_norm"]
            if client_id in accepted_client_ids:
                histories[client_id].trust=min(2.0,histories[client_id].trust*1.01)
            else:
                histories[client_id].trust=max(0.05,histories[client_id].trust*0.85)

        validation=evaluate(model,VALSET)

        improved=(
            validation["accuracy"]>best_val_acc+CFG.best_min_delta
            or (
                abs(validation["accuracy"]-best_val_acc)<=CFG.best_min_delta
                and validation["loss"]<best_val_loss
            )
        )

        if improved:
            best_val_acc=float(validation["accuracy"])
            best_val_loss=float(validation["loss"])
            best_round=round_index+1
            best_state=OrderedDict(
                (k,v.detach().cpu().clone()) for k,v in model.state_dict().items()
            )
            rounds_without_improvement=0
            progress(
                f"  ★ NEW BEST → round={best_round} "
                f"val_acc={best_val_acc:.4f} val_loss={best_val_loss:.4f}"
            )
        else:
            rounds_without_improvement+=1

        true_malicious=np.asarray([m["malicious"] for m in metadata],dtype=bool)
        rejected=~accepted_mask
        tp=int(np.sum(rejected&true_malicious))
        fp=int(np.sum(rejected&~true_malicious))
        fn=int(np.sum(~rejected&true_malicious))

        logs.append({
            "run_name":run_name,
            "system":"EPP_BCFL_RECONSTRUCTION_V4",
            "aggregation_mode":aggregation_mode,
            "seed":seed,
            "num_clients":n_clients,
            "round":round_index+1,
            "learning_rate":current_lr,
            "target_epsilon":target_epsilon,
            "accounted_epsilon":privacy["epsilon"],
            "noise_multiplier":privacy["noise_multiplier"],
            "attack":attack,
            "validation_loss":validation["loss"],
            "validation_accuracy":validation["accuracy"],
            "validation_f1":validation["f1"],
            "is_best":bool(improved),
            "best_round_so_far":best_round,
            "best_val_accuracy_so_far":best_val_acc,
            "selected_clients":len(selected),
            "accepted_clients":len(accepted_client_ids),
            "mean_raw_update_norm":float(np.mean([m["original_norm"] for m in metadata])),
            "clipping_fraction":float(np.mean([bool(m["was_clipped"]) for m in metadata])),
            "communication_bytes":sum(update_bytes(u) for u in accepted_updates),
            "consensus_messages":consensus_messages,
            "consensus_latency_ms":consensus_latency_ms,
            "consensus_validators":consensus_validators,
            "consensus_byzantine":consensus_byzantine,
            "consensus_energy_j":blockchain_energy(
                sum(update_bytes(u) for u in accepted_updates),
                consensus_messages,consensus_latency_ms,
            ),
            "detection_seconds":detection_seconds,
            "security_response_seconds":detection_seconds + consensus_latency_ms/1000.0,
            "malicious_detection_recall":tp/(tp+fn) if tp+fn else np.nan,
            "false_positive_rate":fp/max(1,np.sum(~true_malicious)),
        })

        round_seconds=time.perf_counter()-round_started
        round_times.append(round_seconds)
        eta_seconds=float(np.mean(round_times))*(CFG.global_rounds-(round_index+1))

        progress(
            f"EPP ROUND DONE → {run_name} | {round_index+1}/{CFG.global_rounds} | "
            f"val_acc={validation['accuracy']:.4f} | val_f1={validation['f1']:.4f} | "
            f"best={best_val_acc:.4f}@{best_round} | no_improve={rounds_without_improvement} | "
            f"round={round_seconds:.1f}s | ETA≈{eta_seconds/60:.1f} min"
        )

        torch.save({
            "model":OrderedDict(
                (k,v.detach().cpu().clone()) for k,v in model.state_dict().items()
            ),
            "best_state":best_state,
            "best_round":best_round,
            "best_val_acc":best_val_acc,
            "best_val_loss":best_val_loss,
            "rounds_without_improvement":rounds_without_improvement,
            "histories":{cid:asdict(value) for cid,value in histories.items()},
            "logs":logs,
            "next_round":round_index+1,
            "fingerprint":EXPERIMENT_FINGERPRINT,
        },partial_path)

        pd.DataFrame(logs).to_csv(final_metrics_path,index=False)
        maybe_write_resume_bundle(round_index+1)

        if (
            round_index+1>=CFG.early_stop_min_rounds
            and rounds_without_improvement>=CFG.early_stop_patience
        ):
            progress(
                f"EARLY STOP → {run_name} at round {round_index+1}; "
                f"restoring best round {best_round}"
            )
            break

    if best_state is None:
        raise RuntimeError(f"{run_name}: no valid best checkpoint was produced")

    model.load_state_dict(best_state,strict=True)
    final_state=OrderedDict(
        (k,v.detach().cpu().clone()) for k,v in model.state_dict().items()
    )
    torch.save({
        "state_dict":final_state,
        "run_name":run_name,
        "best_round":best_round,
        "best_val_accuracy":best_val_acc,
        "best_val_loss":best_val_loss,
        "config":asdict(CFG),
        "fingerprint":EXPERIMENT_FINGERPRINT,
    },final_model_path)

    if partial_path.exists():
        partial_path.unlink()

    progress(
        f"EPP RUN COMPLETE → {run_name} | BEST round={best_round} "
        f"val_acc={best_val_acc:.4f}"
    )
    return model.cpu(),pd.DataFrame(logs)

# ===================== notebook cell 29 =====================
# B9 Secure-aggregation ring arithmetic validation
# This checks modular mask cancellation and communication overhead.
# It remains a simulation, not production cryptographic SMPC.

def ring_secure_aggregate_vectors(vectors, seed, scale=1_000_000):
    if not vectors:
        raise ValueError("At least one vector is required.")

    arrays = [np.asarray(vector, dtype=np.float64) for vector in vectors]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError(
            "All secure-aggregation vectors must have the same shape."
        )
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError(
            "Secure-aggregation input contains NaN/Inf."
        )

    modulus = np.int64(2**61 - 1)
    quantized = [
        np.mod(
            np.rint(array * scale).astype(np.int64),
            modulus,
        )
        for array in arrays
    ]

    rng = np.random.default_rng(seed)
    masks = [
        rng.integers(
            0,
            modulus,
            size=arrays[0].shape,
            dtype=np.int64,
        )
        for _ in arrays
    ]

    shares = []
    for index, quantized_vector in enumerate(quantized):
        previous_mask = masks[index - 1]
        own_mask = masks[index]
        # Each intermediate remains within signed int64 range.
        share = np.mod(
            quantized_vector + own_mask - previous_mask,
            modulus,
        )
        shares.append(share)

    # Do not use np.sum(..., dtype=int64): summing many modular shares can
    # overflow before the final modulus is applied.
    recovered_q = np.zeros_like(shares[0], dtype=np.int64)
    for share in shares:
        recovered_q = np.mod(
            recovered_q + share,
            modulus,
        )

    signed = np.where(
        recovered_q > modulus // 2,
        recovered_q - modulus,
        recovered_q,
    )
    recovered = signed.astype(np.float64) / scale
    expected = np.sum(arrays, axis=0)

    # Each independently rounded client contributes at most 0.5/scale error.
    tolerance = len(arrays) / (2 * scale) + 1e-12
    maximum_error = float(
        np.max(np.abs(recovered - expected))
    )

    return {
        "max_abs_error": maximum_error,
        "quantization_tolerance": tolerance,
        "recovered": bool(
            np.allclose(
                recovered,
                expected,
                atol=tolerance,
                rtol=0.0,
            )
        ),
        "input_bytes": int(
            sum(array.nbytes for array in arrays)
        ),
        "share_bytes": int(
            sum(share.nbytes for share in shares)
        ),
        "clients": len(arrays),
        "elements": arrays[0].size,
    }


secure_demo_vectors = [
    np.random.default_rng(CFG.seed + client).normal(size=4096)
    for client in range(
        max(4, min(10, CFG.client_grid[0]))
    )
]
SECURE_AGGREGATION_AUDIT = pd.DataFrame([
    ring_secure_aggregate_vectors(
        secure_demo_vectors,
        CFG.seed,
    )
])
SECURE_AGGREGATION_AUDIT.to_csv(
    DIRS["metrics"] / "secure_aggregation_audit.csv",
    index=False,
)

if not bool(SECURE_AGGREGATION_AUDIT.iloc[0]["recovered"]):
    raise AssertionError(
        "Ring secure-aggregation arithmetic validation failed."
    )

display(SECURE_AGGREGATION_AUDIT)
# ===================== notebook cell 30 =====================
def final_row(model,log,run_name,extra=None):
    if log.empty:
        raise RuntimeError(f'No log rows for {run_name}')
    best_round=int(log['best_round_so_far'].iloc[-1]) if 'best_round_so_far' in log.columns else int(log.loc[log['validation_accuracy'].idxmax(),'round'])
    best_log=log[log['round']==best_round].iloc[-1].to_dict()
    test=evaluate(model,TESTSET)
    best_log.update({
        'best_round':best_round,
        'test_accuracy':test['accuracy'],
        'test_precision':test['precision'],
        'test_recall':test['recall'],
        'test_f1':test['f1'],
        'model_path':str(DIRS['models']/f'{run_name}.pt'),
    })
    if extra:
        best_log.update(extra)
    return best_log

def append_summary(filename,rows):
    path=DIRS['metrics']/filename
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    new=pd.DataFrame(rows)
    merged=pd.concat([old,new],ignore_index=True) if not old.empty else new
    if not merged.empty:
        merged=merged.drop_duplicates(['run_name'],keep='last')
    merged.to_csv(path,index=False)
    return merged

# ===================== notebook cell 31 =====================

def completed_run_names():
    """Return run names already completed in the imported/current resume state."""
    names = set()

    # Completed model files are the strongest signal.
    if DIRS["models"].exists():
        for p in DIRS["models"].glob("*.pt"):
            names.add(p.stem)

    # Summary CSVs are an additional signal.
    for csv_name in (
        "epp_clean_all_summary.csv",
        "fedavg_all_summary.csv",
        "epp_privacy_all_summary.csv",
        "epp_attack_all_summary.csv",
        "epp_fgsm_all_summary.csv",
    ):
        p = DIRS["metrics"] / csv_name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "run_name" in df.columns:
                    names.update(df["run_name"].dropna().astype(str).tolist())
            except Exception:
                pass

    return names


def pending_jobs_for_phase(phase, jobs):
    """
    Filter the full job list so already-completed jobs from the resume bundle are skipped.
    """
    completed = completed_run_names()
    pending = []

    if phase in ("CLEAN", "FEDAVG"):
        prefix = "epp_clean" if phase == "CLEAN" else "fedavg"
        for n, seed in jobs:
            name = f"{prefix}_n{n}_seed{seed}"
            if name not in completed:
                pending.append((n, seed))

    elif phase == "PRIVACY":
        for n, eps, seed in jobs:
            en = str(eps).replace(".", "p")
            name = f"epp_privacy_n{n}_eps{en}_seed{seed}"
            if name not in completed:
                pending.append((n, eps, seed))

    elif phase == "ATTACKS":
        for attack, n, seed in jobs:
            name = f"epp_{attack.lower()}_n{n}_seed{seed}"
            if name not in completed:
                pending.append((attack, n, seed))

    elif phase == "FGSM":
        # FGSM outputs are CSV-only, so check summary run_name.
        for n, seed in jobs:
            name = f"epp_fgsm_n{n}_seed{seed}"
            if name not in completed:
                pending.append((n, seed))

    else:
        pending = list(jobs)

    return completed, pending

# ===================== notebook cell 32 (helpers only) =====================

# B12–B18 — Run all required EPP reconstruction experiments sequentially
# Each run auto-resumes from its own Drive checkpoint.

def append_summary_unique(filename, row, keys=('run_name',)):
    path=DIRS['metrics']/filename
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    merged=pd.concat([old,pd.DataFrame([row])],ignore_index=True)
    if all(k in merged.columns for k in keys):
        merged=merged.drop_duplicates(list(keys),keep='last')
    merged.to_csv(path,index=False)
    return merged

def finish_epp_run(name, n, epsilon, seed, attack='NONE', extra=None):
    model,log=run_epp_experiment(
        name,n,epsilon,seed,attack=attack,aggregation_mode='AMA'
    )
    row=final_row(model,log,name,extra or {})
    if 'security_response_seconds' in log.columns:
        row['mean_security_response_seconds']=float(log['security_response_seconds'].mean())
        row['p95_security_response_seconds']=float(log['security_response_seconds'].quantile(.95))
    if 'detection_seconds' in log.columns:
        row['mean_detection_seconds']=float(log['detection_seconds'].mean())
    write_resume_bundle()
    return model,log,row


# ===================== NEW: matched runs only =====================
n=REPRESENTATIVE_CLIENTS
SCEN=[s.strip() for s in os.environ.get('EPP_SCENARIOS','CLEAN').split(',') if s.strip()]
print('matched EPP runs | seeds', SEEDS, '| scenarios', SCEN, '| local epochs', LOCAL_EPOCHS,
      '| classes/client', CLASSES_PER_CLIENT, flush=True)
for seed in SEEDS:
    if 'CLEAN' in SCEN:
        name=f'epp_clean_n{n}_seed{seed}'
        _,_,row=finish_epp_run(name,n,math.inf,seed,attack='NONE',extra={'scenario':'CLEAN'})
        append_summary_unique('epp_matched_clean_summary.csv',row)
        print('RESULT', name, {k:row.get(k) for k in ('test_accuracy','test_f1')}, flush=True)
    for attack in ('MODEL_POISON','DATA_POISON'):
        if attack in SCEN:
            name=f'epp_{attack.lower()}_n{n}_seed{seed}'
            _,_,row=finish_epp_run(name,n,math.inf,seed,attack=attack,
                                   extra={'scenario':attack,'attack_type':attack})
            append_summary_unique('epp_matched_attack_summary.csv',row)
            print('RESULT', name, {k:row.get(k) for k in ('test_accuracy','test_f1')}, flush=True)
print('DONE: matched EPP CIFAR-10 ->', DIRS['metrics'], flush=True)
