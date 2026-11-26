from __future__ import annotations
# AUTO-GENERATED from experiments/notebooks/lsb_cifar_security_privacy.ipynb for local runs.
# Only the Colab/Drive bootstrap was replaced; the training code is the notebook's own,
# which produced the paper's seed-42 LSB eHealth CIFAR-10 result (88.19%).
# Usage:  LSB_CIFAR_SEEDS=42,43,44,45,46 python run_lsb_cifar.py
import os, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
RESULTS = Path(os.environ.get("LSB_RESULTS", Path(__file__).resolve().parent.parent / "results"))
SMOKE = os.environ.get("LSB_SMOKE", "0") == "1"
SYNTH = os.environ.get("LSB_SYNTHETIC_DATA", "0") == "1"   # developer self-test only
RUN_SEEDS = tuple(int(s) for s in os.environ.get("LSB_CIFAR_SEEDS", "42,43,44,45,46").split(","))
def display(*objs, **kw):
    for o in objs:
        try: print(o.to_string())
        except Exception: print(o)

# ===================== notebook cell 1 =====================
import copy, dataclasses, hashlib, json, math, os, random, shutil, time, warnings, zipfile
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import kagglehub
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from reedsolo import RSCodec
warnings.filterwarnings('default')

def progress(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}",flush=True)

ROOT = RESULTS / 'cifar_lsb'
DATA_CACHE_ROOT = Path(os.environ.get("LSB_DATA_CACHE", RESULTS / 'data_cache'))
KAGGLE_CACHE_ROOT = DATA_CACHE_ROOT / 'kaggle'
CIFAR_CACHE_ROOT = DATA_CACHE_ROOT / 'cifar10'
RESUME_BUNDLE_FILE = RESULTS / 'cifar_lsb_resume.zip'
for p in (ROOT, DATA_CACHE_ROOT, CIFAR_CACHE_ROOT): p.mkdir(parents=True, exist_ok=True)
print("RESULTS ROOT:", ROOT, "| seeds:", RUN_SEEDS, "| smoke:", SMOKE, flush=True)

# ===================== notebook cell 2 (config, Drive part removed) =====================
PROJECT_ID='LSB_EHEALTH_DEADLINE_V6'
PHASE='ONE_SHOT_FINAL'       # BLOCKCHAIN_PILOT | BLOCKCHAIN | CLEAN | PRIVACY | ATTACKS | FGSM | ABLATION | MIA | FINAL
BATCH_ID=0
BATCH_SIZE=1
SEED_PLAN='DEADLINE_1'
RESUME_BUNDLE_PATH=''
RESUME_EVERY_ROUNDS=10
DBH_BASE_RESULTS_PATH=''        # only needed for FINAL
EPP_BASE_RESULTS_PATH=''        # only needed for FINAL

SEEDS=(42,)
CLIENT_GRID=(10,)
REPRESENTATIVE_CLIENTS=10
GLOBAL_ROUNDS=2 if SMOKE else 50
LOCAL_EPOCHS=4
CLIENT_FRACTION=1.0
CLASSES_PER_CLIENT=8
LOCAL_BATCH_SIZE=64
LEARNING_RATE=0.05; MIN_LEARNING_RATE_FACTOR=0.02; MOMENTUM=0.9; WEIGHT_DECAY=5e-4
TARGET_EPSILONS=(0.1,0.5,1.0,5.0); DELTA=1e-5; CLIENT_CLIP_NORM=25.0; CLIP_CLEAN_UPDATES=False; MAX_AGG_WEIGHT_FACTOR=2.0
ADVERSARIAL_CLIENT_FRACTION=0.20; DATA_POISON_FRACTION=0.30; MODEL_POISON_SCALE=-5.0; FGSM_EPSILON=8/255; ANOMALY_MAD_THRESHOLD=3.5
COMMITTEE_SIZE=21; HIGH_BATTERY=0.60; LOW_BATTERY=0.20; TOP_K_FRACTION=0.10; MAX_CLUSTER_SIZE=5; RECLUSTER_EVERY_ROUNDS=10; SHARD_COUNT=4; GOSSIP_FANOUT=3
VALIDATION_FRACTION=0.02; SHADOW_POOL_SIZE=0; NUM_WORKERS=0
MIA_SHADOW_MODELS=1
MIA_SHADOW_EPOCHS=3
MIA_TARGET_SAMPLES_PER_CLASS=1000; PRIVACY_AUC_TARGET=0.55

# DBH-aligned blockchain track
DBH_NODE_COUNTS=(20,50,100,500,1000); DBH_SECONDS=3600
# LSB stress/capacity-validation track (separate from DBH-aligned comparison)
STRESS_PATIENTS=500
STRESS_TX_PER_PATIENT_PER_SEC=1.0
STRESS_OVERHEAD_FRACTION=0.20
STRESS_SECONDS=60
STRESS_BLOCK_WINDOW_MS=50.0
STRESS_SERVICE_RATE_PER_SHARD=600.0
STRESS_PAYLOAD_BYTES=512
STRESS_TAMPER_FRACTION=0.05
MMPS_TRANSITION=((0.90,0.10),(0.20,0.80)); MMPS_RATES=(8.0,25.0)
FOG_SERVICE_RATE=25.0; CLOUD_SERVICE_RATE=45.0; NETWORK_DELAY_MS=(8.0,25.0); DEADLINE_MS=200.0; RESOURCE_CAPACITY=60; TAMPER_FRACTION=0.05
BLOCKCHAIN_BYZANTINE_FRACTION=0.20; SIMULATED_NETWORK_RTT_MS=5.0; SIMULATED_MESSAGE_PROCESSING_MS=0.01; SIMULATED_HASH_VERIFICATION_MS=0.05
DBH_KAGGLE_HANDLES=('palwashakhanzadi/iot-blockchain-data','kanakbaghel/hospital-management-dataset','prasad22/healthcare-dataset')
CMS_HOSPITAL_EXACT_URL_OR_PATH=''; HEALTHDATA_SERVICE_EXACT_URL_OR_PATH=''; STRICT_ALL_FIVE_SOURCES=False

DIRS={n:ROOT/n for n in ('datasets','checkpoints','models','metrics','plots','logs','manifests')}
for p in DIRS.values(): p.mkdir(parents=True,exist_ok=True)

# ===================== notebook cell 3 =====================
SCIENTIFIC_CONFIG={
 'seed_plan':SEED_PLAN,'seeds':SEEDS,'client_grid':CLIENT_GRID,'representative_clients':REPRESENTATIVE_CLIENTS,
 'global_rounds':GLOBAL_ROUNDS,'local_epochs':LOCAL_EPOCHS,'client_fraction':CLIENT_FRACTION,'classes_per_client':CLASSES_PER_CLIENT,
 'optimizer':['SGD',LEARNING_RATE,MOMENTUM,WEIGHT_DECAY,'cosine',MIN_LEARNING_RATE_FACTOR],'validation_fraction':VALIDATION_FRACTION,'shadow_pool_size':SHADOW_POOL_SIZE,
 'dp':[TARGET_EPSILONS,DELTA,CLIENT_CLIP_NORM],
 'clip_clean_updates':CLIP_CLEAN_UPDATES,
 'lsb':[COMMITTEE_SIZE,LOW_BATTERY,TOP_K_FRACTION,MAX_CLUSTER_SIZE,SHARD_COUNT,GOSSIP_FANOUT],
 'dbh_track':[DBH_NODE_COUNTS,DBH_SECONDS,MMPS_TRANSITION,MMPS_RATES],
 'stress_track':[STRESS_PATIENTS,STRESS_TX_PER_PATIENT_PER_SEC,STRESS_OVERHEAD_FRACTION,STRESS_SECONDS,STRESS_BLOCK_WINDOW_MS,STRESS_SERVICE_RATE_PER_SHARD,SHARD_COUNT],
 'architecture':'CIFAR WideResNet-16-4 GroupNorm reconstruction',
 'fl_method':'Deadline clean comparison: moderate limited-class non-IID; adaptive clustering coordination; sample-count FedAvg; no DP',
 'recluster_every_rounds':RECLUSTER_EVERY_ROUNDS
}
EXPERIMENT_FINGERPRINT=hashlib.sha256(json.dumps(SCIENTIFIC_CONFIG,sort_keys=True,default=str).encode()).hexdigest()[:16]

def import_resume():
    # Primary persistence is the Drive ROOT itself.
    clean_summary = DIRS['metrics'] / 'lsb_clean_all_summary.csv'
    clean_model = DIRS['models'] / 'lsb_clean_n10_seed42.pt'

    if clean_summary.exists() or any(DIRS['checkpoints'].glob('*.pt')):
        print('✅ Existing Google Drive state detected; resume will use saved checkpoints.', flush=True)
        return

    # Optional fallback: restore the last Drive ZIP if ROOT is empty.
    if RESUME_BUNDLE_FILE.exists():
        print('Restoring Drive resume ZIP:', RESUME_BUNDLE_FILE, flush=True)
        with zipfile.ZipFile(RESUME_BUNDLE_FILE, 'r') as z:
            z.extractall(ROOT)
        print('✅ Drive resume ZIP restored.', flush=True)
    else:
        print('No previous Drive state found. Starting fresh once.', flush=True)

import_resume()

config_path=DIRS['manifests']/ 'scientific_config.json'
if config_path.exists():
    old_fp=json.loads(config_path.read_text()).get('fingerprint')
    if old_fp and old_fp!=EXPERIMENT_FINGERPRINT: raise RuntimeError(f'Fingerprint mismatch imported={old_fp} active={EXPERIMENT_FINGERPRINT}')
config_path.write_text(json.dumps({'project_id':PROJECT_ID,'fingerprint':EXPERIMENT_FINGERPRINT,'config':SCIENTIFIC_CONFIG},indent=2,default=str))
print('Experiment fingerprint:',EXPERIMENT_FINGERPRINT,flush=True)
# ===================== notebook cell 4 =====================
def detect_device():
    if not torch.cuda.is_available(): return torch.device('cpu'),{'reason':'CUDA unavailable'}
    try:
        major,minor=torch.cuda.get_device_capability(0); arch=f'sm_{major}{minor}'; supported=list(torch.cuda.get_arch_list())
        if False: return torch.device('cpu'),{'gpu':torch.cuda.get_device_name(0),'reason':f'{arch} unsupported'}
        x=torch.ones(1,device='cuda')+1; torch.cuda.synchronize(); del x
        return torch.device('cuda'),{'gpu':torch.cuda.get_device_name(0),'arch':arch}
    except Exception as exc: return torch.device('cpu'),{'reason':str(exc)}
DEVICE,DEVICE_DETAILS=detect_device(); CUDA_USABLE=DEVICE.type=='cuda'
print('Selected device:',DEVICE,DEVICE_DETAILS,flush=True)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if CUDA_USABLE:
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
seed_everything(SEEDS[0])

if SYNTH:
    from PIL import Image as _Img
    class _Fake(torch.utils.data.Dataset):
        def __init__(self, root, train=True, download=True, transform=None):
            g=np.random.default_rng(1 if train else 2); n=800 if train else 200
            self.data=g.integers(0,256,(n,32,32,3),dtype=np.uint8); self.targets=[int(t) for t in g.integers(0,10,n)]; self.transform=transform
        def __len__(self): return len(self.targets)
        def __getitem__(self,i):
            x=_Img.fromarray(self.data[i]); return (self.transform(x) if self.transform else x), self.targets[i]
    datasets.CIFAR10=_Fake

# ===================== notebook cell 5 =====================
@dataclass
class LSBConfig:
    seed: int=42
    force_rerun: bool=False
    num_workers: int=NUM_WORKERS
    learning_rate: float=LEARNING_RATE; min_learning_rate_factor: float=MIN_LEARNING_RATE_FACTOR; momentum: float=MOMENTUM; weight_decay: float=WEIGHT_DECAY
    local_batch_size: int=LOCAL_BATCH_SIZE; local_epochs: int=LOCAL_EPOCHS; global_rounds: int=GLOBAL_ROUNDS; client_fraction: float=CLIENT_FRACTION
    non_iid_method: str='LIMITED_CLASSES'; classes_per_client: int=CLASSES_PER_CLIENT
    max_aggregation_weight_factor: float=MAX_AGG_WEIGHT_FACTOR; delta: float=DELTA; client_clip_norm: float=CLIENT_CLIP_NORM; clip_clean_updates: bool=CLIP_CLEAN_UPDATES
    adversarial_client_fraction: float=ADVERSARIAL_CLIENT_FRACTION; data_poison_fraction: float=DATA_POISON_FRACTION; model_poison_scale: float=MODEL_POISON_SCALE
    anomaly_mad_threshold: float=ANOMALY_MAD_THRESHOLD; pos_bft_byzantine_fraction: float=BLOCKCHAIN_BYZANTINE_FRACTION
    committee_size: int=COMMITTEE_SIZE; high_battery: float=HIGH_BATTERY; low_battery: float=LOW_BATTERY; top_k_fraction: float=TOP_K_FRACTION; max_cluster_size: int=MAX_CLUSTER_SIZE; recluster_every_rounds: int=RECLUSTER_EVERY_ROUNDS; shard_count: int=SHARD_COUNT; gossip_fanout: int=GOSSIP_FANOUT
    deadline_ms: float=DEADLINE_MS; mmps_transition: tuple=MMPS_TRANSITION; mmps_rates: tuple=MMPS_RATES; dbh_seconds: int=DBH_SECONDS; resource_capacity: int=RESOURCE_CAPACITY; tamper_fraction: float=TAMPER_FRACTION
    fog_service_rate: float=FOG_SERVICE_RATE; cloud_service_rate: float=CLOUD_SERVICE_RATE; network_delay_ms: tuple=NETWORK_DELAY_MS; blockchain_byzantine_fraction: float=BLOCKCHAIN_BYZANTINE_FRACTION
    simulated_network_rtt_ms: float=SIMULATED_NETWORK_RTT_MS; simulated_message_processing_ms: float=SIMULATED_MESSAGE_PROCESSING_MS; simulated_hash_verification_ms: float=SIMULATED_HASH_VERIFICATION_MS
    mia_shadow_models: int=MIA_SHADOW_MODELS; mia_shadow_epochs: int=MIA_SHADOW_EPOCHS; mia_target_samples_per_class: int=MIA_TARGET_SAMPLES_PER_CLASS; privacy_auc_target: float=PRIVACY_AUC_TARGET
    dbh_kaggle_handles: tuple[str,...]=DBH_KAGGLE_HANDLES
    cms_hospital_exact_url_or_path: str=CMS_HOSPITAL_EXACT_URL_OR_PATH
    healthdata_service_exact_url_or_path: str=HEALTHDATA_SERVICE_EXACT_URL_OR_PATH
    strict_dbh_all_five_sources: bool=STRICT_ALL_FIVE_SOURCES
    stress_patients: int=STRESS_PATIENTS
    stress_tx_per_patient_per_sec: float=STRESS_TX_PER_PATIENT_PER_SEC
    stress_overhead_fraction: float=STRESS_OVERHEAD_FRACTION
    stress_seconds: int=STRESS_SECONDS
    stress_block_window_ms: float=STRESS_BLOCK_WINDOW_MS
    stress_service_rate_per_shard: float=STRESS_SERVICE_RATE_PER_SHARD
    stress_payload_bytes: int=STRESS_PAYLOAD_BYTES
    stress_tamper_fraction: float=STRESS_TAMPER_FRACTION
    allow_documented_reconstruction: bool=True
CFG=LSBConfig(); CFG.client_grid=CLIENT_GRID

def stage_checkpoint(name): return DIRS['checkpoints']/f'{name}.pt'
def stage_csv(name): return DIRS['metrics']/f'{name}.csv'
def validate_checkpoint_fingerprint(state,name):
    if state.get('fingerprint')!=EXPERIMENT_FINGERPRINT: raise RuntimeError(f'Checkpoint fingerprint mismatch: {name}')
def load_pipeline_checkpoint(path,map_location='cpu'):
    p=Path(path).resolve()
    if ROOT.resolve() not in p.parents and p!=ROOT.resolve(): raise RuntimeError(f'Unsafe checkpoint path {p}')
    progress(f'CHECKPOINT LOAD → {p.name}')
    try: return torch.load(p,map_location=map_location,weights_only=False)
    except TypeError: return torch.load(p,map_location=map_location)
def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def write_resume_bundle():
    bundle=RESUME_BUNDLE_FILE
    tmp=bundle.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for folder_name in ('checkpoints','models','metrics','manifests','logs','plots'):
            for f in DIRS[folder_name].rglob('*'):
                if f.is_file() and not f.name.endswith('.tmp'):
                    z.write(f,f.relative_to(ROOT))
    os.replace(tmp,bundle)
    return bundle
def maybe_write_resume_bundle(round_number):
    if round_number%RESUME_EVERY_ROUNDS==0 or round_number==GLOBAL_ROUNDS: progress(f'RESUME SNAPSHOT → {write_resume_bundle()}')
# ===================== notebook cell 7 =====================
def detect_cifar_root():
    CIFAR_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print('CIFAR-10 cache:', CIFAR_CACHE_ROOT, flush=True)
    return str(CIFAR_CACHE_ROOT)
CIFAR_ROOT=detect_cifar_root(); MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
train_transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD),transforms.RandomErasing(p=0.35,scale=(0.02,0.12),ratio=(0.5,2.0),value='random')])
eval_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
train_aug=datasets.CIFAR10(CIFAR_ROOT,train=True,download=True,transform=train_transform)
train_eval=datasets.CIFAR10(CIFAR_ROOT,train=True,download=True,transform=eval_transform)
test_eval=datasets.CIFAR10(CIFAR_ROOT,train=False,download=True,transform=eval_transform)
rng=np.random.default_rng(42); all_idx=rng.permutation(len(train_eval)); shadow_indices=np.asarray(all_idx[:SHADOW_POOL_SIZE]); target_pool=np.asarray(all_idx[SHADOW_POOL_SIZE:]); target_labels=np.asarray(train_eval.targets)[target_pool]
train_indices,val_indices=train_test_split(target_pool,test_size=VALIDATION_FRACTION,random_state=42,stratify=target_labels); test_indices=np.arange(len(test_eval))
VALSET=Subset(train_eval,list(map(int,val_indices))); TESTSET=Subset(test_eval,list(map(int,test_indices)))
print({'official_train':len(train_eval),'target_train':len(train_indices),'validation':len(val_indices),'shadow':len(shadow_indices),'official_test':len(test_eval)},flush=True)
# ===================== notebook cell 8 =====================
# B2 Shared reconstruction CNN with GroupNorm — no BatchNorm running-stat leakage
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
        self.layer1 = nn.Sequential(WideBasic(16, 64, 1), WideBasic(64, 64, 1))
        self.layer2 = nn.Sequential(WideBasic(64, 128, 2), WideBasic(128, 128, 1))
        self.layer3 = nn.Sequential(WideBasic(128, 256, 2), WideBasic(256, 256, 1))
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
    return all(not torch.is_floating_point(v) or torch.isfinite(v).all().item() for v in model.state_dict().values())


def assert_finite_model(model, context="model"):
    bad = [k for k, v in model.state_dict().items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if bad:
        raise FloatingPointError(f"{context} has non-finite tensors: {bad[:10]}")


@torch.no_grad()
def evaluate(model, dataset, adversarial_epsilon=0.0):
    model = model.to(DEVICE).eval()
    assert_finite_model(model, "evaluation model")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=CFG.num_workers, pin_memory=CUDA_USABLE)
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
                x = torch.maximum(torch.minimum(x_adv.detach() + epsilon_normalized * gradient.sign(), upper), lower)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits in evaluation batch {batch_id}")
        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
        truth.extend(y.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(truth, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(truth, predicted).tolist(),
    }
# ===================== notebook cell 9 =====================
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
# ===================== notebook cell 10 =====================
def blockchain_energy(payload_bytes,messages,cpu_ms): return payload_bytes*5e-7 + messages*2e-4 + cpu_ms*2e-5
# ===================== notebook cell 11 =====================
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
# ===================== notebook cell 12 =====================
# C3 Local training, AMA weights, anomaly filtering and PoS+BFT simulation
@dataclass
class ClientHistory:
    trust: float = 1.0
    previous_norm: float | None = None


def train_local(global_model, indices, seed, label_poison=False, learning_rate=None):
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
    quality = 1.0 / (1.0 + max(0.0, local_loss))
    stability = 1.0
    if history.previous_norm is not None:
        stability = math.exp(-abs(norm - history.previous_norm) / (history.previous_norm + 1e-9))
    return max(1e-9, samples * quality * history.trust * stability)


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
# ===================== notebook cell 13 =====================
# C2–C5 PUF, sharding, Energy-VRF selection and PBFT utilities
class SoftwarePUF:
    def __init__(self, simulation_secret: bytes):
        self.secret = simulation_secret

    def derive_key(self, device_id: str, challenge: bytes, salt: bytes):
        response = hashlib.sha256(self.secret + device_id.encode() + challenge).digest()
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"lsb-ehealth").derive(response)

PUF = SoftwarePUF(hashlib.sha256(f"simulation-{CFG.seed}".encode()).digest())

@dataclass
class Validator:
    node_id: int
    battery: float
    trust: float
    online: bool = True
    byzantine: bool = False


def energy_vrf_commit(validators, seed_material):
    eligible = [v for v in validators if v.online and v.battery >= CFG.low_battery]
    k = min(CFG.committee_size, len(eligible))
    if k < 4:
        raise RuntimeError("Insufficient validators")
    weights = np.asarray([max(1e-9, v.battery * v.trust) for v in eligible])
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big"))
    selected = [eligible[i] for i in rng.choice(len(eligible), size=k, replace=False, p=weights / weights.sum())]
    tolerated = (k - 1) // 3
    byzantine = sum(v.byzantine for v in selected)
    return {
        "committed": byzantine <= tolerated,
        "committee_size": k,
        "byzantine": byzantine,
        "tolerated": tolerated,
        "messages": 2 * k * (k - 1),
        "selected_ids": [validator.node_id for validator in selected],
    }


def erasure_metrics(payload: bytes, parity_bytes=10):
    codec = RSCodec(parity_bytes)
    started = time.perf_counter(); encoded = codec.encode(payload); encode_ms = (time.perf_counter() - started) * 1000
    corrupted = bytearray(encoded)
    for i in range(min(parity_bytes // 2, len(corrupted))):
        corrupted[i] ^= 1
    started = time.perf_counter(); decoded = codec.decode(bytes(corrupted))[0]; decode_ms = (time.perf_counter() - started) * 1000
    return {"original_bytes": len(payload), "encoded_bytes": len(encoded), "encode_ms": encode_ms, "decode_ms": decode_ms, "recovered": decoded == payload}

def assign_shard(record_type: str, device_id: str):
    semantic = {
        "ecg": 0,
        "medication": 1,
        "image": 2,
        "general": 3,
    }
    return semantic.get(record_type, int(hashlib.sha256(device_id.encode()).hexdigest(), 16)) % CFG.shard_count

def gossip_propagation(node_count: int, fanout: int, seed: int):
    rng = np.random.default_rng(seed)
    informed = {0}
    rounds = 0
    messages = 0
    while len(informed) < node_count and rounds < node_count:
        new_nodes = set(informed)
        for source in list(informed):
            candidates = [n for n in range(node_count) if n not in new_nodes]
            if not candidates:
                continue
            selected = rng.choice(
                candidates,
                size=min(fanout, len(candidates)),
                replace=False
            )
            new_nodes.update(map(int, selected))
            messages += len(selected)
        if new_nodes == informed:
            break
        informed = new_nodes
        rounds += 1
    return {
        "node_count": node_count,
        "fanout": fanout,
        "gossip_rounds": rounds,
        "gossip_messages": messages,
        "coverage": len(informed) / node_count,
    }
# ===================== notebook cell 14 =====================
# D2 Sparse updates and adaptive clustering

def top_k_sparsify(update, fraction):
    sparse = OrderedDict(); bytes_sent = 0
    for key, value in update.items():
        flat = value.flatten()
        k = max(1, math.ceil(flat.numel() * fraction))
        if k < flat.numel():
            threshold = torch.topk(flat.abs(), k, sorted=False).values.min()
            result = torch.where(flat.abs() >= threshold, flat, torch.zeros_like(flat))
        else:
            result = flat.clone()
        sparse[key] = result.reshape_as(value)
        bytes_sent += int(torch.count_nonzero(result)) * 8
    sparse, _ = clip_update(sparse, CFG.client_clip_norm)
    return sparse, bytes_sent


def cluster_feature_matrix(client_ids, batteries, coordinates, updates):
    vectors = [sampled_update_vector(updates[cid], maximum=2048) for cid in client_ids]
    maximum = max(map(len, vectors))
    matrix = np.zeros((len(vectors), maximum), dtype=np.float32)
    for i, vector in enumerate(vectors):
        matrix[i, :len(vector)] = vector
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    battery_column = np.asarray([batteries[cid] for cid in client_ids], dtype=np.float32).reshape(-1, 1)
    coordinate_matrix = StandardScaler().fit_transform(np.asarray([coordinates[cid] for cid in client_ids], dtype=np.float32))
    features = np.hstack([battery_column, coordinate_matrix, matrix])
    if not np.isfinite(features).all():
        raise FloatingPointError("Clustering features contain NaN/Inf")
    return features


def methodology_cluster_assignments(client_ids, batteries, coordinates, updates):
    """
    Documented reconstruction of the methodology's three clustering criteria.

    Group formation uses model-update cosine information plus geographic coordinates.
    Battery > HIGH_BATTERY is used to select the cluster head. Exact clustering
    linkage/normalization is not specified by the methodology, so AgglomerativeClustering
    is used transparently.
    """
    client_ids = list(client_ids)
    number = len(client_ids)
    n_clusters = max(1, math.ceil(number / CFG.max_cluster_size))

    vectors = [sampled_update_vector(updates[cid], maximum=2048) for cid in client_ids]
    maximum = max(map(len, vectors))
    update_matrix = np.zeros((number, maximum), dtype=np.float32)
    for i, vector in enumerate(vectors):
        update_matrix[i, :len(vector)] = vector
    update_matrix /= np.linalg.norm(update_matrix, axis=1, keepdims=True) + 1e-12

    gps = np.asarray([coordinates[cid] for cid in client_ids], dtype=np.float32)
    gps_scaled = StandardScaler().fit_transform(gps) if number > 1 else np.zeros_like(gps)

    features = np.hstack([gps_scaled, update_matrix])
    if n_clusters == 1:
        labels = np.zeros(number, dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)

    assignments = {cid: int(label) for cid, label in zip(client_ids, labels)}
    heads = {}
    for label in sorted(set(labels.tolist())):
        members = [cid for cid in client_ids if assignments[cid] == label]
        eligible = [cid for cid in members if batteries[cid] > CFG.high_battery]
        pool = eligible if eligible else members
        heads[int(label)] = max(pool, key=lambda cid: batteries[cid])

    return assignments, heads


def cluster_labels(features):
    number = len(features)
    clusters = max(1, math.ceil(number / CFG.max_cluster_size))
    return np.zeros(number, dtype=int) if clusters == 1 else AgglomerativeClustering(n_clusters=clusters).fit_predict(features)
# ===================== notebook cell 15 =====================
# C9–C11 Adaptive clustering, DP protection and resumable LSB offline trainer

def global_round_lr(round_index):
    frac = round_index / max(1, CFG.global_rounds - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
    factor = CFG.min_learning_rate_factor + (1.0 - CFG.min_learning_rate_factor) * cosine
    return CFG.learning_rate * factor


def run_lsb_experiment(run_name, n_clients, target_epsilon, seed, attack="NONE", component_flags=None):
    run_started = time.perf_counter()
    progress(
        f"LSB RUN START → {run_name} | clients={n_clients} | "
        f"epsilon={target_epsilon} | attack={attack}"
    )
    component_flags = component_flags or {
        "clustering": True,
        "energy_aware": True,
    }
    seed_everything(seed)
    final_model_path = DIRS["models"] / f"{run_name}.pt"
    final_metrics_path = stage_csv(run_name)
    partial_path = stage_checkpoint(run_name)
    if final_model_path.exists() and final_metrics_path.exists() and not CFG.force_rerun:
        checkpoint = load_pipeline_checkpoint(final_model_path, map_location="cpu")
        validate_checkpoint_fingerprint(checkpoint, final_model_path.name)
        model = make_model(); model.load_state_dict(checkpoint["state_dict"])
        existing_log = pd.read_csv(final_metrics_path)
        progress(
            f"LSB RUN REUSED → {run_name} | completed rounds={len(existing_log)}"
        )
        return model.cpu(), existing_log

    partitions = client_partitions(n_clients, seed)
    model = make_model(); trainable = trainable_keys(model)
    rng = np.random.default_rng(seed)
    histories = {cid: ClientHistory() for cid in partitions}
    batteries = {cid: float(rng.uniform(0.20, 1.0)) for cid in partitions}
    coordinates = {
        cid: (
            float(rng.uniform(0, 200)),
            float(rng.uniform(0, 200)),
        )
        for cid in partitions
    }
    validator_count = max(CFG.committee_size, n_clients)
    validators = [
        Validator(
            node_id=index,
            battery=float(rng.uniform(CFG.low_battery, 1.0)),
            trust=float(rng.uniform(0.70, 1.0)),
            byzantine=False,
        )
        for index in range(validator_count)
    ]
    configured_byzantine = int(
        round(
            validator_count
            * (
                CFG.adversarial_client_fraction
                if attack != "NONE"
                else CFG.pos_bft_byzantine_fraction
            )
        )
    )
    for validator in validators[:configured_byzantine]:
        validator.byzantine = True

    logs, start_round = [], 0
    if partial_path.exists() and not CFG.force_rerun:
        state = load_pipeline_checkpoint(partial_path, map_location="cpu")
        validate_checkpoint_fingerprint(state, partial_path.name)
        model.load_state_dict(state["model"])
        histories = {int(k): ClientHistory(**v) for k, v in state["histories"].items()}
        batteries = {int(k): float(v) for k, v in state["batteries"].items()}
        coordinates = {int(k): tuple(v) for k, v in state["coordinates"].items()}
        logs = state["logs"]
        start_round = state["next_round"]
        progress(
            f"LSB RESUME → {run_name} | next round={start_round + 1}/{CFG.global_rounds}"
        )

    malicious_count = int(round(n_clients * CFG.adversarial_client_fraction))
    malicious = set(np.random.default_rng(seed + 701).choice(n_clients, size=malicious_count, replace=False).tolist()) if malicious_count else set()
    sample_rate = CFG.client_fraction

    round_times = []
    cluster_assignments = None
    cluster_heads = {}
    for round_index in range(start_round, CFG.global_rounds):
        round_started = time.perf_counter()
        current_lr = global_round_lr(round_index)
        progress(
            f"LSB ROUND START → {run_name} | "
            f"{round_index + 1}/{CFG.global_rounds} | lr={current_lr:.6f}"
        )
        round_rng = np.random.default_rng(seed + round_index * 100003)
        selected_count = max(1, math.ceil(n_clients * CFG.client_fraction))
        selected = round_rng.choice(n_clients, size=selected_count, replace=False).tolist()
        global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
        updates, metadata = {}, {}
        client_progress_step = max(1, len(selected) // 4)
        for client_position, cid in enumerate(selected, start=1):
            local_state, info = train_local(
                model,
                partitions[cid],
                seed + round_index * 10000 + cid,
                label_poison=cid in malicious and attack == "DATA_POISON",
                learning_rate=current_lr,
            )
            update = state_delta(local_state, global_state, trainable)
            if cid in malicious and attack == "MODEL_POISON":
                update = OrderedDict((k, v * CFG.model_poison_scale) for k, v in update.items())
            original_norm = update_norm(update)
            should_clip = (not math.isinf(target_epsilon)) or CFG.clip_clean_updates or attack != "NONE"
            if should_clip:
                update, _ = clip_update(update, CFG.client_clip_norm)
            # LSB methodology compresses medical payloads (ECG/images/text),
            # not FL model updates. Keep the model update intact here.
            bytes_sent = update_bytes(update)
            updates[cid] = update
            metadata[cid] = {**info, "original_norm": original_norm, "bytes": bytes_sent, "malicious": cid in malicious}
            if (
                client_position == 1
                or client_position % client_progress_step == 0
                or client_position == len(selected)
            ):
                progress(
                    f"  local clients → {client_position}/{len(selected)} "
                    f"for {run_name} round {round_index + 1}"
                )

        # Re-form adaptive clusters every 10 rounds as specified.
        if component_flags.get("clustering", True):
            if cluster_assignments is None or round_index % CFG.recluster_every_rounds == 0:
                cluster_assignments, cluster_heads = methodology_cluster_assignments(
                    selected, batteries, coordinates, updates
                )
        else:
            cluster_assignments = {cid: 0 for cid in selected}
            eligible = [cid for cid in selected if batteries[cid] > CFG.high_battery]
            pool = eligible if eligible else selected
            cluster_heads = {0: max(pool, key=lambda cid: batteries[cid])}

        labels = np.asarray([cluster_assignments[cid] for cid in selected], dtype=int)

        # Accuracy-gate aggregation:
        # clustering still performs coordination / head selection, but the clean
        # global model is updated with standard sample-count FedAvg across clients.
        cluster_updates = [updates[cid] for cid in selected]
        cluster_weights = [float(metadata[cid]["samples"]) for cid in selected]

        accepted_mask = robust_acceptance(cluster_updates, enabled=attack != "NONE")
        accepted_cluster_updates = [u for u, ok in zip(cluster_updates, accepted_mask) if ok]
        accepted_cluster_weights = [w for w, ok in zip(cluster_weights, accepted_mask) if ok]
        if not accepted_cluster_updates:
            raise RuntimeError(f"{run_name}: all clusters rejected")
        aggregate, normalized_weights = weighted_average(accepted_cluster_updates, accepted_cluster_weights)
        # Direct client-level sample-count FedAvg is used in the V6 accuracy/deadline profile.
        # normalized_weights already are the effective client contribution weights.
        # The previous hierarchical-cluster code left cluster_inner_weights empty,
        # causing max([]) after round 1.
        max_effective_weight = float(np.max(normalized_weights))
        aggregate, privacy = add_secure_aggregate_noise(
            aggregate,
            normalized_weights,
            target_epsilon,
            sample_rate,
            CFG.global_rounds,
            seed + 800000 + round_index,
            max_effective_weight=max_effective_weight,
        )

        consensus_started = time.perf_counter()
        consensus = energy_vrf_commit(
            validators,
            f"{run_name}:{seed}:{round_index}",
        )
        consensus_latency_ms = (
            (time.perf_counter() - consensus_started) * 1000
            + 3 * CFG.simulated_network_rtt_ms
            + consensus["messages"] * CFG.simulated_message_processing_ms
        )
        consensus_energy_j = blockchain_energy(
            update_bytes(aggregate),
            consensus["messages"],
            consensus_latency_ms,
        )

        if consensus["committed"]:
            model.load_state_dict(
                apply_delta(global_state, aggregate),
                strict=True,
            )
        assert_finite_model(
            model,
            f"{run_name} round {round_index + 1}",
        )
        validation = evaluate(model, VALSET)

        for cid in selected:
            histories[cid].previous_norm = metadata[cid]["original_norm"]
            batteries[cid] = max(0.0, batteries[cid] - 0.001)
        logs.append({
            "run_name": run_name,
            "system": "LSB_EHEALTH",
            "component_flags": json.dumps(component_flags, sort_keys=True),
            "seed": seed,
            "num_clients": n_clients,
            "round": round_index + 1,
            "learning_rate": current_lr,
            "target_epsilon": target_epsilon,
            "accounted_epsilon": privacy["epsilon"],
            "noise_multiplier": privacy["noise_multiplier"],
            "attack": attack,
            "clusters": len(set(labels.tolist())),
            "cluster_heads": json.dumps({int(k): int(v) for k, v in cluster_heads.items()}, sort_keys=True),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_f1": validation["f1"],
            "communication_bytes": sum(metadata[cid]["bytes"] for cid in selected),
            "mean_battery": float(np.mean(list(batteries.values()))),
            "consensus_committed": consensus["committed"],
            "committee_size": consensus["committee_size"],
            "selected_byzantine": consensus["byzantine"],
            "byzantine_tolerance": consensus["tolerated"],
            "consensus_messages": consensus["messages"],
            "consensus_latency_ms": consensus_latency_ms,
            "consensus_energy_j": consensus_energy_j,
        })
        round_seconds = time.perf_counter() - round_started
        round_times.append(round_seconds)
        remaining_rounds = CFG.global_rounds - (round_index + 1)
        eta_seconds = float(np.mean(round_times)) * remaining_rounds
        progress(
            f"LSB ROUND DONE → {run_name} | "
            f"{round_index + 1}/{CFG.global_rounds} | "
            f"val_acc={validation['accuracy']:.4f} | "
            f"val_f1={validation['f1']:.4f} | "
            f"round={round_seconds:.1f}s | "
            f"ETA current run≈{eta_seconds/60:.1f} min"
        )
        atomic_torch_save({
            "model": OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items()),
            "histories": {cid: asdict(value) for cid, value in histories.items()},
            "batteries": batteries,
            "coordinates": coordinates,
            "logs": logs,
            "next_round": round_index + 1,
            "fingerprint": EXPERIMENT_FINGERPRINT,
        }, partial_path)
        pd.DataFrame(logs).to_csv(final_metrics_path, index=False)
        maybe_write_resume_bundle(round_index + 1)

    final_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    atomic_torch_save({"state_dict": final_state, "run_name": run_name, "config": asdict(CFG), "fingerprint": EXPERIMENT_FINGERPRINT}, final_model_path)
    model.load_state_dict(final_state)
    return model.cpu(), pd.DataFrame(logs)
# ===================== notebook cell 19 =====================
def final_row(model,log,run_name,extra=None):
    last=log.sort_values('round').iloc[-1].to_dict(); test=evaluate(model,TESTSET)
    last.update({'test_accuracy':test['accuracy'],'test_precision':test['precision'],'test_recall':test['recall'],'test_f1':test['f1'],'model_path':str(DIRS['models']/f'{run_name}.pt')})
    if extra: last.update(extra)
    return last

def append_summary(filename,rows,keys=('run_name',)):
    path=DIRS['metrics']/filename; old=pd.read_csv(path) if path.exists() else pd.DataFrame(); new=pd.DataFrame(rows)
    merged=pd.concat([old,new],ignore_index=True) if not old.empty else new
    if not merged.empty: merged=merged.drop_duplicates(list(keys),keep='last')
    merged.to_csv(path,index=False); return merged

# ===================== local controller: clean LSB eHealth runs over seeds =====================
import math
rows=[]
out_csv = DIRS['metrics'] / 'lsb_clean_multiseed_summary.csv'
for seed in RUN_SEEDS:
    name=f'lsb_clean_n10_seed{seed}'
    model, log = run_lsb_experiment(name, 10, math.inf, seed, attack='NONE')
    row = final_row(model, log, name); row['seed']=seed; rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    progress(f"SEED {seed} DONE test_acc={row['test_accuracy']:.4f} test_f1={row['test_f1']:.4f}")
df=pd.DataFrame(rows)
print(df[['seed','test_accuracy','test_precision','test_recall','test_f1']].to_string(index=False))
print('mean acc', df.test_accuracy.mean(), 'sd', df.test_accuracy.std(ddof=1))
print('DONE: LSB CIFAR multiseed ->', out_csv)
