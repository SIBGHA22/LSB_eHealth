# AUTO-GENERATED from LSB_EHEALTH_COMPLETE_BASE_PAPERS_COLAB.ipynb for local (non-Colab) runs.
# Only the Colab/Drive bootstrap was replaced; the experiment code is the notebook's own.
import os, sys, json, time, shutil, zipfile
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
RESULTS = Path(os.environ.get("LSB_RESULTS", Path(__file__).resolve().parent.parent / "results"))
SMOKE = os.environ.get("LSB_SMOKE", "0") == "1"
MASTER_ROOT = RESULTS / 'transaction_simulation'
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

# ===================== notebook cell 5 =====================
import dataclasses, hashlib, json, math, os, random, shutil, time, warnings, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import kagglehub
warnings.filterwarnings("default")

def progress(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
# ===================== notebook cell 6 =====================

# A0 — DBH configuration (resource-aligned reconstruction)
PROJECT_ID="DBH_CHAIN_REPRO_COLAB_V1"
SEED_PLAN="CORE_3"
SEEDS=(42,43,44)
SAVE_TASK_LEVEL_DETAIL=False

DBH_NODE_COUNTS=(20,50,100,500,1000)
DBH_SECONDS=300 if SMOKE else 3600
MMPS_TRANSITION=((0.90,0.10),(0.20,0.80))
MMPS_RATES=(8.0,25.0)
FOG_SERVICE_RATE=25.0
CLOUD_SERVICE_RATE=45.0
NETWORK_DELAY_MS=(8.0,25.0)
DEADLINE_MS=200.0
RESOURCE_CAPACITY=60
TAMPER_FRACTION=0.05
BLOCKCHAIN_BYZANTINE_FRACTION=0.20
SIMULATED_NETWORK_RTT_MS=5.0
SIMULATED_MESSAGE_PROCESSING_MS=0.01
SIMULATED_HASH_VERIFICATION_MS=0.05

DBH_KAGGLE_HANDLES=(
    "palwashakhanzadi/iot-blockchain-data",
    "kanakbaghel/hospital-management-dataset",
    "prasad22/healthcare-dataset",
)

# Leave blank unless you have the exact verified files/URLs.
CMS_HOSPITAL_EXACT_URL_OR_PATH=""
HEALTHDATA_SERVICE_EXACT_URL_OR_PATH=""
STRICT_ALL_FIVE_SOURCES=False

ROOT=MASTER_ROOT/"dbh_chain"
DIRS={n:ROOT/n for n in ("datasets","checkpoints","models","metrics","plots","logs","manifests")}
for p in DIRS.values():
    p.mkdir(parents=True,exist_ok=True)

print('DBH ROOT:', ROOT, flush=True)

# ===================== notebook cell 7 =====================
SCIENTIFIC_CONFIG={
 "node_counts":DBH_NODE_COUNTS,"seconds":DBH_SECONDS,"seeds":SEEDS,
 "mmps_transition":MMPS_TRANSITION,"mmps_rates":MMPS_RATES,
 "fog_rate":FOG_SERVICE_RATE,"cloud_rate":CLOUD_SERVICE_RATE,
 "network_delay_ms":NETWORK_DELAY_MS,"deadline_ms":DEADLINE_MS,
 "resource_capacity":RESOURCE_CAPACITY,"tamper_fraction":TAMPER_FRACTION,
 "byzantine_fraction":BLOCKCHAIN_BYZANTINE_FRACTION,
 "implementation":"v4-consensus-aware-validation-DBH-only",
}
EXPERIMENT_FINGERPRINT=hashlib.sha256(json.dumps(SCIENTIFIC_CONFIG,sort_keys=True,default=str).encode()).hexdigest()[:16]


manifest_cfg=DIRS['manifests']/ 'scientific_config.json'
if manifest_cfg.exists():
    old_fp=json.loads(manifest_cfg.read_text()).get('fingerprint')
    if old_fp and old_fp!=EXPERIMENT_FINGERPRINT:
        raise RuntimeError(
            f'DBH fingerprint mismatch existing={old_fp} active={EXPERIMENT_FINGERPRINT}. '
            'Use a fresh Drive folder for a different scientific configuration.'
        )
manifest_cfg.write_text(json.dumps({
    'project_id':PROJECT_ID,
    'fingerprint':EXPERIMENT_FINGERPRINT,
    'config':SCIENTIFIC_CONFIG
},indent=2,default=str))
print('DBH experiment fingerprint:',EXPERIMENT_FINGERPRINT,flush=True)

def write_resume_bundle():
    bundle=MASTER_ROOT/'dbh_chain_resume.zip'
    tmp=bundle.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for folder_name in ('checkpoints','models','metrics','manifests','logs','plots'):
            folder=DIRS[folder_name]
            for f in folder.rglob('*'):
                if f.is_file():
                    z.write(f,f.relative_to(ROOT))
    tmp.replace(bundle)
    return bundle

# ===================== notebook cell 8 =====================
@dataclass
class DBHConfig:
    dbh_kaggle_handles: tuple[str,...]=DBH_KAGGLE_HANDLES
    cms_hospital_exact_url_or_path: str=CMS_HOSPITAL_EXACT_URL_OR_PATH
    healthdata_service_exact_url_or_path: str=HEALTHDATA_SERVICE_EXACT_URL_OR_PATH
    strict_dbh_all_five_sources: bool=STRICT_ALL_FIVE_SOURCES
    allow_documented_reconstruction: bool=True
    deadline_ms: float=DEADLINE_MS
    mmps_transition: tuple=MMPS_TRANSITION
    mmps_rates: tuple=MMPS_RATES
    dbh_seconds: int=DBH_SECONDS
    resource_capacity: int=RESOURCE_CAPACITY
    tamper_fraction: float=TAMPER_FRACTION
    committee_size: int=21
    low_battery: float=0.20
    fog_service_rate: float=FOG_SERVICE_RATE
    cloud_service_rate: float=CLOUD_SERVICE_RATE
    network_delay_ms: tuple=NETWORK_DELAY_MS
    blockchain_byzantine_fraction: float=BLOCKCHAIN_BYZANTINE_FRACTION
    simulated_network_rtt_ms: float=SIMULATED_NETWORK_RTT_MS
    simulated_message_processing_ms: float=SIMULATED_MESSAGE_PROCESSING_MS
    simulated_hash_verification_ms: float=SIMULATED_HASH_VERIFICATION_MS
CFG=DBHConfig()

ASSUMPTIONS=pd.DataFrame([
 ["IoT Blockchain Data",DBH_KAGGLE_HANDLES[0],"PAPER_SUPPORTED"],
 ["Hospital Management Dataset",DBH_KAGGLE_HANDLES[1],"PAPER_SUPPORTED"],
 ["CMS Hospital Dataset",CMS_HOSPITAL_EXACT_URL_OR_PATH or "UNRESOLVED","UNRESOLVED_SOURCE"],
 ["HealthData Hospital Service Dataset",HEALTHDATA_SERVICE_EXACT_URL_OR_PATH or "UNRESOLVED","UNRESOLVED_SOURCE"],
 ["Kaggle Healthcare Dataset",DBH_KAGGLE_HANDLES[2],"PAPER_SUPPORTED"],
 ["MMPS/service parameters","paper omits exact values used here","DOCUMENTED_RECONSTRUCTION"],
],columns=["item","value","status"])
ASSUMPTIONS.to_csv(DIRS['manifests']/ 'assumptions.csv',index=False)
display(ASSUMPTIONS)
# ===================== notebook cell 9 =====================
# A1–A2 Load, fetch, profile and inventory all DBH paper sources
import urllib.request

DBH_TABLES = {}
DBH_DATASET_REGISTRY = []

def sha256_file(path: Path, chunk_size: int = 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def read_structured_tables(root: Path):
    tables = {}
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    for file in candidates:
        if not file.is_file():
            continue
        try:
            suffix = file.suffix.lower()
            if suffix == ".csv":
                tables[file.name] = pd.read_csv(file)
            elif suffix == ".parquet":
                tables[file.name] = pd.read_parquet(file)
            elif suffix in {".json", ".jsonl"}:
                tables[file.name] = pd.read_json(file, lines=suffix == ".jsonl")
            elif suffix in {".xlsx", ".xls"}:
                tables[file.name] = pd.read_excel(file)
        except Exception as exc:
            print("Skipped unreadable structured file:", file, type(exc).__name__, exc)
    return tables

def resolve_exact_url_or_path(name: str, value: str):
    record = {"name": name, "source": value, "status": "UNRESOLVED", "path": "", "error": ""}
    if not value:
        record["error"] = (
            "The paper does not identify an exact file. Supply a verified direct URL "
            "or attach the exact file and set its Kaggle path."
        )
        return record, {}
    try:
        candidate = Path(value)
        if candidate.exists():
            local = candidate
        elif value.startswith(("https://", "http://")):
            suffix = Path(value.split("?")[0]).suffix or ".csv"
            local = DIRS["datasets"] / f"{name.lower().replace(' ', '_')}{suffix}"
            urllib.request.urlretrieve(value, local)
        else:
            raise FileNotFoundError(value)
        record.update({"status": "READY", "path": str(local)})
        return record, read_structured_tables(local)
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, {}

def find_attached_dataset(handle: str):
    # Optional local mirror in Drive. Otherwise kagglehub downloads the public dataset.
    slug = handle.split("/")[-1]
    local_root = MASTER_ROOT / "DBH_LOCAL_DATA"
    if local_root.exists():
        exact = [p for p in local_root.rglob(slug) if p.is_dir()]
        if exact:
            return exact[0]
    return None

if True:
    for handle in CFG.dbh_kaggle_handles:
        record = {"name": handle.split("/")[-1], "source": handle, "status": "PENDING", "path": "", "error": ""}
        try:
            attached = find_attached_dataset(handle)
            path = attached if attached is not None else Path(kagglehub.dataset_download(handle))
            record.update({"status": "READY", "path": str(path)})
            DBH_TABLES[handle] = read_structured_tables(path)
        except Exception as exc:
            record.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        DBH_DATASET_REGISTRY.append(record)

    for name, value in (
        ("CMS Hospital Dataset", CFG.cms_hospital_exact_url_or_path),
        ("HealthData Hospital Service Dataset", CFG.healthdata_service_exact_url_or_path),
    ):
        record, tables = resolve_exact_url_or_path(name, value)
        DBH_DATASET_REGISTRY.append(record)
        if tables:
            DBH_TABLES[name] = tables

    registry = pd.DataFrame(DBH_DATASET_REGISTRY)
    registry.to_csv(DIRS["manifests"] / "dbh_dataset_registry.csv", index=False)

    # Deterministic cleaning:
    # - remove exact duplicate rows;
    # - remove rows where every field is missing;
    # - do not blindly impute missing clinical values.
    file_rows = []
    cleaned_tables = {}
    for source, tables in DBH_TABLES.items():
        cleaned_tables[source] = {}
        for table_name, frame in tables.items():
            before_rows = len(frame)
            before_duplicates = int(frame.duplicated().sum())
            before_missing = int(frame.isna().sum().sum())

            cleaned = (
                frame.dropna(how="all")
                .drop_duplicates()
                .reset_index(drop=True)
            )
            cleaned_tables[source][table_name] = cleaned

            file_rows.append({
                "source": source,
                "table": table_name,
                "rows_before": before_rows,
                "rows_after": len(cleaned),
                "columns": len(cleaned.columns),
                "duplicates_removed": before_duplicates,
                "missing_cells_before": before_missing,
                "missing_cells_after": int(cleaned.isna().sum().sum()),
                "all_missing_rows_removed": int(
                    before_rows
                    - len(frame.dropna(how="all"))
                ),
                "missing_value_policy": "preserved_and_logged_no_blind_imputation",
            })

    DBH_TABLES = cleaned_tables
    profiling = pd.DataFrame(file_rows)
    profiling.to_csv(
        DIRS["metrics"] / "dbh_dataset_cleaning_profile.csv",
        index=False
    )

    unresolved = registry[registry["status"] != "READY"]
    if CFG.strict_dbh_all_five_sources and not unresolved.empty:
        raise RuntimeError(
            "Strict five-source mode is enabled, but these exact DBH sources are unresolved:\n"
            + unresolved[["name", "error"]].to_string(index=False)
        )
    if not CFG.allow_documented_reconstruction:
        reconstruction_rows = ASSUMPTIONS[
            ASSUMPTIONS["status"] == "DOCUMENTED_RECONSTRUCTION"
        ]
        if not reconstruction_rows.empty:
            raise RuntimeError(
                "Documented reconstruction is disabled, but the papers omit required settings."
            )

    display(registry)
    display(profiling)
# ===================== notebook cell 10 =====================
# A3–A8 Dataset-to-task mapping, MMPS, queue, Knapsack and consensus validation

PRIORITY_MAP = {
    "critical": 10.0,
    "life threatening": 10.0,
    "emergency": 9.0,
    "urgent": 8.0,
    "severe": 8.0,
    "serious": 7.0,
    "high": 7.0,
    "moderate": 5.0,
    "medium": 5.0,
    "stable": 4.0,
    "low": 3.0,
    "routine": 2.0,
    "normal": 2.0,
}

@dataclass
class HealthcareTask:
    task_id: int
    arrival_time: float
    payload_bytes: int
    priority: float
    resource_requirement: int
    deadline_ms: float
    source: str
    expected_digest: str
    observed_digest: str
    tampered: bool = False


def parse_priority(value):
    """Auditable priority parser; unknown categorical values use neutral priority 5."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 5.0, "missing_neutral"

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return 5.0, "nonfinite_neutral"
        if 0.0 <= number <= 1.0:
            return 1.0 + 9.0 * number, "numeric_unit_interval"
        return float(np.clip(number, 1.0, 10.0)), "numeric_clipped"

    text = str(value).strip().lower()
    for label, score in PRIORITY_MAP.items():
        if label in text:
            return score, f"mapped:{label}"
    return 5.0, "unknown_categorical_neutral"


def source_rows_to_templates(tables):
    templates = []
    mapping_rows = []
    task_id = 0

    for dataset_name, group in tables.items():
        for table_name, frame in group.items():
            if frame.empty:
                continue

            priority_columns = [
                column for column in frame.columns
                if any(token in column.lower() for token in (
                    "priority", "urgent", "severity", "condition", "status"
                ))
            ]

            for _, row in frame.head(5000).iterrows():
                raw = row.to_json(date_format="iso", default_handler=str)
                priority_value = (
                    row[priority_columns[0]]
                    if priority_columns else None
                )
                priority, mapping_rule = parse_priority(priority_value)
                digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

                templates.append(HealthcareTask(
                    task_id=task_id,
                    arrival_time=0.0,
                    payload_bytes=len(raw.encode("utf-8")),
                    priority=priority,
                    resource_requirement=max(1, int(row.notna().sum())),
                    deadline_ms=CFG.deadline_ms,
                    source=f"{dataset_name}:{table_name}",
                    expected_digest=digest,
                    observed_digest=digest,
                ))
                mapping_rows.append({
                    "source": f"{dataset_name}:{table_name}",
                    "task_id": task_id,
                    "priority_column": priority_columns[0] if priority_columns else "",
                    "priority_rule": mapping_rule,
                    "priority": priority,
                    "payload_bytes": len(raw.encode("utf-8")),
                    "resource_requirement": max(1, int(row.notna().sum())),
                })
                task_id += 1

    mapping_audit = pd.DataFrame(mapping_rows)
    mapping_audit.to_csv(
        DIRS["metrics"] / "dbh_dataset_to_task_mapping_audit.csv",
        index=False
    )
    return templates


class MMPSGenerator:
    def __init__(self, transition, rates, seed):
        self.transition = np.asarray(transition, dtype=float)
        self.rates = np.asarray(rates, dtype=float)
        if (
            self.transition.shape != (len(rates), len(rates))
            or not np.allclose(self.transition.sum(axis=1), 1.0)
        ):
            raise ValueError("Invalid MMPS transition matrix")
        self.rng = np.random.default_rng(seed)
        self.state = 0

    def arrivals(self, seconds):
        output = []
        for second in range(seconds):
            count = int(self.rng.poisson(self.rates[self.state]))
            output.extend(second + self.rng.random(count))
            self.state = int(
                self.rng.choice(
                    len(self.rates),
                    p=self.transition[self.state]
                )
            )
        return sorted(output)


def knapsack_select(tasks, capacity):
    if not tasks or capacity <= 0:
        return []
    n = len(tasks)
    dp = np.zeros((n + 1, capacity + 1), dtype=float)

    for i, task in enumerate(tasks, start=1):
        weight = min(capacity, max(1, int(task.resource_requirement)))
        dp[i] = dp[i - 1]
        candidate = (
            dp[i - 1, : capacity + 1 - weight]
            + float(task.priority)
        )
        dp[i, weight:] = np.maximum(dp[i, weight:], candidate)

    selected = []
    remaining = capacity
    for i in range(n, 0, -1):
        if dp[i, remaining] > dp[i - 1, remaining] + 1e-12:
            task = tasks[i - 1]
            selected.append(task)
            remaining -= min(
                capacity,
                max(1, int(task.resource_requirement))
            )
    return list(reversed(selected))


def build_workload(templates, seed):
    if not templates:
        raise RuntimeError(
            "No DBH structured rows were loaded. Enable Kaggle Internet "
            "or attach the paper-listed datasets."
        )

    rng = np.random.default_rng(seed)
    arrivals = MMPSGenerator(
        CFG.mmps_transition,
        CFG.mmps_rates,
        seed
    ).arrivals(CFG.dbh_seconds)

    tasks = []
    for task_id, arrival in enumerate(arrivals):
        template = templates[task_id % len(templates)]
        tampered = bool(rng.random() < CFG.tamper_fraction)
        observed_digest = template.expected_digest
        if tampered:
            observed_digest = hashlib.sha256(
                f"tampered:{template.expected_digest}:{task_id}".encode()
            ).hexdigest()

        tasks.append(dataclasses.replace(
            template,
            task_id=task_id,
            arrival_time=float(arrival),
            tampered=tampered,
            observed_digest=observed_digest,
        ))
    return tasks


def blockchain_energy(payload_bytes, messages, cpu_ms):
    return (
        payload_bytes * 5e-7
        + messages * 2e-4
        + cpu_ms * 2e-5
    )


def consensus_parameters(system, node_count):
    if system == "DBH_POV_RECONSTRUCTION":
        validators = min(
            node_count,
            max(4, int(math.ceil(math.log2(max(node_count, 2)))))
        )
        tolerated = (validators - 1) // 2
        threshold = validators // 2 + 1
        messages = 2 * validators
        phases = 1
    elif system == "LSB_ENERGY_VRF_PBFT":
        validators = min(CFG.committee_size, node_count)
        tolerated = (validators - 1) // 3
        threshold = 2 * tolerated + 1
        messages = 2 * validators * max(0, validators - 1)
        phases = 3
    else:
        raise ValueError(system)

    latency_ms = (
        phases * CFG.simulated_network_rtt_ms
        + messages * CFG.simulated_message_processing_ms
        + CFG.simulated_hash_verification_ms
    )
    return {
        "validators": validators,
        "tolerated": tolerated,
        "threshold": threshold,
        "messages": messages,
        "latency_ms": latency_ms,
    }


def validate_transaction(task, consensus, rng):
    actual_valid = task.expected_digest == task.observed_digest

    validators = consensus["validators"]
    byzantine_count = min(
        validators,
        int(round(validators * CFG.blockchain_byzantine_fraction))
    )
    byzantine_ids = set(
        rng.choice(
            validators,
            size=byzantine_count,
            replace=False
        ).tolist()
    ) if byzantine_count else set()

    votes = []
    for validator_id in range(validators):
        honest_vote = actual_valid
        vote = (
            not honest_vote
            if validator_id in byzantine_ids
            else honest_vote
        )
        votes.append(vote)

    predicted_valid = sum(votes) >= consensus["threshold"]
    return {
        "ground_truth_valid": actual_valid,
        "predicted_valid": predicted_valid,
        "validation_correct": predicted_valid == actual_valid,
        "yes_votes": int(sum(votes)),
        "selected_byzantine": byzantine_count,
    }


def run_blockchain_workload(tasks, node_count, seed, system):
    rng = np.random.default_rng(seed)
    rows = []
    batch_size = max(1, int(np.mean(CFG.mmps_rates)))
    consensus = consensus_parameters(system, node_count)
    consensus_lane_available_ms = 0.0

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        selected_tasks = knapsack_select(batch, CFG.resource_capacity)
        selected_ids = {task.task_id for task in selected_tasks}
        used_capacity = sum(
            task.resource_requirement for task in selected_tasks
        )
        utilization = min(
            1.0,
            used_capacity / max(CFG.resource_capacity, 1)
        )

        preliminary = []
        for queue_position, task in enumerate(batch):
            if task.task_id not in selected_ids:
                rows.append({
                    "system": system,
                    "node_count": node_count,
                    "task_id": task.task_id,
                    "arrival_time": task.arrival_time,
                    "scheduled": False,
                    "tampered": task.tampered,
                    "ground_truth_valid": not task.tampered,
                    "predicted_valid": np.nan,
                    "validation_correct": np.nan,
                    "committed": False,
                    "hybrid_delay_ms": np.inf,
                    "deadline_met": False,
                    "energy_j": 0.0,
                    "consensus_messages": 0,
                    "resource_utilization": utilization,
                    "commit_time_ms": np.nan,
                })
                continue

            layer = "fog" if task.priority >= 5 else "cloud"
            service_rate = (
                CFG.fog_service_rate
                if layer == "fog" else CFG.cloud_service_rate
            )
            queue_ms = (
                queue_position / max(service_rate, 1e-9) * 1000
            )
            service_ms = (
                task.resource_requirement
                / max(service_rate, 1e-9)
                * 1000
            )
            scale = (
                1.0
                + math.log2(max(node_count, 2)) / 20.0
            )
            communication_ms = float(
                rng.uniform(*CFG.network_delay_ms)
            ) * scale
            validation = validate_transaction(task, consensus, rng)
            ready_ms = (
                task.arrival_time * 1000
                + queue_ms
                + service_ms
                + communication_ms
            )
            preliminary.append({
                "task": task,
                "layer": layer,
                "queue_ms": queue_ms,
                "service_ms": service_ms,
                "communication_ms": communication_ms,
                "ready_ms": ready_ms,
                "validation": validation,
                "resource_utilization": utilization,
            })

        if not preliminary:
            continue

        block_start_ms = max(
            consensus_lane_available_ms,
            max(item["ready_ms"] for item in preliminary)
        )
        block_finish_ms = (
            block_start_ms + consensus["latency_ms"]
        )
        consensus_lane_available_ms = block_finish_ms

        for item in preliminary:
            task = item["task"]
            validation = item["validation"]
            storage_ms = float(rng.uniform(2.0, 8.0))
            commit_time_ms = block_finish_ms + storage_ms
            hybrid_ms = (
                commit_time_ms - task.arrival_time * 1000
            )
            committed = bool(validation["predicted_valid"])

            rows.append({
                "system": system,
                "node_count": node_count,
                "task_id": task.task_id,
                "arrival_time": task.arrival_time,
                "scheduled": True,
                "tampered": task.tampered,
                **validation,
                "committed": committed,
                "layer": item["layer"],
                "queue_delay_ms": item["queue_ms"],
                "service_delay_ms": item["service_ms"],
                "communication_delay_ms": item["communication_ms"],
                "consensus_delay_ms": consensus["latency_ms"],
                "storage_delay_ms": storage_ms,
                "hybrid_delay_ms": hybrid_ms,
                "deadline_met": (
                    committed and hybrid_ms <= task.deadline_ms
                ),
                "energy_j": blockchain_energy(
                    task.payload_bytes,
                    consensus["messages"],
                    item["service_ms"] + consensus["latency_ms"],
                ),
                "consensus_messages": consensus["messages"],
                "resource_utilization": item["resource_utilization"],
                "commit_time_ms": commit_time_ms if committed else np.nan,
                "committee_size": consensus["validators"],
                "byzantine_tolerance": consensus["tolerated"],
            })

    return pd.DataFrame(rows)


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def summarize_blockchain(frame, seed):
    scheduled = frame[frame["scheduled"]].copy()
    committed = frame[frame["committed"]].copy()
    eligible = frame[~frame["tampered"]].copy()
    completed_eligible = eligible[eligible["committed"]]
    finite = scheduled[np.isfinite(scheduled["hybrid_delay_ms"])]

    valid_scheduled = scheduled[
        scheduled["ground_truth_valid"] == True
    ]
    invalid_scheduled = scheduled[
        scheduled["ground_truth_valid"] == False
    ]

    last_commit_ms = (
        committed["commit_time_ms"].max()
        if not committed.empty else CFG.dbh_seconds * 1000
    )
    effective_seconds = max(
        CFG.dbh_seconds,
        float(last_commit_ms) / 1000
    )

    true_positive = int(
        (
            (invalid_scheduled["predicted_valid"] == False)
        ).sum()
    )
    false_negative = int(
        (
            (invalid_scheduled["predicted_valid"] == True)
        ).sum()
    )
    false_positive = int(
        (
            (valid_scheduled["predicted_valid"] == False)
        ).sum()
    )

    return {
        "system": frame["system"].iloc[0],
        "node_count": int(frame["node_count"].iloc[0]),
        "seed": seed,
        "submitted": len(frame),
        "eligible_tasks": len(eligible),
        "scheduled": int(frame["scheduled"].sum()),
        "committed": int(frame["committed"].sum()),
        "throughput_tx_s": len(committed) / max(effective_seconds, 1e-9),
        "effective_measurement_seconds": effective_seconds,
        "mean_hybrid_delay_ms": finite["hybrid_delay_ms"].mean(),
        "p95_hybrid_delay_ms": finite["hybrid_delay_ms"].quantile(0.95),
        "scheduling_efficiency_pct": (
            safe_rate(len(completed_eligible), len(eligible)) * 100
        ),
        "scheduler_acceptance_pct": (
            safe_rate(int(frame["scheduled"].sum()), len(frame)) * 100
        ),
        "validation_integrity_pct": (
            scheduled["validation_correct"].mean() * 100
            if not scheduled.empty else np.nan
        ),
        "tamper_detection_recall": safe_rate(
            true_positive,
            true_positive + false_negative,
        ),
        "validation_false_positive_rate": safe_rate(
            false_positive,
            len(valid_scheduled),
        ),
        "invalid_acceptance_rate": safe_rate(
            false_negative,
            len(invalid_scheduled),
        ),
        "deadline_miss_pct": (
            (1 - completed_eligible["deadline_met"].mean()) * 100
            if not completed_eligible.empty else np.nan
        ),
        "mean_resource_utilization": (
            finite["resource_utilization"].mean()
            if not finite.empty else np.nan
        ),
        "energy_per_committed_task_j": (
            committed["energy_j"].mean()
            if not committed.empty else np.nan
        ),
        "mean_consensus_messages": (
            finite["consensus_messages"].mean()
            if not finite.empty else np.nan
        ),
        "mean_consensus_latency_ms": (
            finite["consensus_delay_ms"].mean()
            if not finite.empty else np.nan
        ),
    }
# ===================== notebook cell 11 (first line only) =====================
templates=source_rows_to_templates(DBH_TABLES)
print("task templates:", len(templates), flush=True)

# ===================== NEW: matched simulator =====================
# ============================================================
# A11 — MATCHED transaction-track comparison (paste after cell A10
#        of LSB_EHEALTH_COMPLETE_BASE_PAPERS_COLAB.ipynb)
#
# Why: in the submitted results the two arms were run through two
# different simulators:
#   * DBH arm (this notebook):  service_ms = resource_requirement * 1000/rate,
#     1 consensus lane, capacity 60, consensus messages charged per task.
#   * LSB arm (CRASH_SAFE / V7): service_ms = 1000/rate, 4 shards each with
#     its own capacity 60, consensus messages amortised per block.
# This cell runs BOTH systems through ONE simulator, so every difference
# that remains is attributable to the consensus / selection rule, and the
# effect of sharding is reported separately as an ablation.
#
# Re-uses from earlier cells: CFG, templates (or DBH_TABLES),
# build_workload, knapsack_select, consensus_parameters,
# validate_transaction, blockchain_energy, summarize_blockchain, DIRS.
# ============================================================
import itertools, math
import numpy as np, pandas as pd

MATCHED_SEEDS        = (42,) if SMOKE else (42, 43, 44)
MATCHED_NODE_COUNTS  = (20,) if SMOKE else (20, 100, 1000)
MATCHED_SYSTEMS      = ("DBH_POV_RECONSTRUCTION", "LSB_ENERGY_VRF_PBFT", "LSB_K7")
# LSB_K7 = LSB with the committee size the paper itself specifies (k = 7, Eq. 8 and the testbed),
# instead of the simulator default of 21.
def _consensus_for(system, node_count):
    if system == "LSB_K7":
        old = CFG.committee_size
        try:
            CFG.committee_size = 7
            return consensus_parameters("LSB_ENERGY_VRF_PBFT", node_count)
        finally:
            CFG.committee_size = old
    return consensus_parameters(system, node_count)

MATCHED_SHARDS       = (1, 4)
# 'per_task'          : service time = 1000/rate ms per task (the LSB notebook's model)
# 'per_resource_unit' : service time = requirement * 1000/rate ms (this notebook's DBH model)
MATCHED_SERVICE      = ("per_task", "per_resource_unit")
# 'per_shard'   : every shard gets the full RESOURCE_CAPACITY (what the LSB run did)
# 'split_total' : the same total capacity is divided across shards (equal hardware)
MATCHED_CAPACITY     = ("per_shard", "split_total")

MATCHED_CSV = DIRS['metrics'] / 'matched_transaction_factorial.csv'


def run_matched_workload(tasks, node_count, seed, system,
                         shard_count=1, service_model="per_task",
                         capacity_policy="per_shard"):
    """One simulator for both systems. Only `system` changes the consensus rule."""
    rng = np.random.default_rng(seed)
    consensus = _consensus_for(system, node_count)
    batch_size = max(1, int(np.mean(CFG.mmps_rates)))
    if capacity_policy == "per_shard":
        capacity = int(CFG.resource_capacity)
    elif capacity_policy == "split_total":
        capacity = max(1, int(CFG.resource_capacity) // shard_count)
    else:
        raise ValueError(capacity_policy)

    lane_free_ms = [0.0] * shard_count
    rows = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        per_shard = {s: [] for s in range(shard_count)}
        for t in batch:
            per_shard[t.task_id % shard_count].append(t)

        for sid, shard_tasks in per_shard.items():
            if not shard_tasks:
                continue
            selected = knapsack_select(shard_tasks, capacity)
            sel_ids = {t.task_id for t in selected}
            used = sum(t.resource_requirement for t in selected)
            util = min(1.0, used / max(capacity, 1))

            prelim = []
            for qpos, t in enumerate(shard_tasks):
                if t.task_id not in sel_ids:
                    rows.append({
                        "system": system, "node_count": node_count, "task_id": t.task_id,
                        "arrival_time": t.arrival_time, "scheduled": False, "tampered": t.tampered,
                        "ground_truth_valid": not t.tampered, "predicted_valid": np.nan,
                        "validation_correct": np.nan, "committed": False,
                        "hybrid_delay_ms": np.inf, "deadline_met": False, "energy_j": 0.0,
                        "consensus_messages": 0, "resource_utilization": util,
                        "commit_time_ms": np.nan, "consensus_delay_ms": np.nan,
                    })
                    continue
                rate = CFG.fog_service_rate if t.priority >= 5 else CFG.cloud_service_rate
                unit_ms = 1000.0 / max(rate, 1e-9)
                if service_model == "per_task":
                    service_ms = unit_ms
                    queue_ms = qpos * unit_ms
                elif service_model == "per_resource_unit":
                    service_ms = t.resource_requirement * unit_ms
                    queue_ms = qpos * unit_ms          # same queue rule as the DBH cell
                else:
                    raise ValueError(service_model)
                comm_ms = float(rng.uniform(*CFG.network_delay_ms)) * (
                    1.0 + math.log2(max(node_count, 2)) / 20.0)
                val = validate_transaction(t, consensus, rng)
                ready = t.arrival_time * 1000.0 + queue_ms + service_ms + comm_ms
                prelim.append((t, val, service_ms, ready, util))

            if not prelim:
                continue
            block_start = max(lane_free_ms[sid], max(p[3] for p in prelim))
            block_finish = block_start + consensus["latency_ms"]
            lane_free_ms[sid] = block_finish
            # identical accounting for both systems: consensus cost paid once per block
            msgs_per_tx = consensus["messages"] / len(prelim)
            cons_ms_per_tx = consensus["latency_ms"] / len(prelim)

            for t, val, service_ms, ready, util in prelim:
                commit = block_finish + float(rng.uniform(2.0, 8.0))
                hybrid = commit - t.arrival_time * 1000.0
                committed = bool(val["predicted_valid"])
                rows.append({
                    "system": system, "node_count": node_count, "task_id": t.task_id,
                    "arrival_time": t.arrival_time, "scheduled": True, "tampered": t.tampered,
                    **val, "committed": committed, "hybrid_delay_ms": hybrid,
                    "deadline_met": committed and hybrid <= t.deadline_ms,
                    "energy_j": blockchain_energy(t.payload_bytes, msgs_per_tx,
                                                  service_ms + cons_ms_per_tx),
                    "consensus_messages": msgs_per_tx, "consensus_delay_ms": consensus["latency_ms"],
                    "resource_utilization": util, "commit_time_ms": commit if committed else np.nan,
                })
    return pd.DataFrame(rows)


# ---- run the factorial (resumable) ----
done = pd.read_csv(MATCHED_CSV) if MATCHED_CSV.exists() else pd.DataFrame()
key_cols = ["system", "node_count", "seed", "shards", "service_model", "capacity_policy"]
done_keys = set(map(tuple, done[key_cols].astype(str).values)) if not done.empty else set()

out_rows = []
for seed in MATCHED_SEEDS:
    workload = build_workload(templates, seed)
    for nodes, system, shards, svc, cap in itertools.product(
            MATCHED_NODE_COUNTS, MATCHED_SYSTEMS, MATCHED_SHARDS,
            MATCHED_SERVICE, MATCHED_CAPACITY):
        if shards == 1 and cap == "split_total":
            continue                                    # identical to per_shard when shards == 1
        k = tuple(map(str, (system, nodes, seed, shards, svc, cap)))
        if k in done_keys:
            continue
        frame = run_matched_workload(workload, nodes, seed, system, shards, svc, cap)
        row = summarize_blockchain(frame, seed)
        row.update({"shards": shards, "service_model": svc, "capacity_policy": cap})
        out_rows.append(row)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(MATCHED_CSV, index=False)
        print(f"seed={seed} nodes={nodes} {system[:3]} shards={shards} {svc} {cap}: "
              f"thr={row['throughput_tx_s']:.2f} delay={row['mean_hybrid_delay_ms']:.1f} "
              f"miss={row['deadline_miss_pct']:.1f}% sched={row['scheduling_efficiency_pct']:.1f}%",
              flush=True)

res = pd.read_csv(MATCHED_CSV)
METRICS = ["throughput_tx_s", "mean_hybrid_delay_ms", "p95_hybrid_delay_ms",
           "deadline_miss_pct", "scheduling_efficiency_pct",
           "energy_per_committed_task_j", "mean_consensus_messages"]
summary = (res.groupby(["service_model", "shards", "capacity_policy", "system"])[METRICS]
              .mean().round(4).reset_index())
summary.to_csv(DIRS['metrics'] / 'matched_transaction_summary.csv', index=False)
display(summary)

# The rows to report in the paper:
#   (1) matched comparison: same service model, same shards, same capacity -> consensus/selection effect only
#   (2) sharding ablation: LSB shards=4 vs shards=1 under 'split_total' (equal hardware) and 'per_shard'
#   (3) sensitivity: repeat (1) under the other service model

print("DONE: matched transaction factorial ->", DIRS["metrics"], flush=True)
