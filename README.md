# LSB eHealth

Code, logs and results for the paper
**LSB eHealth: A Lightweight Scalable Blockchain Framework for Robust Federated e-Healthcare Services**.

Every comparison re-implements the baselines under matched settings: within one comparison, only the
method under test differs. All result files used in the paper are in `results/`, so every table can be
checked without re-running anything (step 6).

## Repository layout

| Folder | Contents |
|---|---|
| `experiments/` | Training and simulation code (`run_all.sh` runs everything) |
| `experiments/transaction_simulation/` | DBH-Chain PoV vs LSB eHealth transaction simulation (Table 7) |
| `experiments/ecg_robustness/` | PhysioNet 2017 ECG: 7 aggregation rules x 6 attack conditions x 6 seeds (Tables 9-11, Figs. 5, 6, 8) |
| `experiments/cifar_epp/` | CIFAR-10: EPP-BCFL re-implementation, seeds 42-46 (Table 8) |
| `experiments/cifar_lsb/` | CIFAR-10: LSB eHealth, seeds 42-46 (Table 8) |
| `experiments/notebooks/` | Colab notebooks for the security and privacy evaluation (Table 12, Fig. 9, Sec. 10.10), ECG communication volume (Fig. 7) and the first baseline reconstruction |
| `consensus_testbed/` | Docker network comparing DBH-Chain PoV and LSB eHealth consensus (Table 14, Fig. 11) |
| `fabric_deployment/` | Hyperledger Fabric service path: PUF, storage, consensus, chaincode (Table 13, Fig. 10) |
| `results/` | Result files (CSV/JSON) and run logs used in the paper |
| `paper_figures/` | Scripts that rebuild the paper's tables and figures from `results/` |

## 1. Requirements

- Ubuntu 22.04 or later, one NVIDIA GPU with at least 16 GB memory (results were produced on an RTX 4000 Ada, 20 GB)
- Python 3.11 or later
- Docker Engine (for `consensus_testbed/` and `fabric_deployment/`); Go 1.21+ is optional (the build falls back to a Go Docker image)
- About 20 GB of free disk space for datasets
- A Kaggle account for the DBH-Chain workload datasets (`kagglehub` asks for your API token on first use)

Full hardware and package details: `ENVIRONMENT.md`; exact package versions: `requirements-lock.txt`.

## 2. Install

```bash
git clone https://github.com/SIBGHA22/LSB_eHealth.git
cd LSB_eHealth
python3 -m venv ~/lsb-venv
source ~/lsb-venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # choose the CUDA build for your driver
pip install -r requirements.txt
```

Datasets are downloaded automatically on first run and are not redistributed: PhysioNet/CinC 2017
(`training2017.zip`), CIFAR-10 (torchvision), and three Kaggle datasets (see `ENVIRONMENT.md`).

## 3. Quick check (about 20 minutes)

```bash
cd experiments
LSB_SMOKE=1 LSB_RESULTS=results_smoke ./run_all.sh
```
Each script runs two rounds with one seed. The last line must be `ALL DONE`, and no line may say `FAILED`.
Smoke-test numbers are not meaningful.

## 4. Controlled experiments (about 12-14 hours)

```bash
cd experiments
nohup ./run_all.sh > run_all.out 2>&1 &
tail -f run_all.out          # progress; per-experiment logs are in experiments/results/logs/
```

| Step | Script | Time (RTX 4000 Ada) | Output folder in `experiments/results/` |
|---|---|---|---|
| Transaction simulation | `transaction_simulation/run_dbh_matched.py` | ~20 min | `transaction_simulation/` |
| ECG robustness | `ecg_robustness/run_ecg_robust.py` | ~3 h | `ecg_robustness/` |
| CIFAR-10 EPP-BCFL | `cifar_epp/run_epp_matched.py` | ~4 h | `cifar_epp/` |
| CIFAR-10 LSB eHealth | `cifar_lsb/run_lsb_cifar.py` | ~4 h | `cifar_lsb/` |

Each script can also be run on its own, for example `python ecg_robustness/run_ecg_robust.py`.

Options (environment variables):

| Variable | Meaning | Default |
|---|---|---|
| `ECG_ARMS` | Aggregation rules to run, e.g. `LSB_V2,MEDIAN` | all seven |
| `ECG_ATTACKS` | Conditions, e.g. `NONE,SIGNFLIP6` | all six |
| `ECG_REF_ARM` | Rule the paired tests compare against | `LSB_V2` |
| `ECG_EVAL_EVERY` | Evaluate every N rounds (1 for the convergence figure; does not change results) | 5 |
| `EPP_SEEDS`, `LSB_CIFAR_SEEDS` | CIFAR-10 seeds | `42,43,44,45,46` |
| `LSB_RESULTS` | Output folder | `experiments/results` |
| `LSB_DATA_CACHE` | Reuse an existing dataset download | `<LSB_RESULTS>/data_cache` |

Names used in the code and result files:
- Aggregation rules: `FEDAVG_CLIP` (clipped FedAvg), `MEDIAN`, `TRIMMED_MEAN`, `MULTI_KRUM`, `FLTRUST`,
  `LSB` ("LSB eHealth (v1)" in the paper, norm screen only), `LSB_V2` ("LSB eHealth", always-on norm and direction screen).
- Conditions: `NONE` (clean), `SIGNFLIP6` (x(-6) sign flip), `NORM_MATCHED_FLIP`, `ALIE`, `IPM`, `LABEL_FLIP30`.

Runs are seeded per (rule, condition, seed), so running a subset gives the same rows as the full run.
The released ECG results were produced in two passes with the same code: `runs_pass1.csv` (all rules except
`LSB_V2`) and `runs_pass2.csv` (repeats those rows and adds `LSB_V2`).

## 5. Deployed experiments (Docker)

**Consensus comparison (Table 14, Fig. 11), about 2-4 hours:**
```bash
cd consensus_testbed
./build.sh                         # builds bin/node and bin/bench
QUICK=1 ./run_compare.sh           # 5-minute check
MODES="lsb dbh" ./run_compare.sh   # full grid: N = 20, 50; link delay 0, 5 ms; 5 seeds; 3 fault scenarios
python3 summarize.py               # writes results/summary.csv and results/summary.md
```
Run only one `run_compare.sh` at a time (containers are named `node-<i>`). The paper uses the 5 ms rows.

**Hyperledger Fabric service path (Table 13, Fig. 10):** see `fabric_deployment/README.md`.
It was measured with no federated training running on the host.

**Notebooks:** the files in `experiments/notebooks/` run in Google Colab with a GPU runtime
(Runtime > Change runtime type > GPU); open one and choose Runtime > Run all.

## 6. Check the results against the paper

```bash
cd paper_figures
python3 make_tables.py ../results out/    # LaTeX for Tables 9-11 and 14
python3 make_figs.py   ../results out/    # Figs. 5, 6 and 11
```
To check a new run, point the scripts at it, e.g. `python3 make_tables.py ../experiments/results out/`.

Values to expect (mean over seeds; small differences within the reported confidence intervals are normal
on other GPUs or library versions):

| Paper | Quantity | Expected value | File |
|---|---|---|---|
| Table 7 | DBH-Chain, 1 lane: throughput / mean delay / scheduling efficiency | 4.17 tx/s / 888 ms / 31.8% | `results/transaction_simulation/matched_transaction_summary.csv` |
| Table 8 | EPP-BCFL matched clean accuracy | 0.8868 ± 0.0050 | `results/cifar_epp/epp_matched_clean_summary.csv` |
| Table 9 | LSB eHealth macro-F1 under x(-6) sign flip | 0.805 (worst seed 0.722) | `results/ecg_robustness/runs_pass2.csv` |
| Table 9 | Clipped FedAvg macro-F1 under x(-6) sign flip | 0.471 | same |
| Table 14 | Median latency at 5 ms links, 1 client | about 12 ms for both systems | `results/consensus_testbed/summary.md` |
| Table 14 | Unsafe commits per 1000 tx, N = 20, 30% Byzantine | DBH-Chain 18.6, LSB eHealth 2.0 | same |

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` (e.g. `cryptography`, `reedsolo`) | `pip install -r requirements.txt` inside the activated venv |
| `No DBH structured rows were loaded` | Internet access and a Kaggle API token are needed for `kagglehub` |
| GPU not found / very slow | Check `nvidia-smi` and install the PyTorch build that matches your CUDA driver |
| Docker `permission denied` | `sudo usermod -aG docker $USER`, then log out and back in |
| Docker `container name already in use` | `docker rm -f $(docker ps -aq --filter label=lsbdbh=1)` |
| A run stopped part-way | Re-run the same command; finished runs are reused where the script supports resuming |

## Notes for reviewers

- FLTrust receives a 200-record root set from the validation split; no other rule gets server-side data.
- On ECG, "clipped FedAvg" is the arm called the reconstructed EPP-BCFL baseline in the earlier manuscript version.
- The device PUF is software-emulated (HMAC/HKDF); it is not a hardware PUF.
- The Fabric and consensus testbeds run on one host with emulated link delay.
