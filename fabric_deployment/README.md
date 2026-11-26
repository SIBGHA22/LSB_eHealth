# Hyperledger Fabric service path (paper Table 13, Fig. 10)

Executable transaction track: software-emulated PUF key derivation, encrypted Reed-Solomon
storage, the VRF committee consensus service (k = 7, quorum 5) and the ledger chaincode.

| Folder | Contents |
|---|---|
| `src/puf/` | PUF emulator, service and benchmark (Python) |
| `src/storage/` | Storage core and service: Zstandard, AES-GCM, Reed-Solomon (Go) |
| `src/consensus/energy-vrf-pbft/` | VRF committee selection and quorum certificate service (Go) |
| `src/ledger-chaincode/` | Fabric chaincode (Go); build with `go build -o lsb-ledger .` |
| `docker/` | Compose file for the seven validator containers |
| `configs/` | Fabric shard configuration |
| `third_party/install-fabric.sh` | Official Fabric installer (downloads fabric-samples and binaries) |
| `scripts/` | End-to-end benchmark and result extraction |
| `evidence/final/` | Measured runs used in the paper |

Steps (run from this folder):
```bash
./third_party/install-fabric.sh docker binary samples     # Fabric test network
docker compose -f docker/consensus-validators.compose.yml up -d --build
python3 scripts/end_to_end_benchmark.py                     # writes evidence/final/end-to-end-benchmark.json
python3 scripts/generate_paper_metrics.py                   # writes evidence/final/LSB_metrics.csv
python3 scripts/make_deployment_figures.py
```
The benchmark was measured with no federated training running on the host, and the storage
service uses an internal benchmark key rather than the PUF-derived key.
