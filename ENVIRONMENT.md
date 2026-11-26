# Environment used for the released results

| Item | Value |
|---|---|
| OS | Ubuntu (64-bit) |
| GPU | NVIDIA RTX 4000 Ada Generation, 20 GB, compute capability sm_89 |
| Python | 3.14, in a virtual environment |
| Deep learning | PyTorch + torchvision with CUDA (installed from pytorch.org) |
| Consensus testbed and Fabric deployment | Docker Engine; Go 1.21+ |

Exact package versions are in `requirements-lock.txt` (repository root) (output of `pip freeze` on the machine that
produced the results). Small differences in GPU, driver or library versions can change individual
runs slightly; the reported means and confidence intervals should be reproduced within their intervals.

Datasets are downloaded by the scripts on first run and are not redistributed:
- PhysioNet/CinC Challenge 2017 (`training2017.zip`, physionet.org) - ECG track
- CIFAR-10 (torchvision) - EPP-BCFL track
- Kaggle: `palwashakhanzadi/iot-blockchain-data`, `kanakbaghel/hospital-management-dataset`,
  `prasad22/healthcare-dataset` - DBH-Chain transaction track (needs a Kaggle account for kagglehub)
