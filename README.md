# DL-Traff-NeuralODE: Dynamic Traffic Prediction using STGCN + Neural ODEs

This project extends the baseline DL-Traff framework by introducing Neural Ordinary Differential Equations (Neural ODEs) to dynamically evolve the adjacency matrix in traffic prediction tasks. We apply the model on the METR-LA dataset.

---
## 📋 Project Structure
```
DL-Traff-Graph/
├── METRLA/
│   ├── metr-la.h5
│   ├── adj_mx.pkl
├── workMETRLA/
│   ├── STGCN_NODE.py        # Modified STGCN model using Neural ODE adjacency
│   ├── ODEFunc.py           # ODE Function to evolve adjacency matrix
│   ├── Param.py             # Parameters (e.g., N_NODE, TIMESTEP_IN, etc.)
├── results/                 # Output folder (loss plots, predictions, metrics)
├── Parent paper EDA.ipynb    # Exploratory Data Analysis on original paper 
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🛠 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DL-Traff-Graph.git
cd DL-Traff-Graph
```

2. Install all required packages:
```bash
pip install -r requirements.txt
```

Required libraries:
- `torch`
- `torchdiffeq`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `h5py`
- `tqdm`

3. **Preprocessing**

The METRLA data (`metr-la.h5` and `adj_mx.pkl`) are already provided inside the `METRLA/` folder.  
No additional preprocessing steps are needed manually.

4. **Training**

Open the notebook `Parent paper EDA.ipynb` and follow the sections.

Or alternatively:

- Run `modifiedcode.ipynb` 
- Load dataset
- Train the model using:

```python
from workMETRLA.STGCN_NODE import STGCN
from workMETRLA.ODEFunc import ODEFunc
```

Modify the training loop if needed.

---

## 📈 Results

During training:
- Loss is plotted (loss vs epochs)
- Metrics are computed: MAE, RMSE, MAPE
- Prediction vs Ground Truth plots are saved under `/results`

Example Metrics:

| Metric | Value |
|:------|:------|
| MAE   | ~0.2709 |
| RMSE  | ~0.5098 |
| MAPE  | ~185%  |

---

## ⚠️ Common Errors and Solutions
| Error | Reason | Solution |
|:---|:---|:---|
| `ModuleNotFoundError: No module named 'torchdiffeq'` | Neural ODE library missing | `pip install torchdiffeq` |
| `shape mismatch when flattening A_init` | Wrong reshape during ODE evolution | Flatten before evolving, reshape after |
| `forward() takes 2 positional arguments but 3 were given` | Not updating forward methods in STGCN properly | Pass dynamic A only to spatio conv layers, not output layer |
| `FileNotFoundError: ./METRLA/adj_mx.pkl` | Dataset not in right folder | Make sure `/METRLA` contains `adj_mx.pkl` and `metr-la.h5` |

---

## 🤝 Acknowledgments
This project is based on [DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction](https://arxiv.org/abs/2108.09091).

Developed as part of DS340W Research Paper.


