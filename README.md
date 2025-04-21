# DL-Traff-NeuralODE: Dynamic Traffic Prediction using STGCN + Neural ODEs
by Jai abhishek singh and Karthik kuppala
The Pennsylvania state University 

This project extends the **DL-Traff-Graph** benchmark by injecting **Neural Ordinary Differential Equations (Neural ODEs)** into the **STGCN** model for dynamic adjacency matrix learning, enabling better modeling of evolving urban traffic patterns.

# 📂 Project Structure

| Folder/File | Description |
|:---|:---|
| `workMETRLA/STGCN_NODE.py` | **Modified STGCN model** that accepts dynamically evolved graphs (Neural ODE version). |
| `workMETRLA/ODEFunc.py` | Defines the Neural ODE function that evolves the adjacency matrix. |
| `workMETRLA/Training_STGCN_NODE.ipynb` | **Training notebook**: Loads data, applies Neural ODE evolution, trains the STGCN-Node model. |
| `workMETRLA/METR-LA/` | **Dataset folder** containing `metr-la.h5` and adjacency matrix CSV. |
| `workMETRLA/utils/` | Helper scripts, metrics computation functions, etc. |
| `requirements.txt` | List of packages required. |

---

# ⚙️ Requirements

First, lets create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate.bat # On Windows
```

- install the dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
torch>=1.10
torchdiffeq
numpy
pandas
matplotlib
scikit-learn
h5py
scipy
```

---

# 📥 Dataset

Download the **METR-LA traffic dataset** from [here](https://github.com/liyaguang/DCRNN/tree/master/data)  
and place the following files into:

```
workMETRLA/METR-LA/
├── adj_mx.csv
├── metr-la.h5
```

---

# 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jailancer/DL-Traff-Graph
   cd DL-Traff-Graph
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training**:

   ```bash
   workMETRLA/Training_STGCN_NODE.ipynb
   ```
   - This notebook will:
     - Load METR-LA dataset
     - Preprocess features
     - Initialize and evolve adjacency matrix via Neural ODE
     - Train the modified STGCN model
     - Save the model metrics (MAE, RMSE, MAPE) to `STGCN_NODE_metrics.txt`
     - Plot the training loss curve

5. **Evaluate**:
   - After training finishes (20 epochs), the model automatically computes evaluation metrics (MAE, RMSE, MAPE) and saves them.
   - Visualizes prediction vs ground-truth plots for specific sensors.

---

# 📊 Results

| Metric | Score |
|:---|:---|
| MAE | ~0.244 |
| RMSE | ~0.495 |
| MAPE | ~184% |

(Results depend slightly on random seeds.)

---

# ⚠️ Important Notes

- Always **evolve adjacency once per epoch**, not every batch.
- Neural ODE evolution is relatively heavy — training takes ~40–60 minutes depending on GPU/CPU.
- Make sure the folder structure is **preserved exactly** before running.

---

# 🙌 Contact

If you encounter any issue or have questions regarding the implementation:

- Email : jaiabhishek2060@gmail.com
- GitHub Profile: https://github.com/jailancer

---

