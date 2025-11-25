# **Neural Collaborative Filtering (NCF) Recommender System**

## **Project Overview**

This project implements a **Neural Collaborative Filtering (NCF)** recommender system based on **He et al., 2017 (NeuMF)**. It predicts user–item interactions using a **deep learning architecture** that combines:

* **Generalized Matrix Factorization (GMF)**: captures linear user–item relationships.
* **Multi-Layer Perceptron (MLP)**: captures non-linear interaction patterns.
* **NeuMF (Fusion)**: combines GMF + MLP for superior performance.

The system is trained on the **MovieLens 1M dataset** using **implicit feedback** (ratings ≥ 4 treated as positive interactions). This project demonstrates:

* Building **modular, production-ready ML pipelines**
* Implementing **state-of-the-art recommender architectures**
* Evaluating models using **ranking-based metrics (HR@10, NDCG@10)**

---

## **Project Structure**

```
ncf-recommender/
│
├── data/
│   ├── raw/                  # Original MovieLens data
│   ├── processed/            # Processed train/test CSVs
│   └── movielens_1m.csv
│
├── src/
│   ├── data_prep.py          # Load & preprocess data
│   ├── sampler.py            # Negative sampling for implicit feedback
│   ├── model.py              # GMF, MLP, NeuMF architectures
│   ├── train.py              # Training loops
│   ├── evaluate.py           # Evaluation metrics (HR, NDCG)
│   └── inference.py          # Generate top-k recommendations
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_train_debug.ipynb  # Experimentation & debugging
│
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## **Evaluation Metrics**

The models were evaluated on **leave-one-out test data** with **99 negative items per user**.

| Model | HR@10  | NDCG@10 |
| ----- | ------ | ------- |
| GMF   | 0.4661 | 0.2389  |
| MLP   | 0.4101 | 0.2060  |
| NeuMF | 0.5030 | 0.2660  |

**Interpretation:**

* **GMF**: Captures linear interactions; solid baseline.
* **MLP**: Learns non-linear patterns; slightly weaker than GMF on this dataset.
* **NeuMF**: Combines GMF + MLP, achieving the best performance.

> NeuMF demonstrates the effectiveness of **orchestrating linear + non-linear branches** to improve recommendations.

---

## **Installation Instructions**

1. Clone the repository:

```bash
git clone <your-repo-url>
cd ncf-recommender
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download MovieLens 1M dataset:

```bash
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip -d data/raw/
```

---

## **Usage**

### **1. Preprocess Data**

```python
from src.data_prep import preprocess_movielens

train_df, test_df, num_users, num_items = preprocess_movielens("data/raw/ratings.dat")
```

### **2. Train Model**

```python
from src.model import NeuMF
from src.train import train_neumf

model = NeuMF(num_users=num_users, num_items=num_items).to(device)
train_neumf(model, train_loader, epochs=5)
```

### **3. Evaluate Model**

```python
from src.evaluate import evaluate_model

hr, ndcg = evaluate_model(model, test_user_item, user_pos_items, K=10)
print(f"NeuMF — HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")
```

### **4. Generate Recommendations**

```python
from src.inference import recommend_for_user

top_items = recommend_for_user(model, user_id=5, top_k=10)
print(top_items)
```

---

## **Architecture Diagram**

```
User ID → Embedding ──────┐
                          │
Item ID → Embedding ──────┤
GMF branch → element-wise ┘
                            \
MLP branch → concat → Dense → ReLU layers → \
                                               → Concatenate → Dense → Sigmoid → Prediction
```

* GMF captures **linear interactions**
* MLP captures **non-linear patterns**
* NeuMF fuses both for **enhanced recommendations**

---

## **Final Notes**

* Trained models can be saved using `torch.save(model.state_dict(), "neumf.pth")`
* Evaluation metrics and plots can be included in a portfolio report
* The project is **modular** and ready to extend with:

  * Attention layers
  * Context features (genre, time, device)
  * Hybrid recommenders (BERT embeddings)

---

