# E-Commerce Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project implements an unsupervised machine learning model to segment an e-commerce company's customer base based on purchasing behavior. By analyzing **Recency, Frequency, and Monetary (RFM)** metrics, the system automatically groups customers into distinct clusters (e.g., High-Value Loyalists, At-Risk Customers) to enable data-driven marketing strategies.

---

## 📁 Repository Structure
```text
├── data/
│   └── online_retail.csv        # Dataset (or auto-generated synthetic data)
├── src/
│   ├── preprocess.py            # Data cleaning & RFM transformation
│   ├── model.py                 # K-Means clustering & Elbow Method
│   └── visualize.py             # Cluster scatter plot visualization
├── requirements.txt             # Dependency specification
├── run_segmentation.py          # Command-Line Executable Script
└── README.md                    # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/github-username/repo-name.git
cd repo-name
```

### Step 2: Set Up Virtual Environment
* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Execute (CLI Driven)

Run the main pipeline from your terminal:
```bash
python run_segmentation.py
```

### 📤 Generated Outputs:
1. `segmented_customers.csv`: Customer IDs mapped to their assigned cluster labels.
2. `customer_clusters.png`: Visual scatter plot illustrating customer segments.
3. `elbow_curve.png`: Plot confirming optimal K=4 clusters.
