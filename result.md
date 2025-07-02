# SNA Project Results Overview

This document provides a summary of the main results and features from the Streamlit dashboard for the SNA project.

---

## 1. Centrality Measures

For each temporal snapshot, the following centrality measures are computed for all nodes:
- **Degree Centrality**
- **Closeness Centrality**
- **Betweenness Centrality**

**Example Table:**

| Node | Degree Centrality | Closeness Centrality | Betweenness Centrality |
|------|-------------------|---------------------|-----------------------|
| 1    | 0.05              | 0.12                | 0.00                  |
| 2    | 0.10              | 0.15                | 0.01                  |
| ...  | ...               | ...                 | ...                   |

---

## 2. Link Prediction Demo

- Up to 100 real node pairs are sampled for each snapshot.
- For each pair, a link likelihood is predicted using node features (including centralities).
- The actual existence of the link is shown.

**Example Table:**

| Source | Target | Predicted Likelihood | Actual Link Exists |
|--------|--------|---------------------|-------------------|
| 1      | 2      | 0.8123              | Yes               |
| 3      | 4      | 0.1023              | No                |
| ...    | ...    | ...                 | ...               |

---

## 3. Evaluation Metrics

For the link prediction demo, the following metrics are computed (for a user-selected threshold):
- **Precision**
- **Recall**
- **F1-score**

**Example:**

- Precision: 0.67
- Recall: 0.45
- F1-score: 0.54
- Threshold: 0.5000

---

## 4. Community Detection

For each snapshot, communities are detected using:
- **Louvain Method** (if available)
- **Girvan-Newman Method**

**Example Output:**

**Louvain Communities:**
- Community 0: [1, 2, 3, 4]
- Community 1: [5, 6, 7]

**Girvan-Newman Communities (First Split):**
- Community 1: [1, 2, 3]
- Community 2: [4, 5, 6]

---

## 5. Usage Notes
- All results are interactive and can be explored for any snapshot in the Streamlit dashboard.
- Node selection and evaluation are based on real data from `CollegeMsg.txt`.

---

For more details, see the Streamlit app or the codebase. 