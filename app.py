import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from application import EcommerceRecommendationSystem
from deploy import TemporalLinkPredictionService
from monitor import ModelMonitor
import networkx as nx
from data_utils import load_temporal_data, create_temporal_snapshots
from features import extract_node_features
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import random

st.set_page_config(page_title="Graph Transformer Results", layout="wide")
st.title("Graph Transformer Temporal Link Prediction Dashboard")

# --- Training Results ---
st.header("1. Training Results")
try:
    df = pd.read_csv("results.csv")
    st.dataframe(df, use_container_width=True)
    fig, ax = plt.subplots()
    ax.plot(df['epoch'], df['loss'], label='Loss')
    ax.plot(df['epoch'], df['val_auc'], label='Val AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Training Loss and Validation AUC')
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not load results.csv: {e}")

# --- Application Demo ---
st.header("2. Application Demo")
app = EcommerceRecommendationSystem(model=None, user_encoder=None, product_encoder=None)
user_id = st.number_input("User ID for Recommendations", min_value=1, value=1)
top_k = st.slider("Top K Recommendations", 1, 20, 5)
recs = app.generate_recommendations(user_id=user_id, timestamp=0, top_k=top_k)
st.subheader("Recommendations")
st.write(recs)
fraud = app.detect_fraud_patterns(user_interactions=[1,2,3,4,5,6,7,8,9,10,11])
st.subheader("Fraud Patterns")
st.write(fraud)

# --- Deployment Demo ---
st.header("3. Deployment Demo")
service = TemporalLinkPredictionService(model_path='best_model.pth', config={})
preds = service.predict_future_interactions(user_id=user_id, timestamp=0)
st.write("Predicted Future Interactions:", preds)

# --- Monitoring Demo ---
st.header("4. Monitoring Demo")
monitor = ModelMonitor()
monitor.log_prediction_metrics(predictions=[0.9,0.1,0.8,0.2], ground_truth=[1,0,1,0])
monitor.log_prediction_metrics(predictions=[0.7,0.3,0.6,0.4], ground_truth=[1,0,1,0])
st.subheader("Metrics History")
st.dataframe(pd.DataFrame(monitor.metrics_history))
if monitor.detect_performance_drift():
    st.error("Model performance drift detected! Consider retraining.")
else:
    st.success("No performance drift detected.")

# --- SNA Centrality Measures ---
st.header("5. Social Network Analysis (SNA) Centrality Measures")

# Load and process data only once (cache for performance)
@st.cache_data
def get_snapshots():
    df = load_temporal_data('CollegeMsg.txt')
    return create_temporal_snapshots(df, window_size='1D')

snapshots = get_snapshots()

# Let user select a snapshot by date
snapshot_dates = [str(snap['timestamp'].date()) for snap in snapshots]
selected_date = st.selectbox("Select Snapshot Date", snapshot_dates)
selected_snapshot = next(snap for snap in snapshots if str(snap['timestamp'].date()) == selected_date)

G = selected_snapshot['graph']
features_df = extract_node_features(G)

# Only show the requested centralities
centralities_df = features_df[[
    'degree_centrality',
    'closeness_centrality',
    'betweenness_centrality'
]].copy()
centralities_df.index.name = 'Node'

st.subheader(f"Centrality Measures for {selected_date}")
st.dataframe(centralities_df, use_container_width=True)

# --- Simple Link Prediction Demo using Node Features ---
st.header("6. Simple Temporal Link Prediction Demo (using Node Features)")

# Let user pick multiple nodes from the current snapshot
def get_node_options(G):
    nodes = list(G.nodes())
    return sorted(nodes)  # Ensure integer node IDs

# Use real node IDs from CollegeMsg.txt sample for defaults
real_node_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]

node_options = get_node_options(G)
# Filter real_node_ids to those present in the current snapshot
real_node_ids_present = [n for n in real_node_ids if n in node_options]

# --- Automatic random sampling of node pairs for evaluation ---
max_pairs = 100
all_possible_pairs = [(src, tgt) for src in real_node_ids_present for tgt in real_node_ids_present if src != tgt]
random.seed(42)
random_pairs = random.sample(all_possible_pairs, min(max_pairs, len(all_possible_pairs)))

# User can optionally select a few node pairs to highlight
highlight_src = st.multiselect("Highlight Source Nodes (optional)", node_options, default=real_node_ids_present[:2], max_selections=5, key="highlight_src")
highlight_tgt = st.multiselect("Highlight Target Nodes (optional)", node_options, default=real_node_ids_present[2:4], max_selections=5, key="highlight_tgt")
highlight_pairs = set((s, t) for s in highlight_src for t in highlight_tgt if s != t)

# Ensure feature DataFrame index is integer type
features_df.index = features_df.index.astype(int)

results = []
y_true = []
y_pred = []
threshold = st.slider("Prediction Threshold for Evaluation", 0.0, 1.0, 0.5, 0.0001, format="%.4f")
for node1, node2 in random_pairs:
    if node1 in features_df.index and node2 in features_df.index:
        features1 = features_df.loc[node1].values
        features2 = features_df.loc[node2].values
        link_score = float(np.dot(features1, features2))
        norm_score = 1 / (1 + np.exp(-link_score))  # Sigmoid
        exists = G.has_edge(node1, node2)
        results.append({
            'Source': node1,
            'Target': node2,
            'Predicted Likelihood': norm_score,
            'Actual Link Exists': 'Yes' if exists else 'No',
            'Highlight': 'Yes' if (node1, node2) in highlight_pairs else ''
        })
        y_true.append(1 if exists else 0)
        y_pred.append(1 if norm_score >= threshold else 0)

if results:
    df_results = pd.DataFrame(results)
    # Show highlighted pairs at the top
    df_results = pd.concat([
        df_results[df_results['Highlight'] == 'Yes'],
        df_results[df_results['Highlight'] != 'Yes']
    ], ignore_index=True)
    st.dataframe(df_results.drop(columns=['Highlight']))
    # Show evaluation metrics
    if len(set(y_true)) > 1:  # Avoid metrics if all true labels are the same
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        st.markdown(f"**Evaluation Metrics (Threshold = {threshold}):**")
        st.write(f"Precision: {precision:.3f}")
        st.write(f"Recall: {recall:.3f}")
        st.write(f"F1-score: {f1:.3f}")
    else:
        st.info("Not enough positive/negative samples for evaluation metrics.")
else:
    st.warning("No valid node pairs for evaluation.")

# --- Community Detection ---
st.header("7. Community Detection (Louvain & Girvan-Newman)")
import networkx as nx
try:
    import community as community_louvain
    louvain_available = True
except ImportError:
    louvain_available = False

# Louvain method (if available)
if louvain_available:
    partition = community_louvain.best_partition(G.to_undirected())
    louvain_communities = {}
    for node, comm in partition.items():
        louvain_communities.setdefault(comm, []).append(node)
    st.subheader("Louvain Communities")
    for comm, members in louvain_communities.items():
        st.write(f"Community {comm}: {sorted(members)}")
else:
    st.info("Louvain method not available. Install the 'python-louvain' package to enable it.")

# Girvan-Newman method (always available)
gn_generator = nx.community.girvan_newman(G.to_undirected())
try:
    top_level_communities = next(gn_generator)
    gn_communities = [sorted(list(c)) for c in top_level_communities]
    st.subheader("Girvan-Newman Communities (First Split)")
    for i, comm in enumerate(gn_communities):
        st.write(f"Community {i+1}: {comm}")
except Exception as e:
    st.warning(f"Girvan-Newman community detection failed: {e}") 