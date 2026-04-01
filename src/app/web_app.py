import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]  
sys.path.append(str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from torch.utils.data import Subset

# imports de TON projet
from src.dataset.cifar_loader import CIFARLoader
from src.knn.knn_search import KNNFinder


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Embeddings Explorer", layout="wide")
st.title("Visualisation des embeddings d'images (CLIP + UMAP)")


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    embeddings = np.load("data/processed/embeddings.npy")
    coords = np.load("data/processed/coords_2d.npy")
    labels = np.load("data/processed/labels.npy")

    loader = CIFARLoader(root_dir="./data/raw", train=True)
    dataset = loader.load()
    subset = Subset(dataset, range(len(embeddings)))

    return embeddings, coords, labels, subset, dataset.classes


embeddings, coords, labels, subset, class_names = load_data()


# =========================
# DATAFRAME FOR PLOT
# =========================
df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "label": labels
})
df["label_name"] = df["label"].apply(lambda x: class_names[x])


# =========================
# KNN MODEL
# =========================
knn = KNNFinder(metric="cosine")
knn.fit(embeddings)


# =========================
# UI - FILTER
# =========================
selected_class = st.selectbox("Filtrer par classe", ["All"] + class_names)

if selected_class != "All":
    df_plot = df[df["label_name"] == selected_class]
else:
    df_plot = df


# =========================
# SCATTER PLOT
# =========================
fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="label_name",
    title="Projection UMAP des embeddings CLIP",
    width=900,
    height=600
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# SELECT IMAGE
# =========================
st.subheader("Sélection d'une image")

selected_idx = st.slider("Choisir une image (id)", 0, len(subset) - 1, 0)

img, lbl = subset[selected_idx]

col1, col2 = st.columns([1, 3])

with col1:
    st.image(img.permute(1, 2, 0).numpy(), caption=f"Image sélectionnée\n{class_names[lbl]}")

with col2:
    st.write(f"**ID :** {selected_idx}")
    st.write(f"**Classe :** {class_names[lbl]}")


# =========================
# KNN
# =========================
st.subheader("Plus proches voisins")

k = st.slider("Nombre de voisins", 1, 10, 5)

neighbors, distances = knn.query(selected_idx, k=k)

cols = st.columns(k)

for i, (nid, dist) in enumerate(zip(neighbors, distances)):
    img_n, lbl_n = subset[nid]

    with cols[i]:
        st.image(
            img_n.permute(1, 2, 0).numpy(),
            caption=f"id={nid}\n{class_names[lbl_n]}\n{dist:.3f}"
        )