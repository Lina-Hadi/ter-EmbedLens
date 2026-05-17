"""
web_app.py — Streamlit Embeddings Explorer

Modifications par rapport à la version initiale (Feature 1) :
- Ajout de "ResNet50 Fine-tuné + UMAP" dans load_all_combinations()
- Ajout des métriques correspondantes dans METRICS
- Tab 1 : sélecteur de combinaison (plus seulement CLIP + UMAP)
- Tab 3 : la nouvelle combinaison apparaît automatiquement dans la comparaison
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from PIL import Image
from torch.utils.data import Subset
from collections import Counter

from src.dataset.cifar_loader import CIFARLoader
from src.knn.knn_search import KNNFinder
from src.embedding.extractor import EmbeddingExtractor


# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Embeddings Explorer", layout="wide")
st.title("Visualisation des embeddings d'images")

UNKNOWN_THRESHOLD = 0.35

# Combinaison affichée par défaut dans Tab 1
DEFAULT_COMBO = "CLIP + UMAP"


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

# Toutes les combinaisons disponibles — ajoutez une entrée ici pour en ajouter une nouvelle
ALL_COMBO_FILES = {
    "CLIP + UMAP": (
        "data/processed/embeddings.npy",
        "data/processed/coords_2d.npy",
        "data/processed/labels.npy",
    ),
    "CLIP + t-SNE": (
        "data/processed/embeddings_tsne.npy",
        "data/processed/coords_2d_tsne.npy",
        "data/processed/labels_tsne.npy",
    ),
    "ResNet50 + UMAP": (
        "data/processed/embeddings_resnet_umap.npy",
        "data/processed/coords_2d_resnet_umap.npy",
        "data/processed/labels_resnet_umap.npy",
    ),
    "ResNet50 + t-SNE": (
        "data/processed/embeddings_resnet_tsne.npy",
        "data/processed/coords_2d_resnet_tsne.npy",
        "data/processed/labels_resnet_tsne.npy",
    ),
    # ← Feature 1 : ResNet50 fine-tuné
    "ResNet50 Fine-tuné + UMAP": (
        "data/processed/embeddings_resnet50_ft_umap.npy",
        "data/processed/coords_2d_resnet50_ft_umap.npy",
        "data/processed/labels_resnet50_ft_umap.npy",
    ),
}

# Métriques pré-calculées (à mettre à jour après chaque pipeline)
METRICS = {
    "CLIP + UMAP":              {"Silhouette": 0.3528, "Davies-Bouldin": 1.3132, "Trustworthiness": 0.9637, "k-NN Accuracy": 0.8650},
    "CLIP + t-SNE":             {"Silhouette": 0.3304, "Davies-Bouldin": 1.3253, "Trustworthiness": 0.9785, "k-NN Accuracy": 0.8650},
    "ResNet50 + UMAP":          {"Silhouette": 0.1074, "Davies-Bouldin": 3.4907, "Trustworthiness": 0.9337, "k-NN Accuracy": 0.7630},
    "ResNet50 + t-SNE":         {"Silhouette": 0.0690, "Davies-Bouldin": 3.0964, "Trustworthiness": 0.9593, "k-NN Accuracy": 0.7630},
    # ← Remplissez ces valeurs après avoir exécuté resnet50_ft_umap.py
    "ResNet50 Fine-tuné + UMAP": {"Silhouette": None,   "Davies-Bouldin": None,   "Trustworthiness": None,   "k-NN Accuracy": None},
}


@st.cache_data
def load_all_combinations():
    """Charge toutes les combinaisons dont les fichiers .npy existent."""
    combos = {}
    for name, (emb_path, coord_path, lbl_path) in ALL_COMBO_FILES.items():
        try:
            combos[name] = {
                "embeddings": np.load(emb_path),
                "coords":     np.load(coord_path),
                "labels":     np.load(lbl_path),
            }
        except FileNotFoundError:
            pass  # combo ignorée si les fichiers n'ont pas encore été générés
    return combos


@st.cache_data
def load_cifar_subset(n: int):
    """Charge les n premières images CIFAR-10 (pour affichage dans Tab 1)."""
    loader  = CIFARLoader(root_dir="./data/raw", train=True)
    dataset = loader.load()
    subset  = Subset(dataset, range(n))
    return subset, dataset.classes


@st.cache_resource
def load_extractor():
    return EmbeddingExtractor()


# =============================================================================
# INITIALISATION
# =============================================================================
all_combos = load_all_combinations()
extractor  = load_extractor()

if not all_combos:
    st.error(
        "Aucun fichier .npy trouvé dans data/processed/. "
        "Exécutez d'abord un script de pipeline (ex: clip_umap.py)."
    )
    st.stop()

available_combos = list(all_combos.keys())

# Charger le subset CIFAR correspondant à la combinaison par défaut
default_data  = all_combos.get(DEFAULT_COMBO, list(all_combos.values())[0])
n_default     = len(default_data["embeddings"])
subset, class_names = load_cifar_subset(n_default)


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "Explorer les embeddings",
    "Classifier une image",
    "Comparer les modèles",
])


# =============================================================================
# TAB 1 — VISUALISATION + KNN
# =============================================================================
with tab1:

    # Sélecteur de combinaison (nouveauté Feature 1)
    combo_choice = st.selectbox(
        "Combinaison modèle + réduction",
        available_combos,
        index=available_combos.index(DEFAULT_COMBO) if DEFAULT_COMBO in available_combos else 0,
        key="tab1_combo",
    )
    combo_data = all_combos[combo_choice]
    embeddings = combo_data["embeddings"]
    coords     = combo_data["coords"]
    labels     = combo_data["labels"]

    df = pd.DataFrame({
        "x":           coords[:, 0],
        "y":           coords[:, 1],
        "label":       labels,
        "dataset_idx": range(len(labels)),  # index stable même après filtrage
    })
    df["label_name"] = df["label"].apply(lambda x: class_names[x])

    # KNN sur la combinaison choisie
    knn = KNNFinder(metric="cosine")
    knn.fit(embeddings)

    selected_class = st.selectbox("Filtrer par classe", ["All"] + list(class_names))
    df_plot = df[df["label_name"] == selected_class] if selected_class != "All" else df

    fig = px.scatter(
        df_plot, x="x", y="y", color="label_name",
        title=f"Projection 2D — {combo_choice}",
        width=900, height=600,
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sélection d'une image")
    selected_idx = st.slider("Choisir une image (id)", 0, len(subset) - 1, 0)
    img, lbl = subset[selected_idx]

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(img.permute(1, 2, 0).numpy(), caption=f"{class_names[lbl]}")
    with col2:
        st.write(f"**ID :** {selected_idx}")
        st.write(f"**Classe :** {class_names[lbl]}")

    st.subheader("Plus proches voisins")
    k = st.slider("Nombre de voisins", 1, 10, 5)
    neighbors, distances = knn.query(selected_idx, k=k)

    cols = st.columns(k)
    for i, (nid, dist) in enumerate(zip(neighbors, distances)):
        img_n, lbl_n = subset[nid]
        with cols[i]:
            st.image(
                img_n.permute(1, 2, 0).numpy(),
                caption=f"id={nid}\n{class_names[lbl_n]}\n{dist:.3f}",
            )


# =============================================================================
# TAB 2 — CLASSIFICATION D'UNE IMAGE IMPORTÉE
# =============================================================================
with tab2:

    st.subheader("Importer une image et prédire sa classe")
    st.write(
        "L'image sera encodée par CLIP et comparée aux embeddings du dataset "
        "via k-NN. Si elle est trop éloignée de toutes les classes connues, "
        "elle sera signalée comme **hors distribution**."
    )

    # Tab 2 utilise toujours CLIP + UMAP comme référence de classification
    ref_combo   = all_combos.get("CLIP + UMAP", list(all_combos.values())[0])
    ref_emb     = ref_combo["embeddings"]
    ref_labels  = ref_combo["labels"]

    knn_tab2 = KNNFinder(metric="cosine")
    knn_tab2.fit(ref_emb)

    n_ref           = len(ref_emb)
    subset_ref, _   = load_cifar_subset(n_ref)

    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    k_classify    = st.slider("Nombre de voisins pour le vote (k)", 1, 15, 5, key="k_classify")

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")

        col_img, col_result = st.columns([1, 2])
        with col_img:
            st.image(pil_image, caption="Image importée", use_container_width=True)

        with st.spinner("Extraction de l'embedding CLIP..."):
            clip_model = extractor.model
            preprocess = extractor.preprocess
            device     = extractor.device

            img_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                query_emb    = clip_model.encode_image(img_tensor)
                query_emb    = query_emb / query_emb.norm(dim=-1, keepdim=True)
                query_emb_np = query_emb.cpu().float().numpy()

        distances_raw, indices_raw = knn_tab2.nn.kneighbors(
            query_emb_np, n_neighbors=k_classify, return_distance=True
        )
        neighbor_indices   = indices_raw[0].tolist()
        neighbor_distances = distances_raw[0].tolist()
        neighbor_labels    = [ref_labels[nid] for nid in neighbor_indices]

        vote_counts        = Counter(neighbor_labels)
        predicted_label_id = vote_counts.most_common(1)[0][0]
        predicted_class    = class_names[predicted_label_id]
        vote_score         = vote_counts.most_common(1)[0][1] / k_classify
        mean_distance      = np.mean(neighbor_distances)

        is_unknown = mean_distance > UNKNOWN_THRESHOLD and vote_score < 0.5

        with col_result:
            if is_unknown:
                st.error(
                    f"⚠️ **Classe inconnue / Hors distribution**\n\n"
                    f"Distance cosine moyenne : **{mean_distance:.4f}** (seuil : {UNKNOWN_THRESHOLD})\n\n"
                    f"Cette image ne ressemble à aucune des 10 classes CIFAR-10 avec suffisamment de confiance."
                )
            else:
                st.success(
                    f"✅ **Classe prédite : {predicted_class.upper()}**\n\n"
                    f"Confiance (vote) : **{vote_score * 100:.0f}%** "
                    f"({vote_counts.most_common(1)[0][1]}/{k_classify} voisins)\n\n"
                    f"Distance cosine moyenne : **{mean_distance:.4f}**"
                )

            st.write("**Détail du vote k-NN :**")
            vote_df = pd.DataFrame([
                {"Classe": class_names[lid], "Votes": cnt}
                for lid, cnt in vote_counts.most_common()
            ])
            st.dataframe(vote_df, hide_index=True)

        st.subheader(f"Les {k_classify} plus proches voisins dans le dataset")
        neighbor_cols = st.columns(min(k_classify, 5))
        for i, (nid, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
            img_n, lbl_n = subset_ref[nid]
            with neighbor_cols[i % 5]:
                st.image(
                    img_n.permute(1, 2, 0).numpy(),
                    caption=f"id={nid}\n{class_names[lbl_n]}\n{dist:.3f}",
                )


# =============================================================================
# TAB 3 — COMPARAISON CÔTE À CÔTE
# =============================================================================
with tab3:

    st.subheader("Comparaison des modèles d'embeddings")

    if len(available_combos) < 2:
        st.warning("Générez au moins deux combinaisons avant de comparer.")
    else:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            choice_left = st.selectbox("Combinaison gauche", available_combos, index=0, key="left")
        with col_sel2:
            choice_right = st.selectbox(
                "Combinaison droite", available_combos,
                index=min(2, len(available_combos) - 1), key="right",
            )

        filter_class = st.selectbox(
            "Filtrer par classe (les deux plots)", ["All"] + list(class_names), key="compare_filter"
        )

        def make_df(combo_name):
            data = all_combos[combo_name]
            df_c = pd.DataFrame({
                "x": data["coords"][:, 0],
                "y": data["coords"][:, 1],
                "label": data["labels"],
            })
            df_c["label_name"] = df_c["label"].apply(lambda x: class_names[x])
            if filter_class != "All":
                df_c = df_c[df_c["label_name"] == filter_class]
            return df_c

        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            df_left = make_df(choice_left)
            st.plotly_chart(
                px.scatter(df_left, x="x", y="y", color="label_name",
                           title=choice_left, height=500,
                           color_discrete_sequence=px.colors.qualitative.D3),
                use_container_width=True,
            )
        with col_plot2:
            df_right = make_df(choice_right)
            st.plotly_chart(
                px.scatter(df_right, x="x", y="y", color="label_name",
                           title=choice_right, height=500,
                           color_discrete_sequence=px.colors.qualitative.D3),
                use_container_width=True,
            )

        # Tableau comparatif des métriques
        st.subheader("Comparaison des métriques")
        m_left  = METRICS.get(choice_left,  {})
        m_right = METRICS.get(choice_right, {})

        metric_names = ["Silhouette", "Davies-Bouldin", "Trustworthiness", "k-NN Accuracy"]
        better_high  = {"Silhouette", "Trustworthiness", "k-NN Accuracy"}

        rows = []
        for m in metric_names:
            vl = m_left.get(m)
            vr = m_right.get(m)

            if isinstance(vl, float) and isinstance(vr, float):
                if m in better_high:
                    winner = choice_left if vl > vr else (choice_right if vr > vl else "Égalité")
                else:
                    winner = choice_left if vl < vr else (choice_right if vr < vl else "Égalité")
            else:
                winner = "—"

            rows.append({
                "Métrique":   m,
                choice_left:  f"{vl:.4f}" if isinstance(vl, float) else "—",
                choice_right: f"{vr:.4f}" if isinstance(vr, float) else "—",
                "Meilleur":   winner,
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Verdict global
        st.subheader("Verdict")
        scores = {choice_left: 0, choice_right: 0}
        for row in rows:
            if row["Meilleur"] in scores:
                scores[row["Meilleur"]] += 1

        if scores[choice_left] == scores[choice_right]:
            st.info("Les deux combinaisons sont équivalentes sur ces métriques.")
        else:
            best = max(scores, key=scores.get)
            st.success(
                f"**{best}** remporte **{scores[best]}/4 métriques** "
                f"et constitue la meilleure combinaison dans cette comparaison."
            )
