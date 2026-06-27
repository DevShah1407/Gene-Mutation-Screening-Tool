import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
 
st.set_page_config(page_title="Medulloblastoma Mutation Profile", layout="wide")
 
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 22,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
    'figure.titlesize': 26
})
 
st.title("Medulloblastoma Mutation Profile & Metadata")
 
uploaded_file = st.file_uploader("Upload the medulloblastoma CSV file", type=["csv"])
 
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.set_index("Patient_ID", inplace=True)
 
        metadata_cols = ['Age', 'Sex', 'Subtype', 'Survival_Status']
        mutation_df = df.drop(columns=metadata_cols)
 
        df["Mutation_Count"] = mutation_df.sum(axis=1)
        gene_mutation_freq = mutation_df.sum().sort_values(ascending=False)
 
        # ---------------------------------------------------------------
        # Overview pie charts
        # ---------------------------------------------------------------
        st.subheader("Cohort Overview")
 
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
 
        sex_counts = df["Sex"].value_counts()
        axes[0, 0].pie(
            sex_counts,
            labels=sex_counts.index,
            colors=['#7DB8F3', '#F7A8C4'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 16, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        axes[0, 0].set_title("Sex Distribution", fontsize=24, fontweight="bold")
 
        surv_counts = df["Survival_Status"].value_counts()
        surv_map = {'Alive': '#98F08B', 'Deceased': '#333333'}
        axes[0, 1].pie(
            surv_counts,
            labels=surv_counts.index,
            colors=[surv_map[i] for i in surv_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 16, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        axes[0, 1].set_title("Survival Status", fontsize=24, fontweight="bold")
 
        top_genes = gene_mutation_freq[gene_mutation_freq > 0]
        if len(top_genes) > 8:
            top_genes = top_genes.head(8)
 
        axes[1, 0].pie(
            top_genes.values,
            labels=top_genes.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 14, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        axes[1, 0].set_title("Most Frequently Mutated Genes", fontsize=24, fontweight="bold")
 
        subtype_counts = df["Subtype"].value_counts()
        subtype_colors = ['#7DB8F3', '#F7A8C4', '#98F08B', '#FFD34D']
        axes[1, 1].pie(
            subtype_counts,
            labels=subtype_counts.index,
            colors=subtype_colors[:len(subtype_counts)],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 16, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        axes[1, 1].set_title("Molecular Subtype Distribution", fontsize=24, fontweight="bold")
 
        plt.subplots_adjust(hspace=0.35, wspace=0.30)
        st.pyplot(fig)
        plt.close(fig)
 
        # ---------------------------------------------------------------
        # Clustermap of mutation profile + metadata annotations
        # ---------------------------------------------------------------
        st.subheader("Mutation Profile Heatmap")
 
        mutations = mutation_df.T
 
        subtype_pal = sns.color_palette("Set1", len(df["Subtype"].unique()))
        subtype_lut = dict(zip(df["Subtype"].unique(), subtype_pal))
        subtype_colors = df["Subtype"].map(subtype_lut)
 
        sex_lut = {'M': '#87CEFA', 'F': '#FFB6C1'}
        sex_colors = df["Sex"].map(sex_lut)
 
        survival_lut = {'Alive': '#90EE90', 'Deceased': '#333333'}
        survival_colors = df["Survival_Status"].map(survival_lut)
 
        age_norm = mcolors.Normalize(vmin=df["Age"].min(), vmax=df["Age"].max())
        age_cmap = plt.get_cmap("viridis")
        age_colors = df["Age"].apply(lambda x: mcolors.to_hex(age_cmap(age_norm(x))))
 
        col_colors = pd.DataFrame({
            "Subtype": subtype_colors,
            "Age": age_colors,
            "Sex": sex_colors,
            "Survival": survival_colors
        })
 
        g = sns.clustermap(
            mutations,
            col_colors=col_colors,
            cmap="plasma",
            yticklabels=True,
            xticklabels=False,
            cbar_kws={'label': 'Mutation Status (1=Mutated, 0=Wildtype)', 'ticks': [0, 1]},
            figsize=(14, 10),
            linewidths=0.5
        )
 
        subtype_handles = [Patch(facecolor=subtype_lut[n]) for n in subtype_lut]
        sex_handles = [Patch(facecolor=sex_lut[n]) for n in sex_lut]
        survival_handles = [Patch(facecolor=survival_lut[n]) for n in survival_lut]
 
        l1 = g.fig.legend(subtype_handles, subtype_lut.keys(), title="Subtype",
                           bbox_to_anchor=(1.1, 1.0), bbox_transform=g.fig.transFigure, loc="upper left")
        l2 = g.fig.legend(sex_handles, sex_lut.keys(), title="Sex",
                           bbox_to_anchor=(1.1, 0.85), bbox_transform=g.fig.transFigure, loc="upper left")
        g.fig.add_artist(l1)
 
        g.fig.legend(survival_handles, survival_lut.keys(), title="Survival Status",
                     bbox_to_anchor=(1.1, 0.70), bbox_transform=g.fig.transFigure, loc="upper left")
        g.fig.add_artist(l2)
 
        g.fig.suptitle("Medulloblastoma Mutation Profile & Metadata", fontsize=16, y=1.02)
 
        st.pyplot(g.fig)
        plt.close(g.fig)
 
        df.drop(columns="Mutation_Count", inplace=True)
 
        with st.expander("View raw data"):
            st.dataframe(df)
 
    except KeyError as e:
        st.error(f"Missing required column: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
else:
    st.info("Upload a CSV file to generate the analysis.")
