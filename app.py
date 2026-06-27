import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import streamlit as st
from io import StringIO


st.set_page_config(
    page_title="Medulloblastoma Mutation Heatmap",
    layout="wide",
)

st.title("Medulloblastoma Mutation Profile & Metadata")
st.write(
    "Upload a CSV file containing `Patient_ID`, `Age`, `Sex`, `Subtype`, `Survival_Status`, and mutation columns. "
    "The app will generate a clustered heatmap with metadata annotations."
)

uploaded_file = st.file_uploader("Upload your medulloblastoma CSV file", type=["csv"])


def load_data(file) -> pd.DataFrame:
    """Load the uploaded CSV into a DataFrame and validate required columns."""
    df = pd.read_csv(file)
    required_cols = ["Patient_ID", "Age", "Sex", "Subtype", "Survival_Status"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    if df["Patient_ID"].duplicated().any():
        raise ValueError("Patient_ID values must be unique.")

    df = df.copy()
    df.set_index("Patient_ID", inplace=True)
    return df


def make_clustermap(df: pd.DataFrame):
    """Create the clustered heatmap and return the matplotlib figure."""
    metadata_cols = ["Age", "Sex", "Subtype", "Survival_Status"]
    mutation_cols = [c for c in df.columns if c not in metadata_cols]

    if not mutation_cols:
        raise ValueError("No mutation columns were found. At least one gene/mutation column is required.")

    mutations = df[mutation_cols].T

    # Ensure numeric values for clustering/heatmap
    mutations = mutations.apply(pd.to_numeric, errors="coerce")

    if mutations.isna().all().all():
        raise ValueError("Mutation columns must contain numeric values such as 0/1.")

    # Subtype (categorical)
    subtype_order = list(pd.Series(df["Subtype"]).dropna().unique())
    subtype_pal = sns.color_palette("Set1", len(subtype_order))
    subtype_lut = dict(zip(subtype_order, subtype_pal))
    subtype_colors = df["Subtype"].map(subtype_lut)

    # Sex (categorical)
    sex_lut = {"M": "#87CEFA", "F": "#FFB6C1"}  # light blue / pink
    sex_colors = df["Sex"].map(sex_lut).fillna("#D3D3D3")

    # Survival status (categorical)
    survival_lut = {"Alive": "#90EE90", "Deceased": "#333333"}
    survival_colors = df["Survival_Status"].map(survival_lut).fillna("#D3D3D3")

    # Age (continuous)
    age_series = pd.to_numeric(df["Age"], errors="coerce")
    if age_series.isna().all():
        raise ValueError("Age column must contain numeric values.")
    age_norm = mcolors.Normalize(vmin=age_series.min(), vmax=age_series.max())
    age_cmap = plt.get_cmap("viridis")
    age_colors = age_series.apply(lambda x: mcolors.to_hex(age_cmap(age_norm(x))) if pd.notna(x) else "#D3D3D3")

    col_colors = pd.DataFrame(
        {
            "Subtype": subtype_colors,
            "Age": age_colors,
            "Sex": sex_colors,
            "Survival": survival_colors,
        },
        index=df.index,
    )

    sns.set_theme(style="white")

    g = sns.clustermap(
        mutations,
        col_colors=col_colors,
        cmap="plasma",
        yticklabels=True,
        xticklabels=False,
        cbar_kws={"label": "Mutation Status (1=Mutated, 0=Wildtype)", "ticks": [0, 1]},
        figsize=(14, 10),
        linewidths=0.5,
    )

    # Legends
    subtype_handles = [Patch(facecolor=subtype_lut[name], label=name) for name in subtype_lut]
    sex_handles = [Patch(facecolor=sex_lut[name], label=name) for name in sex_lut]
    survival_handles = [Patch(facecolor=survival_lut[name], label=name) for name in survival_lut]
    age_handle = Patch(facecolor="#808080", label="Age (viridis scale)")

    g.fig.legend(
        handles=subtype_handles,
        title="Subtype",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )
    g.fig.legend(
        handles=sex_handles,
        title="Sex",
        bbox_to_anchor=(1.02, 0.82),
        loc="upper left",
        borderaxespad=0.0,
    )
    g.fig.legend(
        handles=survival_handles,
        title="Survival Status",
        bbox_to_anchor=(1.02, 0.68),
        loc="upper left",
        borderaxespad=0.0,
    )
    g.fig.legend(
        handles=[age_handle],
        title="Age",
        bbox_to_anchor=(1.02, 0.54),
        loc="upper left",
        borderaxespad=0.0,
    )

    g.fig.suptitle("Medulloblastoma Mutation Profile & Metadata", y=1.02, fontsize=16)
    g.fig.tight_layout()
    return g.fig


if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)

        st.success("CSV loaded successfully.")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head())

        fig = make_clustermap(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except FileNotFoundError:
        st.error("The file could not be found. Please upload the CSV again.")
    except KeyError as e:
        st.error(f"A required column is missing: {e}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Upload a CSV to generate the heatmap.")
