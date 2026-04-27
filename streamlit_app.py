import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import sys
import os
sys.path.append(os.path.join("..", "src"))

from utils import (
    _format_weight,
    _matrix_to_weighted_graph,
    generalized_sunlet_matrix,
    sunlet_layout,
)


DEFAULT_LAMBDAS = "1, 2, 3"
DEFAULT_MUS = "1.5, 2.5, 3.5"
DEFAULT_P = "3, 2, 1, 3, 1, 1"
DEFAULT_K = 6
DEFAULT_DECIMALS = 6
DEFAULT_TOLERANCE = 1e-6


def parse_float_list(raw_value, field_name):
    cleaned = raw_value.strip()
    if not cleaned:
        raise ValueError(f"Informe valores para {field_name}.")

    cleaned = re.sub(r"[\[\]\(\)]", "", cleaned)
    tokens = [token for token in re.split(r"[\s,;]+", cleaned) if token]

    if not tokens:
        raise ValueError(f"Nao foi possivel ler a lista {field_name}.")

    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(
            f"A lista {field_name} deve conter apenas numeros separados por virgula, espaco ou ponto e virgula."
        ) from exc


def parse_int_list(raw_value, field_name):
    float_values = parse_float_list(raw_value, field_name)
    int_values = []

    for value in float_values:
        if not float(value).is_integer():
            raise ValueError(f"A lista {field_name} deve conter apenas inteiros.")
        int_values.append(int(value))

    return int_values


def validate_inputs(k, p, lambdas, mus):
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if len(p) != k:
        raise ValueError(f"The list p must satisfy |p|=k. Got len(p)={len(p)} and k={k}.")

    if any(value <= 0 for value in p):
        raise ValueError("All values in p must be positive integers.")

    if len(lambdas) < 2:
        raise ValueError("The list lambdas must have at least 2 values.")

    if len(mus) != len(lambdas):
        raise ValueError(
            "The current construction requires len(mus) = len(lambdas), because utils.generalized_sunlet_matrix uses mu[:-1] and lambdas[1:]."
        )


def cluster_distinct_values(values, tolerance):
    values = np.sort(np.asarray(values, dtype=float))
    groups = []

    for value in values:
        if not groups or abs(value - groups[-1]["center"]) > tolerance:
            groups.append({"center": float(value), "count": 1})
            continue

        group = groups[-1]
        updated_count = group["count"] + 1
        group["center"] = (group["center"] * group["count"] + float(value)) / updated_count
        group["count"] = updated_count

    return groups


def build_spectrum_tables(matrix, decimals, tolerance):
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    grouped = cluster_distinct_values(eigenvalues, tolerance)

    full_spectrum = [
        {"Index": index + 1, "Eigenvalue": round(float(value), decimals)}
        for index, value in enumerate(eigenvalues)
    ]
    distinct_spectrum = [
        {
            "DSpec(A)": round(group["center"], decimals),
            "Multiplicity": group["count"],
        }
        for group in grouped[::-1]
    ]

    return eigenvalues, full_spectrum, distinct_spectrum


def build_weighted_graph_figure(matrix, weight_precision=2):
    matrix, graph = _matrix_to_weighted_graph(matrix, threshold=1e-6)
    pos = sunlet_layout(
        graph,
        cycle_radius=8.0,
        arm_step=1.0,
        arm_aperture=0.9,
    )

    node_labels = {
        node: _format_weight(matrix[node, node], weight_precision)
        for node in graph.nodes
    }
    edge_labels = {
        (u, v): _format_weight(data["weight"], weight_precision)
        for u, v, data in graph.edges(data=True)
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=2.2, edge_color="dimgray")
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="#B9D8F3",
        node_size=1300,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels=node_labels,
        font_size=9,
        font_weight="bold",
        font_color="black",
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        ax=ax,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    ax.set_title("Weighted graph representation of the generalized sunlet matrix", fontsize=14, pad=20)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()

    return fig


def render_results(k, p, lambdas, mus, decimals, tolerance):
    validate_inputs(k, p, lambdas, mus)

    matrix = generalized_sunlet_matrix(k=k, p=p, lambdas=lambdas, mu=mus)
    eigenvalues, full_spectrum, distinct_spectrum = build_spectrum_tables(
        matrix=matrix,
        decimals=decimals,
        tolerance=tolerance,
    )
    graph_figure = build_weighted_graph_figure(matrix)

    st.success("Construction completed successfully.")

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Matrix Order", f"{matrix.shape[0]} x {matrix.shape[1]}")
    metric_col_2.metric("Total Eigenvalues", len(eigenvalues))
    metric_col_3.metric("Distinct Eigenvalues", len(distinct_spectrum))

    st.subheader("Weighted Graph")
    st.pyplot(graph_figure, clear_figure=True)

    st.subheader("Matrix Spectrum")
    st.dataframe(full_spectrum, use_container_width=True, hide_index=True)

    st.subheader("Distinct Eigenvalues")
    st.dataframe(distinct_spectrum, use_container_width=True, hide_index=True)

    with st.expander("See Matrix Entries"):
        st.dataframe(np.round(matrix, decimals), use_container_width=True)


def main():
    st.set_page_config(
        page_title="Diminimal Unicyclic Graphs",
        layout="wide",
    )

    st.title("Diminimal Unicyclic Graphs")
    st.write(
        "Enter `lambdas`, `mus`, `k` and `p` to construct the generalized sunlet matrix, visualize the weighted graph and inspect its spectrum."
    )
    st.caption(
        "Accepted format for lists: `1, 2, 3` or `1 2 3`. The current validation requires `len(p) = k` and `len(mus) = len(lambdas)`."
    )

    with st.sidebar:
        st.header("Settings")
        decimals = st.number_input(
            "Precision for displaying eigenvalues",
            min_value=2,
            max_value=12,
            value=DEFAULT_DECIMALS,
            step=1,
        )
        tolerance = st.number_input(
            "Tolerance for clustering eigenvalues into distinct values",
            min_value=1e-10,
            max_value=1e-1,
            value=DEFAULT_TOLERANCE,
            format="%.8f",
        )

    with st.form("sunlet_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            lambdas_raw = st.text_area("lambdas", value=DEFAULT_LAMBDAS, height=120)
            mus_raw = st.text_area("mus", value=DEFAULT_MUS, height=120)

        with right_col:
            k = st.number_input("k", min_value=1, value=DEFAULT_K, step=1)
            p_raw = st.text_area("p", value=DEFAULT_P, height=120)

        submitted = st.form_submit_button("Generate Matrix, Graph and Spectrum", use_container_width=True)

    if not submitted:
        return

    try:
        lambdas = parse_float_list(lambdas_raw, "lambdas")
        mus = parse_float_list(mus_raw, "mus")
        p = parse_int_list(p_raw, "p")
        render_results(
            k=int(k),
            p=p,
            lambdas=lambdas,
            mus=mus,
            decimals=int(decimals),
            tolerance=float(tolerance),
        )
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
