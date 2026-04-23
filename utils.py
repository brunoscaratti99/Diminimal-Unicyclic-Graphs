import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx


def plot_graph(N, edge_index, pos):
    graph_data = Data(x=torch.zeros(N), edge_index=edge_index)

    G = to_networkx(graph_data, to_undirected=True)

    plt.figure(figsize=(30, 30))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        width=3,
        font_weight="bold",
    )
    nx.draw_networkx_labels(G, pos, font_color="black")
    plt.show()


def _format_weight(value, digits=2):
    value = float(value)
    if abs(value) < 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _matrix_to_weighted_graph(A, threshold=1e-6):
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
    else:
        A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A matriz deve ser NxN. Recebido shape={A.shape}")

    N = A.shape[0]
    G = nx.Graph()

    for i in range(N):
        G.add_node(i, diagonal=float(A[i, i]))

    for i in range(N):
        for j in range(i + 1, N):
            if abs(A[i, j]) > threshold:
                G.add_edge(i, j, weight=float(A[i, j]))

    return A, G


def sunlet_layout(G, cycle_radius=3.5, arm_step=1.6, arm_aperture=0.8):
    """
    Posiciona os vertices do ciclo em uma circunferencia central e
    distribui os bracos radialmente para fora do ciclo.
    """
    if G.number_of_nodes() == 0:
        return {}

    if not nx.is_connected(G):
        raise ValueError("O layout sunlet requer um grafo conexo.")

    cycles = nx.cycle_basis(G)
    if len(cycles) != 1:
        raise ValueError("O layout sunlet requer um grafo com exatamente um ciclo.")

    cycle_nodes = list(cycles[0])
    start = min(range(len(cycle_nodes)), key=lambda i: cycle_nodes[i])
    cycle_nodes = cycle_nodes[start:] + cycle_nodes[:start]

    if len(cycle_nodes) > 2 and cycle_nodes[1] > cycle_nodes[-1]:
        cycle_nodes = [cycle_nodes[0]] + list(reversed(cycle_nodes[1:]))

    cycle_set = set(cycle_nodes)
    pos = {}
    cycle_angles = {}
    subtree_cache = {}

    def tree_children(node, parent):
        return sorted(
            neighbor
            for neighbor in G.neighbors(node)
            if neighbor != parent and neighbor not in cycle_set
        )

    def subtree_size(node, parent):
        key = (node, parent)
        if key in subtree_cache:
            return subtree_cache[key]

        children = tree_children(node, parent)
        if not children:
            subtree_cache[key] = 1
        else:
            subtree_cache[key] = sum(subtree_size(child, node) for child in children)

        return subtree_cache[key]

    def angular_sectors(nodes, parent, angle_min, angle_max):
        if not nodes:
            return []

        if len(nodes) == 1:
            return [(nodes[0], angle_min, angle_max)]

        span = angle_max - angle_min
        gap = min(np.deg2rad(3.0), 0.08 * span)
        weights = np.array([subtree_size(node, parent) for node in nodes], dtype=float)
        weights = weights / weights.sum()

        sectors = []
        cursor = angle_min
        for node, weight in zip(nodes, weights):
            width = span * weight
            child_min = cursor + gap / 2.0
            child_max = cursor + width - gap / 2.0

            if child_max < child_min:
                midpoint = cursor + width / 2.0
                child_min = midpoint
                child_max = midpoint

            sectors.append((node, child_min, child_max))
            cursor += width

        return sectors

    def place_subtree(node, parent, depth, angle_min, angle_max):
        theta = 0.5 * (angle_min + angle_max)
        radius = cycle_radius + depth * arm_step
        pos[node] = np.array(
            [radius * np.cos(theta), radius * np.sin(theta)],
            dtype=float,
        )

        children = tree_children(node, parent)
        for child, child_min, child_max in angular_sectors(children, node, angle_min, angle_max):
            place_subtree(child, node, depth + 1, child_min, child_max)

    angle_step = 2.0 * np.pi / len(cycle_nodes)
    half_aperture = min(np.pi / 4.0, arm_aperture * np.pi / len(cycle_nodes))

    for i, node in enumerate(cycle_nodes):
        theta = np.pi / 2.0 - i * angle_step
        cycle_angles[node] = theta
        pos[node] = np.array(
            [cycle_radius * np.cos(theta), cycle_radius * np.sin(theta)],
            dtype=float,
        )

    for node in cycle_nodes:
        theta = cycle_angles[node]
        roots = sorted(neighbor for neighbor in G.neighbors(node) if neighbor not in cycle_set)
        for child, child_min, child_max in angular_sectors(
            roots,
            node,
            theta - half_aperture,
            theta + half_aperture,
        ):
            place_subtree(child, node, 1, child_min, child_max)

    return {node: tuple(coords) for node, coords in pos.items()}


def plot_weighted_sunlet_graph_from_matrix(
    A,
    threshold=1e-6,
    cycle_radius=8.0,
    arm_step=1.0,
    arm_aperture=0.9,
    figsize=(26, 26),
    node_size=1500,
    node_color="lightblue",
    edge_color="dimgray",
    node_font_size=7,
    edge_font_size=6,
    weight_precision=2,
    title="Grafo sunlet ponderado",
):
    """
    Plota um grafo ponderado a partir de uma matriz de adjacencia.

    - Nos vertices aparece apenas o peso da diagonal A[i, i].
    - Nas arestas aparece o peso A[i, j].
    - O layout posiciona o ciclo no centro e os bracos para fora.
    """
    A, G = _matrix_to_weighted_graph(A, threshold=threshold)
    pos = sunlet_layout(
        G,
        cycle_radius=cycle_radius,
        arm_step=arm_step,
        arm_aperture=arm_aperture,
    )

    node_labels = {
        node: _format_weight(A[node, node], weight_precision)
        for node in G.nodes
    }
    edge_labels = {
        (u, v): _format_weight(data["weight"], weight_precision)
        for u, v, data in G.edges(data=True)
    }

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.2, edge_color=edge_color)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels=node_labels,
        font_size=node_font_size,
        font_weight="bold",
        font_color="black",
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        edge_labels=edge_labels,
        font_size=edge_font_size,
        rotate=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return pos, G


def weighted_sunlet_graph_to_tikz(
    A,
    threshold=1e-6,
    cycle_radius=8.0,
    arm_step=1.0,
    arm_aperture=0.9,
    coord_scale=0.6,
    weight_precision=2,
    node_options=r"circle, draw, fill=blue!20, minimum size=10mm, inner sep=1.5pt, font=\small",
    edge_options="draw=black!70, line width=0.9pt",
    edge_label_options=r"fill=white, inner sep=1pt, font=\scriptsize",
    output_path=None,
):
    """
    Gera o codigo TikZ de um grafo sunlet ponderado a partir da matriz A.

    - Cada vertice recebe como rotulo apenas o peso diagonal A[i, i].
    - Cada aresta recebe como rotulo o peso A[i, j].
    - O layout e o mesmo usado no plot sunlet do notebook.

    Se `output_path` for informado, o codigo e salvo em arquivo .tex/.tikz.
    """
    A, G = _matrix_to_weighted_graph(A, threshold=threshold)
    pos = sunlet_layout(
        G,
        cycle_radius=cycle_radius,
        arm_step=arm_step,
        arm_aperture=arm_aperture,
    )

    lines = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]

    for node in sorted(G.nodes):
        x, y = pos[node]
        x *= coord_scale
        y *= coord_scale
        label = _format_weight(A[node, node], weight_precision)
        lines.append(
            rf"  \node[{node_options}] (v{node}) at ({x:.4f}, {y:.4f}) {{{label}}};"
        )

    for u, v, data in sorted(G.edges(data=True)):
        weight = _format_weight(data["weight"], weight_precision)
        lines.append(
            rf"  \draw[{edge_options}] (v{u}) -- node[midway, {edge_label_options}] {{{weight}}} (v{v});"
        )

    lines.append(r"\end{tikzpicture}")
    tikz_code = chr(10).join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.write_text(tikz_code, encoding="utf-8")

    return tikz_code


def graph_matrix_index(A, threshold=1e-6, directed=False, include_self=False):
    # A: [N, N]
    N = A.shape[0]
    mask = A.abs() > threshold

    if not include_self:
        mask = mask & ~torch.eye(N, dtype=torch.bool, device=A.device)

    idx = mask.nonzero(as_tuple=False)  # [E, 2] com pares (i,j)

    if not directed:
        # mantem so triangulo superior e espelha -> evita duplicatas
        idx = idx[idx[:, 0] < idx[:, 1]]
        idx = torch.cat([idx, idx[:, [1, 0]]], dim=0)

    edge_index = idx.t().contiguous().long()  # [2, E]
    return edge_index


def plot_heatmap_nn(M, title="Mapa de calor", cmap="viridis", vmin=None, vmax=None, figsize=(10,10)):
    """
    Plota mapa de calor de uma matriz NxN.
    Aceita numpy array, lista de listas ou tensor do PyTorch.
    Retorna (fig, ax).
    """
    if hasattr(M, "detach"):  # torch.Tensor
        M = M.detach().cpu().numpy()
    else:
        M = np.asarray(M)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"A matriz deve ser NxN. Recebido shape={M.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Valor")

    ax.set_title(title)
    ax.set_xlabel("No j")
    ax.set_ylabel("No i")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))

    plt.tight_layout()
    plt.show()
    return fig, ax



def construir_matriz_jacobi(lambdas, mus):
    """
    Constrói a matriz de Jacobi J a partir dos autovalores lambdas (de J)
    e mus (da submatriz truncada J_tau).
    """
    # Ordenar autovalores conforme o artigo: lambda0 > mu1 > lambda1 > ... [8]
    lambdas = np.sort(lambdas)[::-1]
    mus = np.sort(mus)[::-1]
    
    n_total = len(lambdas)
    n_trunc = len(mus)
    
    if n_total != n_trunc + 1:
        raise ValueError("O número de lambdas deve ser exatamente um a mais que o de mus.")

    # 1. Calcular os pesos w_i usando a fórmula do artigo [3, 9]
    # w_i = A(lambda_i) / B'(lambda_i)
    weights = []
    for i in range(n_total):
        # Numerador: A(lambda_i) = produtório(lambda_i - mu_j)
        num = np.prod(lambdas[i] - mus)
        # Denominador: B'(lambda_i) = produtório(lambda_i - lambda_j) para j != i
        den = np.prod(lambdas[i] - np.delete(lambdas, i))
        weights.append(num / den)
    
    weights = np.array(weights)

    # 2. Procedimento de construção recursiva (Algoritmo de Lanczos/Stieltjes)
    # Este método implementa a lógica das equações (17), (18), (22) e (23) [6, 7]
    a = np.zeros(n_total)
    b = np.zeros(n_trunc)
    
    # Vetor inicial (raiz quadrada dos pesos para ortogonalização)
    q = np.sqrt(weights)
    Q = [q]

    for i in range(n_total):
        # Calcula termo da diagonal a_i [6, 7]
        a[i] = np.sum(lambdas * (Q[i]**2))
        
        if i < n_trunc:
            # Calcula resíduo para o próximo termo
            r = (lambdas - a[i]) * Q[i]
            if i > 0:
                r -= b[i-1] * Q[i-1]
            
            # O termo b_i é a norma do resíduo, garantindo b_i > 0 [5, 6]
            b[i] = np.linalg.norm(r)
            Q.append(r / b[i])

    # 3. Montagem da matriz J conforme a definição (1) [5]
    matriz_J = np.diag(a) + np.diag(b, k=1) + np.diag(b, k=-1)
    return matriz_J




def generalized_sunlet_matrix(k, p, lambdas, mu):
    L = len(lambdas)-1
    A_1 = construir_matriz_jacobi(lambdas, mu[:-1])
    A_2 = construir_matriz_jacobi(mu, lambdas[1:])

    prev_index = 0
    
    p = np.array(p)
    
    A_sunlet = np.zeros((p.sum() * L + k, p.sum() * L + k))
    
    
    for i in range(k):
        
        if i % 2 == 0:
            
            A_brace = np.zeros((L * p[i] + 1, L * p[i] + 1))
            A_brace[0,0] = A_1[0,0]
            
            A_brace[0, 1:] = [A_1[1,0] if (k - 1) % (L)==0 else 0 for k in range(1, L * p[i] + 1)] / np.sqrt(p[i])
            
            #aux_0 = [A_1[1,1] if (k-1)%(L)==0 else 0 for k in range(1,L*p[i]+1)] / np.sqrt(p[i])
            #print(A_brace[1:,0].shape)
            #print(aux_0.shape)
            
            
            A_brace[1:, 0] = [A_1[1,0] if (k + L - 1)%L==0 else 0 for k in range(1, L * p[i] + 1)] / np.sqrt(p[i])
            
            
            for j in range(p[i]):
                A_brace[1 + j * L:(j + 1) * L + 1, 1 + j * L:(j + 1) * L + 1] = A_1[1:,1:] 
        
        else:
            
            A_brace = np.zeros((L * p[i] + 1, L * p[i] + 1))
            A_brace[0,0] = A_2[0,0]
            
            A_brace[0, 1:] = [A_2[1,0] if (k - 1) % (L)==0 else 0 for k in range(1, L * p[i] + 1)] / np.sqrt(p[i])
            

            
            A_brace[1:, 0] = [A_2[1,0] if (k + L - 1)%L==0 else 0 for k in range(1, L * p[i] + 1)] / np.sqrt(p[i])
            
            
            for j in range(p[i]):
                A_brace[1 + j * L:(j + 1) * L + 1, 1 + j * L:(j + 1) * L + 1] = A_2[1:,1:] 
        
        
        A_sunlet[prev_index:prev_index + A_brace.shape[0], prev_index: prev_index + A_brace.shape[1]] = A_brace
        
        
        if i == 0:
            A_sunlet[prev_index, prev_index+A_brace.shape[0]] = -1
            A_sunlet[prev_index+A_brace.shape[0], prev_index] = -1
        elif i == k-1:
            A_sunlet[prev_index, 0] = 1
            A_sunlet[0, prev_index] = 1
        else:
            A_sunlet[prev_index, prev_index+A_brace.shape[0]] = 1
            A_sunlet[prev_index+A_brace.shape[0], prev_index] = 1
        
        
        prev_index += A_brace.shape[0]

    return A_sunlet