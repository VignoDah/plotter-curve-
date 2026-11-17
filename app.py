import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

st.set_page_config(page_title="PyCurve 2D/3D", layout="wide")

# Configurazione parser matematico (supporta '2t', 't^2', sin, cos, exp, pi, e...)
t_sym = sp.Symbol("t")
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
LOCAL_DICT = {"t": t_sym, "pi": sp.pi, "e": sp.E}


def valuta_espressione(expr_str: str, t_vals: np.ndarray) -> np.ndarray:
    """Converte una stringa matematica in un array NumPy valutato su t_vals."""
    expr = parse_expr(expr_str, local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)
    func = sp.lambdify(t_sym, expr, modules=["numpy"])
    risultato = func(t_vals)
    # Se l'espressione è una costante (es. z = 0), espande allo stesso shape di t_vals
    if np.ndim(risultato) == 0:
        risultato = np.full_like(t_vals, float(risultato), dtype=float)
    return np.asarray(risultato, dtype=float)


# Inizializzazione stato delle curve (stile Vista Algebra di GeoGebra)
if "curve" not in st.session_state:
    st.session_state.curve = [
        {
            "id": 1,
            "visibile": True,
            "colore": "#1f77b4",
            "x": "cos(t)",
            "y": "sin(t)",
            "z": "t / 5",
            "t_min": 0.0,
            "t_max": 12.56,
        },
        {
            "id": 2,
            "visibile": True,
            "colore": "#e63946",
            "x": "cos(3t)",
            "y": "sin(2t)",
            "z": "sin(t)",
            "t_min": 0.0,
            "t_max": 6.28,
        },
    ]
    st.session_state.next_id = 3

# --- BARRA LATERALE: CONTROLLI E LISTA FUNZIONI ---
with st.sidebar:
    st.title("Plotter Curve")

    spazio = st.radio(
        "Spazio ambiente",
        options=["R² (Piano)", "R³ (Spazio)"],
        horizontal=True,
    )
    is_3d = spazio.startswith("R³")

    risoluzione = st.slider("Punti di campionamento", 100, 2000, 600, step=100)
    st.divider()

    if st.button("+ Aggiungi curva", use_container_width=True, type="primary"):
        st.session_state.curve.append(
            {
                "id": st.session_state.next_id,
                "visibile": True,
                "colore": "#2a9d8f",
                "x": "t",
                "y": "sin(t)",
                "z": "cos(t)",
                "t_min": -5.0,
                "t_max": 5.0,
            }
        )
        st.session_state.next_id += 1
        st.rerun()

    # Elenco schede delle curve
    da_eliminare = None
    for idx, c in enumerate(st.session_state.curve):
        with st.container(border=True):
            col_vis, col_tit, col_col, col_del = st.columns([1, 4, 2, 1])
            c["visibile"] = col_vis.checkbox(
                "Visibile", value=c["visibile"], key=f"vis_{c['id']}", label_visibility="collapsed"
            )
            col_tit.markdown(f"**Curva $\\gamma_{{{idx + 1}}}(t)$**")
            c["colore"] = col_col.color_picker(
                "Colore", value=c["colore"], key=f"col_{c['id']}", label_visibility="collapsed"
            )
            if col_del.button("✕", key=f"del_{c['id']}", help="Elimina curva"):
                da_eliminare = idx

            c["x"] = st.text_input("x(t) =", value=c["x"], key=f"x_{c['id']}")
            c["y"] = st.text_input("y(t) =", value=c["y"], key=f"y_{c['id']}")
            if is_3d:
                c["z"] = st.text_input("z(t) =", value=c["z"], key=f"z_{c['id']}")

            c1, c2 = st.columns(2)
            c["t_min"] = c1.number_input("t min", value=float(c["t_min"]), key=f"tmin_{c['id']}")
            c["t_max"] = c2.number_input("t max", value=float(c["t_max"]), key=f"tmax_{c['id']}")

    if da_eliminare is not None:
        st.session_state.curve.pop(da_eliminare)
        st.rerun()

# --- AREA PRINCIPALE: GRAFICO ---
fig = go.Figure()

for idx, c in enumerate(st.session_state.curve):
    if not c["visibile"]:
        continue
    if c["t_min"] >= c["t_max"]:
        st.sidebar.warning(f"Curva {idx + 1}: t_min deve essere minore di t_max.")
        continue

    t_vals = np.linspace(c["t_min"], c["t_max"], risoluzione)
    nome_curva = f"γ{idx + 1}(t)"

    try:
        x_vals = valuta_espressione(c["x"], t_vals)
        y_vals = valuta_espressione(c["y"], t_vals)

        if is_3d:
            z_vals = valuta_espressione(c["z"], t_vals)
            fig.add_trace(
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="lines",
                    name=nome_curva,
                    line=dict(color=c["colore"], width=5),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=nome_curva,
                    line=dict(color=c["colore"], width=3),
                )
            )
    except Exception as err:
        st.sidebar.error(f"Errore nella formula di {nome_curva}: {err}")

# Configurazione assi e proporzioni
if is_3d:
    fig.update_layout(
        height=750,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
else:
    fig.update_layout(
        height=750,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="X", zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
        yaxis=dict(
            title="Y",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#888",
            scaleanchor="x",
            scaleratio=1,
        ),
    )

st.plotly_chart(fig, use_container_width=True)
