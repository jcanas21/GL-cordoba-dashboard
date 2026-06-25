"""Córdoba — Anchored Proximity Analysis (Streamlit).

Mirrors the Argentina dashboard's Page 3 layout and visual conventions
(palette, sliders, treemap colorscale, sankey styling), adapted for
Córdoba's manually-defined presence set:

  • OPEX-threshold slider = the single source of truth for the anchor set.
    Moving it re-derives anchors → links → candidates → all panels.
  • Attractiveness = w·PCI + w·market_size + w·market_growth  (NO COG).
  • Feasibility    = w·DAI_percentile + w·distance_percentile.
        distance_travelled = km promedio que viaja el HS4 a nivel global
        (ponderado por flujo bilateral 2020-2024). NO se invierte:
        mayor distancia recorrida → mayor factibilidad (productos
        globalmente comercializables; Argentina está lejos de los grandes
        mercados, así que los productos que ya viajan lejos son la
        oportunidad natural).
  • Strategic balance dial: Feasibility ↔ Attractiveness.
  • No anchor-density filter (omitted by design for Córdoba).
  • No "above-median proximity" toggle.

Run:    streamlit run app/cordoba_anchored_proximity_app.py
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hs4_names_es import HS4_NAMES_ES as SPANISH_OVERRIDES

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def hs4_name_es(hs4: str, fallback_en: str = "") -> str:
    """Spanish if curated, else fallback (typically Atlas English)."""
    return SPANISH_OVERRIDES.get(str(hs4).zfill(4), fallback_en or "")

# ---------------------------------------------------------------------------
# Visual constants — matches Argentina Page 3
# ---------------------------------------------------------------------------
SECTOR_COLORS = {
    "Services": "#b23c6f",
    "Textiles": "#7bc8a4",
    "Agriculture": "#e5c21a",
    "Stone": "#caa46b",
    "Minerals": "#a88b7d",
    "Metals": "#c9656b",
    "Chemicals": "#b07ac9",
    "Vehicles": "#7a6cc3",
    "Machinery": "#6e8fc3",
    "Electronics": "#74c5c6",
    "Other": "#2f5d74",
}

PCI_COLORSCALE = [
    (0.0, "rgb(227, 159, 96)"),
    (0.278697, "rgb(231, 173, 120)"),
    (0.338965, "rgb(235, 188, 143)"),
    (0.398272, "rgb(240, 202, 168)"),
    (0.448314, "rgb(244, 217, 191)"),
    (0.493999, "rgb(248, 231, 215)"),
    (0.494099, "rgb(192, 228, 225)"),
    (0.533691, "rgb(154, 211, 207)"),
    (0.571435, "rgb(116, 195, 189)"),
    (0.606597, "rgb(77, 178, 171)"),
    (0.661681, "rgb(40, 162, 153)"),
    (1.0, "rgb(2, 146, 135)"),
]

UMAP_TO_CLUSTER = {
    "Agriculture": "Agricultural Goods",
    "Construction, Building, and Home Supplies": "Construction Goods",
    "Electronic and Electrical Goods": "Electronics",
    "Industrial Chemicals and Metals": "Chemicals & Basic Metals",
    "Metalworking and Electrical Machinery and Parts": "Metalworking Machinery",
    "Minerals": "Minerals",
    "Textile Apparel and Accessories": "Apparel",
    "Textile and Home Goods": "Textile & Home Goods",
}

NATURAL_RESOURCE_HS4 = ["2711", "2710", "7108", "2709", "2713", "2701", "2603", "2616"]

HS_SECTIONS = [
    (1,  1,  5,  "1. Live animals; animal products"),
    (2,  6,  14, "2. Vegetable products"),
    (3,  15, 15, "3. Animal or vegetable fats and oils"),
    (4,  16, 24, "4. Prepared foods, beverages and tobacco"),
    (5,  25, 27, "5. Mineral products"),
    (6,  28, 38, "6. Products of the chemical or allied industries"),
    (7,  39, 40, "7. Plastics and rubber"),
    (8,  41, 43, "8. Hides, skins, leather, fur"),
    (9,  44, 46, "9. Wood and articles of wood"),
    (10, 47, 49, "10. Pulp of wood, paper, paperboard"),
    (11, 50, 63, "11. Textiles and textile articles"),
    (12, 64, 67, "12. Footwear, headgear, umbrellas"),
    (13, 68, 70, "13. Stone, ceramic, glass"),
    (14, 71, 71, "14. Natural or cultured pearls, precious metals"),
    (15, 72, 83, "15. Base metals and articles of base metal"),
    (16, 84, 85, "16. Machinery and electrical equipment"),
    (17, 86, 89, "17. Vehicles, aircraft, vessels"),
    (18, 90, 92, "18. Optical, precision, medical, musical instruments"),
    (19, 93, 93, "19. Arms and ammunition"),
    (20, 94, 96, "20. Miscellaneous manufactured articles"),
    (21, 97, 97, "21. Works of art and antiques"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    hex_color = str(hex_color).lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(47,93,116,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _wrap_label(text: str, width: int = 18) -> str:
    words = str(text).split()
    if not words:
        return str(text)
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "<br>".join(lines)


def _section_sort_key(label: str) -> tuple[int, str]:
    text = str(label).strip()
    match = re.match(r"^(\d+)\.\s*(.*)$", text)
    if match:
        return (int(match.group(1)), match.group(2).lower())
    return (10_000, text.lower())


def hs_section_for(hs4: str) -> tuple[int, str]:
    try:
        ch = int(str(hs4)[:2])
    except Exception:
        return 0, "(unknown)"
    for num, lo, hi, name in HS_SECTIONS:
        if lo <= ch <= hi:
            return num, name
    return 0, "(unknown)"


def normalize_0_1(s: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    out = (s - lo) / (hi - lo)
    return 1.0 - out if invert else out


def fmt_usd(v: float) -> str:
    if v is None or pd.isna(v): return "—"
    v = float(v)
    if v >= 1e9: return f"USD {v/1e9:.2f} B"
    if v >= 1e6: return f"USD {v/1e6:.2f} M"
    if v >= 1e3: return f"USD {v/1e3:.1f} K"
    return f"USD {v:.0f}"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    links = pd.read_csv(
        DATA_DIR / "cordoba_anchored_proximity.csv",
        dtype={"anchor_hs4": str, "candidate_hs4": str},
    )
    links["anchor_hs4"] = links["anchor_hs4"].astype(str).str.zfill(4)
    links["candidate_hs4"] = links["candidate_hs4"].astype(str).str.zfill(4)

    for c in ["pci", "accessible_market_size", "accessible_market_growth_5y",
              "dai_percentile", "dai_index", "distance_travelled",
              "proximity", "proximity_rank"]:
        if c in links.columns:
            links[c] = pd.to_numeric(links[c], errors="coerce")
    # Add Billions column to match Argentina convention
    links["accessible_market_size_b"] = links["accessible_market_size"] / 1e9

    # Attach HS section name
    secs = {h: hs_section_for(h) for h in
            pd.concat([links["anchor_hs4"], links["candidate_hs4"]]).dropna().unique()}
    links["anchor_hs_section_name"] = links["anchor_hs4"].map(lambda h: secs.get(h, (0, ""))[1])
    links["candidate_hs_section_name"] = links["candidate_hs4"].map(lambda h: secs.get(h, (0, ""))[1])

    # Spanish HS4 short names (override the English `*_product_name_short`)
    links["anchor_product_name_es"] = links.apply(
        lambda r: hs4_name_es(r["anchor_hs4"], r.get("anchor_product_name_short", "")),
        axis=1,
    )
    links["candidate_product_name_es"] = links.apply(
        lambda r: hs4_name_es(r["candidate_hs4"], r.get("candidate_product_name_short", "")),
        axis=1,
    )

    presence = pd.read_csv(
        DATA_DIR / "hs4_presence_by_opex_threshold.csv",
        dtype={"hs4": str},
    )
    presence["hs4"] = presence["hs4"].str.zfill(4)
    presence["max_rubro_opex_2023_2025_avg_usd"] = pd.to_numeric(
        presence["max_rubro_opex_2023_2025_avg_usd"], errors="coerce"
    )

    umap = pd.read_csv(DATA_DIR / "umap_layout_hs92.csv",
                       dtype={"product_hs92_code": str})
    umap["product_hs92_code"] = umap["product_hs92_code"].str.zfill(4)

    trade = pd.read_csv(DATA_DIR / "hs92_product_year_4.csv",
                        dtype={"product_hs92_code": str})
    trade["product_hs92_code"] = trade["product_hs92_code"].str.zfill(4)
    trade_2024 = trade[trade["year"] == 2024][["product_hs92_code", "export_value"]].copy()
    trade_2024["export_value"] = pd.to_numeric(trade_2024["export_value"], errors="coerce").fillna(0)

    clusters = pd.read_csv(DATA_DIR / "product_space_clusters.csv")
    cluster_color = dict(zip(clusters["Name"], clusters["Hex Code"]))

    names = pd.read_csv(DATA_DIR / "product_hs92.csv",
                        dtype={"product_hs92_code": str, "product_level": str})
    names = names[names["product_level"] == "4"][
        ["product_hs92_code", "product_name", "product_name_short"]
    ].drop_duplicates("product_hs92_code")
    names = names.rename(columns={"product_hs92_code": "hs4"})
    names["hs4"] = names["hs4"].str.zfill(4)
    # Spanish short name (Atlas English as fallback when no Spanish override)
    names["product_name_es"] = names.apply(
        lambda r: hs4_name_es(r["hs4"], r["product_name_short"]),
        axis=1,
    )

    return links, presence, umap, trade_2024, cluster_color, names


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Córdoba — Anchored Proximity",
    page_icon="🇦🇷",
)

st.title("Córdoba — Análisis de proximidad anclada")
st.caption(
    "Explorá productos candidatos conectados por proximidad en el espacio "
    "de productos a los HS4 donde Córdoba tiene presencia evidenciada."
)

df, presence, umap, trade_2024, cluster_color, names = load_data()

if df.empty:
    st.warning("No anchor proximity data available.")
    st.stop()

# ---------------------------------------------------------------------------
# Filter universes
# ---------------------------------------------------------------------------
anchor_sectors = sorted(df["anchor_sector"].dropna().astype(str).unique().tolist())
candidate_sectors = sorted(df["candidate_sector"].dropna().astype(str).unique().tolist())
anchor_sections = sorted(
    df["anchor_hs_section_name"].dropna().astype(str).unique().tolist(),
    key=_section_sort_key,
)
candidate_sections = sorted(
    df["candidate_hs_section_name"].dropna().astype(str).unique().tolist(),
    key=_section_sort_key,
)
candidate_products_df = (
    df[["candidate_hs4", "candidate_product_name_es"]]
    .dropna(subset=["candidate_hs4"])
    .copy()
    .assign(
        candidate_hs4=lambda d: d["candidate_hs4"].astype(str).str.zfill(4),
        candidate_product_name_es=lambda d: d["candidate_product_name_es"].fillna("").astype(str).str.strip(),
    )
    .drop_duplicates(subset=["candidate_hs4"])
    .sort_values("candidate_hs4")
)
candidate_products_df["candidate_label"] = (
    candidate_products_df["candidate_hs4"] + " - " + candidate_products_df["candidate_product_name_es"]
)
candidate_label_to_code = dict(
    zip(candidate_products_df["candidate_label"], candidate_products_df["candidate_hs4"])
)
proximity_rank_max = int(pd.to_numeric(df["proximity_rank"], errors="coerce").max())
accessible_market_max = float(df["accessible_market_size_b"].max()) if not df.empty else 0.0
top_n_default = min(40, max(10, int(df["candidate_hs4"].nunique())))
excluded_hs4_preset_codes = {x for x in NATURAL_RESOURCE_HS4}
excluded_labels_by_code = (
    candidate_products_df[candidate_products_df["candidate_hs4"].isin(excluded_hs4_preset_codes)]
    .sort_values("candidate_hs4")["candidate_label"]
    .tolist()
)
anchor_sections_excluding_123 = [
    s for s in anchor_sections if not str(s).startswith(("1.", "2.", "3."))
]
candidate_sections_excluding_123 = [
    s for s in candidate_sections if not str(s).startswith(("1.", "2.", "3."))
]

OPEX_OPTIONS = [0, 100_000, 500_000, 1_000_000, 5_000_000,
                10_000_000, 50_000_000, 100_000_000, 500_000_000]
OPEX_LABELS = {
    0: "0 (todos los HS4 evidenciados)",
    100_000: "≥ USD 100 K",
    500_000: "≥ USD 500 K",
    1_000_000: "≥ USD 1 M",
    5_000_000: "≥ USD 5 M",
    10_000_000: "≥ USD 10 M",
    50_000_000: "≥ USD 50 M",
    100_000_000: "≥ USD 100 M",
    500_000_000: "≥ USD 500 M",
}

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
st.session_state.setdefault("c4_opex_threshold", 1_000_000)
st.session_state.setdefault("c4_accessible_market_min", 0.0)
st.session_state.setdefault("c4_am_cagr_only", False)
st.session_state.setdefault("c4_strategic_balance", 0.30)
st.session_state.setdefault("c4_w_dai", 0.50)
st.session_state.setdefault("c4_w_distance", 0.50)
st.session_state.setdefault("c4_w_anchor_count", 0.0)
st.session_state.setdefault("c4_w_pci", 0.50)
st.session_state.setdefault("c4_w_growth", 0.25)
st.session_state.setdefault("c4_w_market", 0.25)
st.session_state.setdefault("c4_candidates_to_display", top_n_default)
st.session_state.setdefault("c4_selected_anchor_sectors", anchor_sectors)
st.session_state.setdefault("c4_selected_candidate_sectors", candidate_sectors)
st.session_state.setdefault("c4_selected_candidate_sections", candidate_sections)
st.session_state.setdefault("c4_selected_anchor_sections", anchor_sections)
st.session_state.setdefault("c4_excluded_product_labels", [])
st.session_state.setdefault("c4_proximity_rank_range", (1, min(100, max(1, proximity_rank_max))))


def _apply_profile(profile_name: str) -> None:
    """Reset filters and apply a preset overlay."""
    st.session_state["c4_accessible_market_min"] = 0.5
    st.session_state["c4_am_cagr_only"] = True
    st.session_state["c4_selected_anchor_sectors"] = anchor_sectors
    st.session_state["c4_selected_candidate_sectors"] = candidate_sectors
    st.session_state["c4_selected_candidate_sections"] = candidate_sections
    st.session_state["c4_selected_anchor_sections"] = anchor_sections
    st.session_state["c4_excluded_product_labels"] = []
    st.session_state["c4_proximity_rank_range"] = (1, min(100, max(1, proximity_rank_max)))

    if profile_name == "top_candidates":
        st.session_state["c4_strategic_balance"] = 0.70
        st.session_state["c4_w_pci"] = 0.50
        st.session_state["c4_w_growth"] = 0.25
        st.session_state["c4_w_market"] = 0.25
        st.session_state["c4_w_dai"] = 0.40
        st.session_state["c4_w_distance"] = 0.40
        st.session_state["c4_w_anchor_count"] = 0.20
        st.session_state["c4_candidates_to_display"] = 30
        st.session_state["c4_selected_anchor_sections"] = anchor_sections_excluding_123
        st.session_state["c4_selected_candidate_sections"] = candidate_sections_excluding_123
        st.session_state["c4_excluded_product_labels"] = excluded_labels_by_code
        st.session_state["c4_proximity_rank_range"] = (1, min(10, max(1, proximity_rank_max)))
        st.session_state["c4_opex_threshold"] = 10_000_000


def _reset_filters() -> None:
    st.session_state["c4_opex_threshold"] = 1_000_000
    st.session_state["c4_accessible_market_min"] = 0.0
    st.session_state["c4_am_cagr_only"] = False
    st.session_state["c4_selected_anchor_sectors"] = anchor_sectors
    st.session_state["c4_selected_candidate_sectors"] = candidate_sectors
    st.session_state["c4_selected_candidate_sections"] = candidate_sections
    st.session_state["c4_selected_anchor_sections"] = anchor_sections
    st.session_state["c4_excluded_product_labels"] = []
    st.session_state["c4_proximity_rank_range"] = (1, min(100, max(1, proximity_rank_max)))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Perfiles predefinidos")
    st.button(
        "Top perfil seleccionado",
        on_click=_apply_profile,
        args=("top_candidates",),
        use_container_width=True,
    )

    st.header("Presencia (set de anclas)")
    opex_threshold = st.select_slider(
        "Umbral OPEX (promedio 2023-2025)",
        options=OPEX_OPTIONS,
        value=st.session_state["c4_opex_threshold"],
        format_func=lambda v: OPEX_LABELS[v],
        key="c4_opex_threshold",
        help=(
            "Define el set de anclas (anchors) para Córdoba. Sólo los HS4 con "
            "presencia evidenciada cuya CCOD_RUBRO de destino tiene un promedio "
            "anual ≥ este umbral se consideran anclas."
        ),
    )

    st.header("Filtros")
    st.button(
        "Reiniciar todos los filtros",
        on_click=_reset_filters,
        use_container_width=True,
    )
    selected_anchor_sectors = st.multiselect(
        "Sector del ancla",
        options=anchor_sectors,
        default=st.session_state["c4_selected_anchor_sectors"],
        key="c4_selected_anchor_sectors",
    )
    selected_candidate_sectors = st.multiselect(
        "Sector del candidato",
        options=candidate_sectors,
        default=st.session_state["c4_selected_candidate_sectors"],
        key="c4_selected_candidate_sectors",
    )
    selected_anchor_sections = st.multiselect(
        "Sección HS del ancla",
        options=anchor_sections,
        default=st.session_state["c4_selected_anchor_sections"],
        key="c4_selected_anchor_sections",
    )
    selected_candidate_sections = st.multiselect(
        "Sección HS del candidato",
        options=candidate_sections,
        default=st.session_state["c4_selected_candidate_sections"],
        key="c4_selected_candidate_sections",
    )
    excluded_product_labels = st.multiselect(
        "Excluir productos (HS4)",
        options=candidate_products_df["candidate_label"].tolist(),
        default=st.session_state["c4_excluded_product_labels"],
        key="c4_excluded_product_labels",
    )
    proximity_rank_range = st.slider(
        "Rango de ranking de proximidad",
        min_value=1,
        max_value=max(1, proximity_rank_max),
        value=st.session_state["c4_proximity_rank_range"],
        step=1,
        key="c4_proximity_rank_range",
        help=(
            "Por cada ancla, los candidatos están rankeados por proximidad "
            "(1 = más cercano). Filtrá para ver sólo los más cercanos a cada ancla."
        ),
    )
    accessible_market_min = st.number_input(
        "Mercado accesible mínimo (USD mil millones)",
        min_value=0.0,
        max_value=float(max(accessible_market_max, 0.1)),
        value=float(st.session_state["c4_accessible_market_min"]),
        step=0.1,
        format="%.2f",
        key="c4_accessible_market_min",
    )
    am_cagr_only = st.toggle(
        "Sólo crecimiento del mercado accesible (5 años) > 0",
        value=bool(st.session_state["c4_am_cagr_only"]),
        key="c4_am_cagr_only",
    )

    st.header("Balance de dimensiones")
    strategic_balance = st.slider(
        "Factibilidad (100%) = 0 | Atractividad (100%) = 1",
        0.0, 1.0,
        float(st.session_state["c4_strategic_balance"]),
        0.05,
        key="c4_strategic_balance",
        help=(
            "Score combinado = (1 − valor)·Factibilidad + valor·Atractividad. "
            "0 = sólo factibilidad; 1 = sólo atractividad."
        ),
    )

    st.header("Pesos de los componentes")
    st.caption("Factibilidad: DAI + distancia + # de anclas. Atractividad: PCI + tamaño + crecimiento (sin COG).")
    with st.expander("Componentes de factibilidad", expanded=True):
        w_dai = st.slider(
            "Peso del DAI", 0.0, 1.0,
            float(st.session_state["c4_w_dai"]), 0.05, key="c4_w_dai",
            help="Demand Alignment Index: qué tan alineada está la demanda externa con el patrón exportador.",
        )
        w_distance = st.slider(
            "Peso del percentil de distancia recorrida", 0.0, 1.0,
            float(st.session_state["c4_w_distance"]), 0.05, key="c4_w_distance",
            help=(
                "Distancia geográfica promedio (km) que viaja cada HS4 a nivel "
                "global, ponderada por flujos bilaterales 2020-2024. Mayor "
                "distancia = producto globalmente comercializable = más factible "
                "para Córdoba (el percentil entra directo, sin inversión)."
            ),
        )
        w_anchor_count = st.slider(
            "Peso del # de anclas (normalizado)", 0.0, 1.0,
            float(st.session_state["c4_w_anchor_count"]), 0.05, key="c4_w_anchor_count",
            help=(
                "Cantidad de anclas que tienen al candidato en su top-1% de "
                "proximidad. Más anclas = el salto de capacidades se sostiene "
                "desde varias bases existentes = más factible. Se normaliza "
                "min-max dentro del set filtrado."
            ),
        )
    with st.expander("Componentes de atractividad", expanded=True):
        w_pci = st.slider(
            "Peso del PCI", 0.0, 1.0,
            float(st.session_state["c4_w_pci"]), 0.05, key="c4_w_pci",
            help="Product Complexity Index — sofisticación productiva del HS4.",
        )
        w_growth = st.slider(
            "Peso del crecimiento del mercado accesible (5 años)", 0.0, 1.0,
            float(st.session_state["c4_w_growth"]), 0.05, key="c4_w_growth",
        )
        w_market = st.slider(
            "Peso del tamaño del mercado accesible", 0.0, 1.0,
            float(st.session_state["c4_w_market"]), 0.05, key="c4_w_market",
        )

# ---------------------------------------------------------------------------
# 1. Derive anchor universe from OPEX threshold + apply filters
# ---------------------------------------------------------------------------
# evidenced_set is the full set of 184 firm-evidenced HS4. The anchor universe
# is the subset of those whose OPEX clears the threshold. Evidenced HS4 that
# fall BELOW the threshold can resurface as candidates of the surviving
# anchors — they're flagged `posible_ancla = 1` so users can spot them.
evidenced_set = set(presence["hs4"].astype(str).str.zfill(4))
anchor_universe = set(
    presence.loc[
        presence["max_rubro_opex_2023_2025_avg_usd"].fillna(0) >= opex_threshold, "hs4"
    ]
)
flt = df[
    df["anchor_hs4"].isin(anchor_universe)
    & ~df["candidate_hs4"].isin(anchor_universe)
].copy()
flt["posible_ancla"] = flt["candidate_hs4"].isin(evidenced_set).astype(int)

if selected_anchor_sectors:
    flt = flt[flt["anchor_sector"].isin(selected_anchor_sectors)]
if selected_candidate_sectors:
    flt = flt[flt["candidate_sector"].isin(selected_candidate_sectors)]
if selected_anchor_sections:
    flt = flt[flt["anchor_hs_section_name"].isin(selected_anchor_sections)]
if selected_candidate_sections:
    flt = flt[flt["candidate_hs_section_name"].isin(selected_candidate_sections)]
excluded_hs4_codes = {candidate_label_to_code[label] for label in excluded_product_labels}
if excluded_hs4_codes:
    flt = flt[~flt["candidate_hs4"].astype(str).str.zfill(4).isin(excluded_hs4_codes)]
flt = flt[
    (pd.to_numeric(flt["proximity_rank"], errors="coerce") >= proximity_rank_range[0])
    & (pd.to_numeric(flt["proximity_rank"], errors="coerce") <= proximity_rank_range[1])
]
flt = flt[pd.to_numeric(flt["accessible_market_size_b"], errors="coerce") >= float(accessible_market_min)]
if am_cagr_only:
    flt = flt[pd.to_numeric(flt["accessible_market_growth_5y"], errors="coerce") > 0.0]

# ---------------------------------------------------------------------------
# 2. Candidate-level aggregation + scoring
# ---------------------------------------------------------------------------
if flt.empty:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Universo de anclas", f"{len(anchor_universe)}", f"de {len(presence)} evidenciados")
    k2.metric("Links visibles", "0")
    k3.metric("Anclas únicas", "0")
    k4.metric("Candidatos únicos", "0")
    st.info("Ningún link ancla–candidato coincide con los filtros actuales.")
    st.stop()

candidate_scores = (
    flt.groupby(["candidate_hs4", "candidate_product_name_es", "candidate_sector"], as_index=False)
    .agg(
        accessible_market_size=("accessible_market_size", "first"),
        accessible_market_size_b=("accessible_market_size_b", "first"),
        accessible_market_growth_5y=("accessible_market_growth_5y", "first"),
        dai_index=("dai_index", "first"),
        dai_percentile=("dai_percentile", "first"),
        pci=("pci", "first"),
        distance_travelled=("distance_travelled", "first"),
        avg_proximity=("proximity", "mean"),
        anchor_count=("anchor_hs4", "nunique"),
        posible_ancla=("posible_ancla", "max"),
    )
)

# Normalise components within the filtered candidate set
candidate_scores["dai_mm"] = normalize_0_1(candidate_scores["dai_percentile"])
# Distance: percentile rank (0 → 1, higher rank = travels farther = more
# globally tradeable = more feasible for Córdoba). NOT inverted.
candidate_scores["distance_pctile"] = (
    pd.to_numeric(candidate_scores["distance_travelled"], errors="coerce")
    .rank(pct=True)
    .fillna(0.5)
)
candidate_scores["pci_mm"] = normalize_0_1(candidate_scores["pci"])
candidate_scores["accessible_market_growth_mm"] = normalize_0_1(candidate_scores["accessible_market_growth_5y"])
candidate_scores["accessible_market_size_mm"] = normalize_0_1(candidate_scores["accessible_market_size"])
# Anchor count: more anchors pointing to a candidate = more feasible (the
# capability stretch is supported from several existing bases).
candidate_scores["anchor_count_mm"] = normalize_0_1(candidate_scores["anchor_count"])

# Feasibility: DAI + distance percentile + anchor count (all direct, higher = more feasible)
feas_cols = ["dai_mm", "distance_pctile", "anchor_count_mm"]
feas_weights = np.array([w_dai, w_distance, w_anchor_count], dtype=float)
feas_total = float(feas_weights.sum())
if feas_total <= 0:
    candidate_scores["feasibility_raw"] = candidate_scores[feas_cols].mean(axis=1)
else:
    candidate_scores["feasibility_raw"] = (
        candidate_scores[feas_cols].to_numpy() * feas_weights
    ).sum(axis=1) / feas_total
candidate_scores["feasibility_index"] = normalize_0_1(candidate_scores["feasibility_raw"])

# Attractiveness: PCI + market size + market growth (NO COG)
attr_cols = ["pci_mm", "accessible_market_growth_mm", "accessible_market_size_mm"]
attr_weights = np.array([w_pci, w_growth, w_market], dtype=float)
attr_total = float(attr_weights.sum())
if attr_total <= 0:
    candidate_scores["attractiveness_raw"] = candidate_scores[attr_cols].mean(axis=1)
else:
    candidate_scores["attractiveness_raw"] = (
        candidate_scores[attr_cols].to_numpy() * attr_weights
    ).sum(axis=1) / attr_total
candidate_scores["attractiveness_index"] = normalize_0_1(candidate_scores["attractiveness_raw"])

candidate_scores["combined_raw"] = (
    (1 - strategic_balance) * candidate_scores["feasibility_index"]
    + strategic_balance * candidate_scores["attractiveness_index"]
)
candidate_scores["combined_score"] = normalize_0_1(candidate_scores["combined_raw"])

# Merge back into links for downstream coloring/sankey
flt = flt.merge(
    candidate_scores[[
        "candidate_hs4", "candidate_product_name_es",
        "feasibility_index", "attractiveness_index", "combined_score",
    ]],
    on=["candidate_hs4", "candidate_product_name_es"],
    how="left",
)

# ---------------------------------------------------------------------------
# 3. KPI strip
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Universo de anclas", f"{len(anchor_universe)}", f"de {len(presence)} evidenciados")
k2.metric("Links visibles", f"{len(flt):,}")
k3.metric("Anclas únicas", f"{flt['anchor_hs4'].nunique():,}")
k4.metric("Candidatos únicos", f"{flt['candidate_hs4'].nunique():,}")

# ---------------------------------------------------------------------------
# 4. Product space (Plotly scatter)
# ---------------------------------------------------------------------------
st.subheader("Espacio de productos — Córdoba")
st.caption(
    "Anclas (HS4 con presencia ≥ umbral OPEX) coloreadas por cluster del Atlas; "
    "el resto del universo en gris claro. Pasá el mouse sobre un punto para ver HS4 + nombre."
)

trade_lookup = dict(zip(trade_2024["product_hs92_code"], trade_2024["export_value"]))
name_es_lookup = dict(zip(names["hs4"], names["product_name_es"]))


def dot_radius(v: float) -> float:
    """log10-based radius. Same formula for all dots: presence is signalled
    by color, not size."""
    if v is None or v <= 0: return 4.0
    lv = math.log10(v)
    return float(np.clip(4 + (lv - 6) * 12 / 6, 4, 16))


umap_plot = umap.copy()
umap_plot["radius"] = umap_plot["product_hs92_code"].map(lambda h: dot_radius(trade_lookup.get(h, 0)))
umap_plot["in_anchor"] = umap_plot["product_hs92_code"].isin(anchor_universe)
umap_plot["cluster_std"] = umap_plot["product_space_cluster_name"].map(UMAP_TO_CLUSTER).fillna("")
umap_plot["color"] = umap_plot["cluster_std"].map(cluster_color).fillna("#e0e0e3")
umap_plot["product_name_es"] = umap_plot["product_hs92_code"].map(lambda h: name_es_lookup.get(h, ""))

fig_ps = go.Figure()
# Background: HS4 NOT in the current anchor set — same size, light grey
bg = umap_plot[~umap_plot["in_anchor"]]
fig_ps.add_trace(go.Scatter(
    x=bg["product_space_x"], y=bg["product_space_y"], mode="markers",
    marker=dict(size=bg["radius"], color="#eeeef2",
                line=dict(width=0.4, color="#d0d0d6")),
    text=[f"HS {h} · {n} · no es ancla" for h, n in zip(bg["product_hs92_code"], bg["product_name_es"])],
    hoverinfo="text", name="Otros", showlegend=False,
))
# Anchors — same size as background, coloured by cluster (presence = colour)
ank = umap_plot[umap_plot["in_anchor"]]
fig_ps.add_trace(go.Scatter(
    x=ank["product_space_x"], y=ank["product_space_y"], mode="markers",
    marker=dict(size=ank["radius"], color=ank["color"],
                line=dict(width=0.6, color="#ffffff"), opacity=0.92),
    text=[f"<b>HS {h}</b> · {n}<br>Ancla" for h, n in zip(ank["product_hs92_code"], ank["product_name_es"])],
    hoverinfo="text", name="Anclas", showlegend=False,
))
fig_ps.update_layout(
    height=620, margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    plot_bgcolor="white",
    hoverlabel=dict(bgcolor="rgba(16,24,44,0.95)", font_color="white"),
)
st.plotly_chart(fig_ps, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Sankey diagram (anchor → candidate)
# ---------------------------------------------------------------------------
anchor_labels = (
    flt[["anchor_hs4", "anchor_product_name_es", "anchor_sector"]]
    .drop_duplicates()
    .assign(node_label=lambda d: d["anchor_hs4"] + " - " + d["anchor_product_name_es"].astype(str))
)
candidate_labels = (
    flt[["candidate_hs4", "candidate_product_name_es", "candidate_sector"]]
    .drop_duplicates()
    .assign(node_label=lambda d: d["candidate_hs4"] + " - " + d["candidate_product_name_es"].astype(str))
)

anchor_node_ids = {k: i for i, k in enumerate(anchor_labels["node_label"].tolist())}
candidate_offset = len(anchor_node_ids)
candidate_node_ids = {k: candidate_offset + i for i, k in enumerate(candidate_labels["node_label"].tolist())}

links_built = flt.assign(
    anchor_node=lambda d: d["anchor_hs4"].astype(str).str.zfill(4) + " - " + d["anchor_product_name_es"].astype(str),
    candidate_node=lambda d: d["candidate_hs4"].astype(str).str.zfill(4) + " - " + d["candidate_product_name_es"].astype(str),
)

sankey_df = (
    links_built.groupby(
        ["anchor_node", "candidate_node", "anchor_sector", "candidate_sector"],
        as_index=False,
    )["proximity"].sum()
)
sankey_df["source"] = sankey_df["anchor_node"].map(anchor_node_ids)
sankey_df["target"] = sankey_df["candidate_node"].map(candidate_node_ids)

node_labels = anchor_labels["node_label"].tolist() + candidate_labels["node_label"].tolist()
node_colors = (
    [SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]) for s in anchor_labels["anchor_sector"].tolist()]
    + [SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]) for s in candidate_labels["candidate_sector"].tolist()]
)
link_colors = [
    _hex_to_rgba(SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]), alpha=0.4)
    for s in sankey_df["candidate_sector"].tolist()
]

sankey = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=18, thickness=18,
        line=dict(color="rgba(15,23,42,0.15)", width=0.6),
        label=node_labels, color=node_colors,
    ),
    link=dict(
        source=sankey_df["source"],
        target=sankey_df["target"],
        value=sankey_df["proximity"].clip(lower=0.000001),
        color=link_colors,
        customdata=np.stack(
            [sankey_df["anchor_sector"], sankey_df["candidate_sector"], sankey_df["proximity"]],
            axis=-1,
        ),
        hovertemplate=(
            "Sector del ancla: %{customdata[0]}<br>"
            "Sector del candidato: %{customdata[1]}<br>"
            "Proximidad total: %{customdata[2]:.4f}<extra></extra>"
        ),
    ),
))
sankey.update_layout(
    title="Sankey — anclas → candidatos por proximidad",
    font=dict(size=12),
    margin=dict(t=60, l=10, r=10, b=10),
    height=760,
)
st.plotly_chart(sankey, use_container_width=True)

# ---------------------------------------------------------------------------
# 6. Candidate ranking table
# ---------------------------------------------------------------------------
top_n_max = max(10, int(candidate_scores["candidate_hs4"].nunique()))
st.session_state["c4_candidates_to_display"] = min(
    int(st.session_state.get("c4_candidates_to_display", top_n_default)),
    top_n_max,
)
top_n = st.slider(
    "Candidatos a mostrar (tabla + treemap)",
    min_value=10,
    max_value=top_n_max,
    value=int(st.session_state["c4_candidates_to_display"]),
    step=1,
    key="c4_candidates_to_display",
)

candidate_table = (
    candidate_scores
    .sort_values(["combined_score", "accessible_market_size"], ascending=[False, False])
    .reset_index(drop=True)
)
candidate_table["rank"] = np.arange(1, len(candidate_table) + 1)
candidate_display = (
    candidate_table.head(top_n).copy().assign(
        accessible_market_growth_5y=lambda d: d["accessible_market_growth_5y"] * 100,
    )
)

st.dataframe(
    candidate_display[[
        "rank", "candidate_hs4", "candidate_product_name_es", "candidate_sector",
        "posible_ancla",
        "combined_score", "attractiveness_index", "feasibility_index",
        "dai_index", "pci", "dai_percentile", "distance_travelled",
        "accessible_market_growth_5y", "accessible_market_size_b",
        "avg_proximity", "anchor_count",
    ]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("Ranking", format="%.0f"),
        "candidate_hs4": st.column_config.TextColumn("HS4"),
        "candidate_product_name_es": st.column_config.TextColumn("Producto"),
        "candidate_sector": st.column_config.TextColumn("Sector"),
        "posible_ancla": st.column_config.NumberColumn(
            "Posible ancla",
            format="%d",
            help=(
                "1 = HS4 con presencia evidenciada en Córdoba pero OPEX por "
                "debajo del umbral actual; reaparece como candidato del set "
                "de anclas que sí supera el umbral. 0 = candidato 'puro' "
                "(sin evidencia previa de presencia)."
            ),
        ),
        "combined_score": st.column_config.NumberColumn("Puntaje combinado", format="%.4f"),
        "attractiveness_index": st.column_config.NumberColumn("Índice de atractividad", format="%.4f"),
        "feasibility_index": st.column_config.NumberColumn("Índice de factibilidad", format="%.4f"),
        "dai_index": st.column_config.NumberColumn("DAI (crudo)", format="%.3f"),
        "pci": st.column_config.NumberColumn("PCI", format="%.3f"),
        "dai_percentile": st.column_config.NumberColumn("DAI (percentil)", format="%.1f"),
        "distance_travelled": st.column_config.NumberColumn("Distancia recorrida", format="%.1f"),
        "accessible_market_growth_5y": st.column_config.NumberColumn("Crecimiento del mercado accesible % (5 años)", format="%.2f%%"),
        "accessible_market_size_b": st.column_config.NumberColumn("Mercado accesible (USD mil M)", format="%.3f"),
        "avg_proximity": st.column_config.NumberColumn("Proximidad promedio", format="%.4f"),
        "anchor_count": st.column_config.NumberColumn("# anclas", format="%.0f"),
    },
)

# Download button (Córdoba-specific, kept from previous version)
csv_bytes = candidate_display.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Descargar top candidatos (CSV)",
    csv_bytes,
    "cordoba_top_candidates.csv",
    "text/csv",
)

# ---------------------------------------------------------------------------
# 7. Treemap (Argentina styling)
# ---------------------------------------------------------------------------
st.subheader("Candidatos por proximidad anclada")
treemap_color_label = st.selectbox(
    "Variable de color del treemap",
    ["Sector", "PCI (crudo)"],
    key="c4_treemap_color",
)

treemap_df = candidate_table.head(top_n).copy()
treemap_df["product_label"] = (
    treemap_df["candidate_hs4"].astype(str).str.zfill(4)
    + " - " + treemap_df["candidate_product_name_es"].astype(str)
)
treemap_df["product_label_wrapped"] = treemap_df["product_label"].map(_wrap_label)

# PCI color bounds — use 2-98 percentiles like Argentina, fallback to min/max
pci_series = pd.to_numeric(candidate_scores["pci"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
if pci_series.empty:
    pci_color_min, pci_color_max = 0.0, 1.0
else:
    pci_color_min = float(pci_series.quantile(0.02))
    pci_color_max = float(pci_series.quantile(0.98))
    if pci_color_max <= pci_color_min:
        pci_color_max = pci_color_min + 1e-6
treemap_df["pci_for_color"] = pd.to_numeric(treemap_df["pci"], errors="coerce").clip(pci_color_min, pci_color_max)

st.metric(
    label="Mercado accesible total mostrado (USD mil millones)",
    value=f"{treemap_df['accessible_market_size_b'].sum():,.3f}",
)

if treemap_color_label == "Sector":
    treemap_color_kwargs = {
        "color": "candidate_sector",
        "color_discrete_map": SECTOR_COLORS,
    }
else:
    treemap_color_kwargs = {
        "color": "pci_for_color",
        "color_continuous_scale": PCI_COLORSCALE,
        "range_color": (pci_color_min, pci_color_max),
    }

treemap = px.treemap(
    treemap_df,
    path=["candidate_sector", "product_label_wrapped"],
    values="accessible_market_size_b",
    **treemap_color_kwargs,
    hover_data={
        "combined_score": ":.3f",
        "accessible_market_size_b": ":.3f",
        "accessible_market_growth_5y": ":.3%",
        "dai_index": ":.3f",
        "dai_percentile": ":.1f",
        "pci": ":.3f",
        "anchor_count": ":.0f",
        "avg_proximity": ":.4f",
        "candidate_sector": False,
        "product_label_wrapped": False,
    },
    title=(
        f"Candidatos por proximidad anclada (n = {len(treemap_df)} candidatos | "
        f"Mercado accesible total = {treemap_df['accessible_market_size_b'].sum():,.3f} USD mil M) "
        f"| tamaño = mercado accesible (USD mil M) | color = {treemap_color_label}"
    ),
)
treemap.update_traces(
    textinfo="label",
    textfont=dict(size=18, color="#ffffff"),
    marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
)
treemap.update_layout(margin=dict(t=60, l=10, r=10, b=95))
if treemap_color_label == "Sector":
    treemap.update_layout(
        legend=dict(
            orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
            title_text="Sector",
        ),
    )
else:
    treemap.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="PCI", font=dict(color="#0f172a", size=16)),
            orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
            len=0.7,
            bgcolor="rgba(255,255,255,0.96)",
            borderwidth=0,
            tickfont=dict(color="#0f172a", size=14),
            tickcolor="#0f172a", ticklen=6, tickwidth=1.2,
        ),
    )

st.plotly_chart(treemap, use_container_width=True)
