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
import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


def _data(subdir: str, name: str) -> Path:
    """Locate a data file across two layouts:

      • Monorepo:  Oportunidades/data/{output,intermediate,input}/{name}
                   (ROOT = Oportunidades/)
      • Standalone (gl-cordoba-dashboard on Streamlit Cloud):
                   gl-cordoba-dashboard/data/{name}   (flat)

    Tries monorepo first (subdir-aware), then the flat layout beside the app.
    Returns the monorepo path as fallback so the natural FileNotFoundError
    message is informative.
    """
    p_mono = ROOT / "data" / subdir / name
    if p_mono.exists():
        return p_mono
    p_flat = Path(__file__).resolve().parent / "data" / name
    if p_flat.exists():
        return p_flat
    p_root_flat = ROOT / "data" / name
    if p_root_flat.exists():
        return p_root_flat
    return p_mono

# Spanish HS4 short names. `hs4_names_es.py` (~1,240 entries) already covers
# the anchor + candidate universe. When running inside the monorepo, we still
# layer script-13's dict on top (canonical source) so any edits there
# propagate; when running standalone (Streamlit Cloud), `scripts/` is absent
# and we fall back to `hs4_names_es.py` alone.
_cand_spec = importlib.util.spec_from_file_location(
    "hs4_names_es", Path(__file__).resolve().parent / "hs4_names_es.py"
)
_cand_mod = importlib.util.module_from_spec(_cand_spec)
_cand_spec.loader.exec_module(_cand_mod)
SPANISH_OVERRIDES: dict[str, str] = dict(_cand_mod.HS4_NAMES_ES)

_s13_path = ROOT / "scripts" / "13_filter_and_visualize_by_opex.py"
if _s13_path.exists():
    _s13_spec = importlib.util.spec_from_file_location("s13", _s13_path)
    _s13 = importlib.util.module_from_spec(_s13_spec)
    _s13_spec.loader.exec_module(_s13)
    SPANISH_OVERRIDES.update(_s13.SPANISH_OVERRIDES)


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

# Legend labels in Spanish (for the product-space scatter legend below page 2)
CLUSTER_ES_LABEL: dict[str, str] = {
    "Agricultural Goods":       "Bienes agrícolas",
    "Construction Goods":       "Bienes de construcción",
    "Electronics":              "Electrónica",
    "Chemicals & Basic Metals": "Químicos y metales básicos",
    "Metalworking Machinery":   "Maquinaria metalmecánica",
    "Minerals":                 "Minerales",
    "Textile & Home Goods":     "Textiles y hogar",
    "Apparel":                  "Indumentaria",
}

NATURAL_RESOURCE_HS4 = [
    # Ch. 25 — Salt, sulfur, earths, stone; plaster, cement (28 códigos)
    "2501", "2502", "2503", "2504", "2505", "2506", "2507", "2508",
    "2509", "2510", "2511", "2512", "2513", "2514", "2515", "2516",
    "2517", "2518", "2519", "2520", "2521", "2524", "2525", "2526",
    "2527", "2528", "2529", "2530",
    # Ch. 26 — Minerales, escorias y cenizas (17 códigos)
    "2601", "2602", "2603", "2604", "2605", "2606", "2607", "2608",
    "2609", "2610", "2611", "2612", "2613", "2614", "2615", "2616",
    "2617",
    # Ch. 27 — Combustibles minerales, petróleo, gas (16 códigos)
    "2701", "2702", "2703", "2704", "2705", "2706", "2707", "2708",
    "2709", "2710", "2711", "2712", "2713", "2714", "2715", "2716",
    # Ch. 71 — Piedras, metales preciosos (4 códigos)
    "7101", "7102", "7103", "7108",
]

# Additional HS4 excluded from the "Recomendado" preset by hand — plausible
# via proximity but not realistic diversification targets for Córdoba.
#   4011 — Neumáticos nuevos (industria capital-intensiva sin base local)
#   8473 — Partes y accesorios para máquinas de oficina / cómputo
#   8542 — Circuitos integrados electrónicos (semiconductores)
PRESET_EXCLUDED_HS4 = ["4011", "8473", "8542"]

# Default exporter-profile subset used to compute the anchor universe.
# 'No Exporta / Próxima a Exportar' is excluded — those firms declared NCM
# in the Procórdoba registry but don't actually export, so they don't
# constitute evidence for the diversification analysis.
DEFAULT_ANCHOR_PROFILES = ("Exportadora Habitual", "Exportadora Ocasional")

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
        _data("output", "cordoba_anchored_proximity.csv"),
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
        _data("intermediate", "hs4_presence_by_opex_threshold.csv"),
        dtype={"hs4": str},
    )
    presence["hs4"] = presence["hs4"].str.zfill(4)
    # `usd_promedio_anual` is the new per-HS4 export value from the DPE
    # (Dirección de Estadística de la Provincia) share file scaled by the
    # OPEX absolute total. Replaces the old rubro-inherited OPEX value.
    for c in ["usd_promedio_anual", "usd_acumulado_2023_2025", "share_2023_2025",
              "max_rubro_opex_2023_2025_avg_usd"]:
        if c in presence.columns:
            presence[c] = pd.to_numeric(presence[c], errors="coerce")

    umap = pd.read_csv(_data("input", "umap_layout_hs92.csv"),
                       dtype={"product_hs92_code": str})
    umap["product_hs92_code"] = umap["product_hs92_code"].str.zfill(4)

    trade = pd.read_csv(_data("input", "hs92_product_year_4.csv"),
                        dtype={"product_hs92_code": str})
    trade["product_hs92_code"] = trade["product_hs92_code"].str.zfill(4)
    trade_2024 = trade[trade["year"] == 2024][["product_hs92_code", "export_value"]].copy()
    trade_2024["export_value"] = pd.to_numeric(trade_2024["export_value"], errors="coerce").fillna(0)

    clusters = pd.read_csv(_data("input", "product_space_clusters.csv"))
    cluster_color = dict(zip(clusters["Name"], clusters["Hex Code"]))

    names = pd.read_csv(_data("input", "product_hs92.csv"),
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

# ---------------------------------------------------------------------------
# Loaders + page functions
# ---------------------------------------------------------------------------
@st.cache_data
def load_firms_data(_signature: str = ""):
    """Union of two firm-HS4 evidence layers:
       - curated:      hand-verified attributions with URL (03).
                       Assumed 'Exportadora Habitual' by default.
       - declared-ncm: NCM codes declared by firms in their Procórdoba
                       'Oferta exportable' table (06). Includes all three
                       exporter_profile values (Habitual, Ocasional, and
                       'No Exporta / Próxima a Exportar') — the user can
                       filter them in the sidebar.
       Curated wins over declared-ncm on conflict."""
    curated = pd.read_csv(
        _data("output", "03_firm_hs4_evidence.csv"),
        dtype={"firm_id": "string", "hs4": str, "supports_top50_line": str},
    )
    curated["hs4"] = curated["hs4"].astype(str).str.zfill(4)
    curated["supports_top50_line"] = curated["supports_top50_line"].astype(str).str.strip()

    curated_norm = pd.DataFrame({
        "firm_id": curated["firm_id"].astype(str),
        "firm_name": curated["firm_name"].astype(str),
        "razon_social": curated["razon_social"].astype(str),
        "hs4": curated["hs4"],
        "rubro_indec": curated["supports_top50_line"],
        "rubro_indec_nombre": curated["top50_line_name"].astype(str),
        "attribution_type": curated["attribution_type"].astype(str),
        "confidence": curated["confidence"].astype(str),
        "evidence_text": curated["hs4_evidence"].astype(str),
        "evidence_url": curated["hs4_source_url"].astype(str),
        "source_url": curated["cordoba_evidence_url"].astype(str),
        "evidence_layer": "curated",
        "exporter_profile": "Exportadora Habitual",
    })

    declared = pd.read_csv(
        _data("output", "06_registry_ncm_declared.csv"),
        dtype={"firm_id": "string", "hs4": str, "ncm": str, "exporter_profile": str},
    )
    declared["hs4"] = declared["hs4"].astype(str).str.zfill(4)
    declared["exporter_profile"] = declared["exporter_profile"].fillna("").astype(str)

    # Look up CCOD_RUBRO for every HS4 in declared-ncm. The mapping covers
    # 1,243 HS4; the four not covered (3824, 4114, 6006, 8487) fall back to
    # 39899 'Manufacturas de origen industrial (confidencial)' — INDEC's
    # industrial catch-all — so no HS4 is left without a rubro attribution.
    hs4_map = pd.read_csv(
        _data("intermediate", "hs4_to_ccod_rubro_mapping.csv"),
        dtype={"hs4": str, "ccod_rubro": str},
    )
    hs4_map["hs4"] = hs4_map["hs4"].astype(str).str.zfill(4)
    hs4_map = hs4_map.drop_duplicates("hs4").set_index("hs4")
    rubro_code_lookup = hs4_map["ccod_rubro"].to_dict()
    rubro_name_lookup = hs4_map["rubro_name"].to_dict()
    _fallback_rubro = "39899"
    _fallback_name = "Manufacturas de origen industrial (confidencial)"

    _declared_rubro_code = declared["hs4"].map(rubro_code_lookup).fillna(_fallback_rubro)
    _declared_rubro_name = declared["hs4"].map(rubro_name_lookup).fillna(_fallback_name)

    declared_norm = pd.DataFrame({
        "firm_id": declared["firm_id"].astype(str),
        "firm_name": declared["firm_name"].astype(str),
        "razon_social": declared["razon_social"].astype(str),
        "hs4": declared["hs4"],
        "rubro_indec": _declared_rubro_code.astype(str),
        "rubro_indec_nombre": _declared_rubro_name.astype(str),
        "attribution_type": "declared-ncm",
        "confidence": "high",
        "evidence_text": (
            "NCM " + declared["ncm"].astype(str)
            + " — " + declared["product_name_declared"].astype(str)
            + " · perfil: " + declared["exporter_profile"]
        ),
        "evidence_url": declared["source_url"].astype(str),
        "source_url": declared["source_url"].astype(str),
        "evidence_layer": "declared-ncm",
        "exporter_profile": declared["exporter_profile"],
    })

    curated_pairs = set(curated_norm["firm_id"] + "|" + curated_norm["hs4"])
    dec_dedup = declared_norm[~(declared_norm["firm_id"] + "|" + declared_norm["hs4"]).isin(curated_pairs)].copy()
    firm_ev = pd.concat([curated_norm, dec_dedup], ignore_index=True)

    opex = pd.read_csv(
        _data("input", "exportaciones_opex_cordoba.csv"),
        dtype={"CCOD_RUBRO": str},
    )
    opex.columns = [c.lstrip("\ufeff").strip() for c in opex.columns]
    for c in ["2023", "2024", "2025", "2023_2025_avg"]:
        if c in opex.columns:
            opex[c] = (
                opex[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.strip()
            )
            opex[c] = pd.to_numeric(opex[c], errors="coerce")
    opex["CCOD_RUBRO"] = opex["CCOD_RUBRO"].astype(str).str.strip()
    return firm_ev, opex


def _page_header(title: str, caption: str = "") -> None:
    """Render the shared page header: bandera Córdoba (left) + title +
    Growth Lab logo (right). Used across all pages."""
    _assets = Path(__file__).resolve().parent / "assets"
    col_flag, col_title, col_gl = st.columns([1, 4, 2], vertical_alignment="center")
    with col_flag:
        st.image(str(_assets / "bandera_cordoba.svg"), width=130)
    with col_title:
        st.title(title)
        if caption:
            st.caption(caption)
    with col_gl:
        st.image(str(_assets / "growth_lab_logo.png"), width=220)
    st.markdown("---")


# ISO3 → continente (Spanish). Cubre ~180 países comerciales.
_ISO3_TO_CONTINENT: dict[str, str] = {
    # América
    **{c: "América" for c in [
        "USA","CAN","MEX","BRA","ARG","CHL","COL","PER","URY","VEN","BOL","PRY","ECU",
        "GTM","CRI","PAN","DOM","CUB","HTI","SLV","JAM","TTO","HND","NIC","BLZ","SUR",
        "GUY","BHS","BRB","VCT","ATG","GRD","LCA","DMA","KNA","PRI"]},
    # Europa
    **{c: "Europa" for c in [
        "ESP","FRA","DEU","ITA","GBR","NLD","BEL","PRT","POL","AUT","CHE","SWE","DNK",
        "FIN","NOR","IRL","GRC","CZE","ROU","HUN","BGR","HRV","SVK","SVN","EST","LTU",
        "LVA","LUX","MLT","RUS","UKR","BLR","SRB","BIH","ALB","MKD","MDA","MNE","ISL",
        "CYP","AND","MCO","SMR","LIE","VAT","XKX"]},
    # Asia
    **{c: "Asia" for c in [
        "CHN","JPN","KOR","IND","IDN","TUR","SAU","IRN","ISR","ARE","PAK","BGD","VNM",
        "THA","MYS","PHL","SGP","HKG","TWN","MAC","IRQ","SYR","JOR","LBN","YEM","OMN",
        "KWT","QAT","BHR","KAZ","UZB","KGZ","TJK","TKM","AFG","NPL","LKA","MMR","KHM",
        "LAO","MNG","PRK","BTN","MDV","BRN","PSE","TLS","GEO","ARM","AZE"]},
    # África
    **{c: "África" for c in [
        "ZAF","EGY","MAR","TUN","DZA","NGA","KEN","ETH","GHA","SEN","CIV","AGO","COD",
        "MOZ","TZA","UGA","CMR","MDG","LBY","SDN","ZWE","ZMB","MLI","BFA","NER","TCD",
        "SOM","GIN","RWA","BEN","TGO","SLE","LBR","MRT","NAM","BWA","GMB","GNB","GAB",
        "CAF","COG","ERI","DJI","SSD","BDI","CPV","COM","STP","SYC","SWZ","LSO","MWI",
        "MUS","GNQ"]},
    # Oceanía
    **{c: "Oceanía" for c in [
        "AUS","NZL","PNG","FJI","SLB","VUT","WSM","TON","KIR","MHL","PLW","FSM","NRU","TUV"]},
}

CONTINENT_COLORS: dict[str, str] = {
    "África": "#773bd8",
    "América": "#9e4643",
    "Asia": "#6bc285",
    "Europa": "#5780b7",
    "Oceanía": "#f2bc67",
    "Otros": "#2f5d74",
}


@st.cache_data
def anchor_firms_per_hs4(_signature: str, profiles: tuple[str, ...]) -> dict:
    """Per-HS4 count of distinct firms whose exporter_profile is in `profiles`.
    Used to make the anchor universe respond to the profile filter set on
    the Firmas y Rubros page — so 'No Exporta' firms unchecked upstream
    stop propping up HS4 downstream.
    Returns a plain dict[hs4 -> int] so it hashes cleanly under st.cache_data."""
    firm_ev, _ = load_firms_data(_signature)
    keep = set(profiles or DEFAULT_ANCHOR_PROFILES)
    filtered = firm_ev[firm_ev["exporter_profile"].astype(str).isin(keep)]
    return (
        filtered.drop_duplicates(["firm_id", "hs4"])
        .groupby("hs4")["firm_id"]
        .nunique()
        .to_dict()
    )


def _selected_anchor_profiles() -> tuple[str, ...]:
    """Read the user's exporter_profile selection from Page 2's sidebar
    (session_state) and return it as a canonicalized tuple. Falls back to
    the default (Habitual + Ocasional) if the user hasn't opened Page 2 yet.
    Intentionally drops empty selections so a stray reset doesn't produce an
    empty anchor universe."""
    sel = st.session_state.get("firms_sel_profile")
    if not sel:
        return DEFAULT_ANCHOR_PROFILES
    return tuple(sorted(sel))


@st.cache_data
def load_accessible_market(_signature: str = ""):
    df = pd.read_csv(_data("intermediate", "accessible_market_arg.csv"), dtype={"hs92": str})
    df["hs92"] = df["hs92"].str.zfill(4)
    df["continente"] = df["iso3_d"].map(_ISO3_TO_CONTINENT).fillna("Otros")
    return df


@st.cache_data
def load_competitors_bilateral(_signature: str = ""):
    """Per-(HS4, exporter, destination) export values into Argentina's
    accessible destinations only. BACI 2024. ~289k rows. Used to power
    the competitors treemap with an optional per-market filter."""
    df = pd.read_csv(
        _data("intermediate", "competitors_by_hs4.csv"),
        dtype={"hs92": str, "iso3_o": str, "iso3_d": str},
    )
    df["hs92"] = df["hs92"].str.zfill(4)
    return df



def page_inicio():
    _page_header(
        "Diversificación productiva — Córdoba",
        "Tablero exploratorio para la agenda de diversificación productiva "
        "de la Provincia de Córdoba, Argentina."
    )

    st.markdown("""
### Qué hace este tablero

Identifica **productos candidatos** a los que Córdoba podría diversificarse,
partiendo de los productos **HS4 (HS 1992)** donde la provincia ya tiene
presencia exportadora evidenciada por firmas reales. Usa como núcleo
metodológico la **proximidad en el espacio de productos** (Hidalgo, Hausmann
et al.), construida con datos BACI 2020–2024 y la implementación
[`ecomplexity`](https://github.com/cid-harvard/py-ecomplexity) del Growth Lab.
Debajo se detalla el enfoque, las fuentes de evidencia, el glosario y las
fórmulas.
    """)

    st.markdown("""
### Seis páginas

- **Inicio** (acá): contexto, glosario, fórmulas.
- **Exportaciones por producto**: composición 2023-2025 de las
  exportaciones cordobesas a nivel HS4, agrupadas por sector Atlas
  (fuente: Dirección de Estadística de la Provincia). Click en un HS4
  para ver las firmas identificadas que lo exportan.
- **Análisis de Proximidad**: el tablero interactivo principal con
  todos los filtros, el espacio de productos, el Sankey, la tabla de
  candidatos rankeados y el treemap. Basado en la lógica de anclas
  evidenciadas por firmas.
- **Oportunidades**: **NUEVO**. Ranking HS4 combinando factibilidad
  (proximidad al know-how cordobés) y atractivo (complejidad + mercado
  accesible). Réplica de la metodología Growth Lab / ARG_Dashboard_V2
  aplicada al caso Córdoba: complejidad y densidad del panel
  Córdoba+BACI, métricas de mercado heredadas de Argentina. Presets
  *Margen Intensivo* y *Margen Extensivo*.
- **Mercado Accesible por Producto**: para cada uno de los top-30
  candidatos del preset Recomendado, la composición geográfica del
  mercado accesible (países destino con sus importaciones).
- **Firmas y Rubros [Legacy]**: vista original firma→rubro INDEC con
  OPEX por rubro entero, conservada para comparación con la nueva
  vista por HS4.
    """)

    st.subheader("Qué es el análisis de proximidad anclada")
    st.markdown(r"""
La pregunta operativa de fondo es simple: **¿a qué productos podría
Córdoba diversificarse que sean realistas dada su base productiva
actual?** No cualquiera. Un análisis útil tiene que arrancar de lo que
la provincia efectivamente sabe hacer y buscar afuera productos
adyacentes.

**El punto de partida: las anclas.** Llamamos *anclas* al conjunto de
HS4 donde Córdoba ya tiene presencia exportadora evidenciada por firmas
reales (soja, autopartes, maní, cueros, medicamentos, maquinaria
agrícola, etc.). Capacidades vinculadas a empresas que hoy exportan.
Esos productos anclan el análisis: fijan un piso sobre el mapa de
posibilidades de diversificación.

**La medida de cercanía: proximidad en el espacio de productos.** El
espacio de productos (Hidalgo, Hausmann et al.) es un mapa empírico
donde dos productos están cerca si los mismos países tienden a
exportarlos con ventaja competitiva. Producir uno requiere capacidades
—conocimientos técnicos, redes de proveedores, logística, regulación—
que suelen ser útiles para el otro. Esas capacidades son **pegajosas**:
los países diversifican hacia productos cercanos, casi nunca hacia
saltos aleatorios. Los *rodamientos y ejes de transmisión* están cerca
de las *autopartes*; los *medicamentos formulados* están cerca de los
*principios activos farmacéuticos*. La proximidad convierte una
intuición sobre similitud productiva en un número entre 0 y 1.

**El resultado: candidatos anclados.** Para cada ancla de Córdoba
extraemos su top-1 % de productos más cercanos en el espacio global.
La unión de esos vecinos —descontando lo que Córdoba ya exporta— es el
conjunto de **candidatos**: HS4 que la provincia no exporta hoy pero
que quedan a distancia productiva razonable de su base actual. No es
una lista mágica de "esto va a funcionar"; es un filtrado sobre el
universo HS 1992 que descarta los productos que están fuera del rango
de capacidades reales.

**Rankeo por factibilidad y atractivo.** Los candidatos se ordenan
combinando dos dimensiones:

- **Factibilidad** — qué tan cerca está el candidato de la base
  productiva actual (proximidad), qué tan alineada está su demanda
  global con el patrón exportador argentino (DAI), y qué tan lejos
  viaja el producto en el comercio mundial (`distance_travelled`). Un
  producto es más factible cuando la provincia tiene más chances de
  producirlo competitivamente y cuando su mercado global es alcanzable.
- **Atractivo** — qué tan complejo es el producto (PCI, un indicador
  de sofisticación), qué tan grande es el mercado accesible para
  Argentina, y a qué tasa crece ese mercado.

Un dial estratégico en el sidebar permite priorizar factibilidad
(diversificaciones cercanas, casi seguras) o atractivo (saltos más
ambiciosos hacia productos complejos y de mercados grandes). Los pesos
internos de cada dimensión también son configurables.

**Qué no es este análisis.** No estima costos de entrada, no evalúa
políticas específicas, no proyecta volúmenes de exportación. Es un
tamiz de plausibilidad sobre 1.243 productos posibles: te da un punto
de partida ordenado para conversaciones con firmas, cámaras y
policymakers, no una recomendación cerrada.
    """)

    st.subheader("De dónde salen los HS4 evidenciados")
    st.markdown(r"""
Los **462 HS4 evidenciados** son productos que sabemos que Córdoba
exporta porque hay al menos una firma real que los declaró o que
verificamos exportándolos. Se combinan dos fuentes independientes:

| Fuente | Qué es | Firmas | HS4 |
|---|---|---|---|
| **Códigos aduaneros declarados** | Códigos NCM que cada firma cargó en su ficha del registro Procórdoba (sección "Oferta exportable"). Es la lista oficial que la firma reporta como su canasta exportable. Para el set de anclas contamos sólo firmas que efectivamente exportan (perfil Habitual u Ocasional); las aspirantes ("No Exporta / Próxima a Exportar") aparecen en la tabla de Firmas y Rubros con un filtro dedicado. | 857 | 458 |
| **Curado manual** | Búsqueda dirigida de los principales exportadores de Córdoba que **no** figuraban en el registro Procórdoba — plantas industriales grandes como Renault, Stellantis, VW, Iveco, Quilmes, Petroquímica Río Tercero, Atanor — más 52 firmas del registro donde el analista verificó a mano la atribución producto→HS4 con URL fuente. | 66 | 95 |
| **Unión** | (después de deduplicar) | 874 | **462** |

El código aduanero declarado es la evidencia más fuerte: no interpretamos
descripciones de texto, tomamos el HS4 directamente de los primeros 4
dígitos del NCM que la firma reportó a la aduana. El curado manual
complementa esto con los exportadores grandes que faltaban del registro.

**Firmas multiproducto — por qué una firma puede aparecer varias veces.**
La tabla en *Firmas y Rubros* muestra pares **firma × HS4**, no firmas.
Una firma que exporta varios productos aparece con una fila por HS4. Por
ejemplo, ESNAOLA (Grupo Dulcor) figura tres veces: HS 2007 (mermeladas)
y HS 1806 (chocolate) desde el curado manual, y HS 1901 (extractos de
malta) desde el NCM declarado. Los HS4 que existen en las dos capas
para la misma firma se deduplican y el analista humano gana; los HS4
adicionales de cada capa se conservan porque documentan porciones
distintas del portafolio.

### Base de exportaciones — ground truth desde DPE (2023-2025)

A partir de julio de 2026 la base cuantitativa del dashboard es el
archivo de la **Dirección de Estadística de la Provincia (DPE)** de
Córdoba: participación por NCM 8-dígitos en las exportaciones
provinciales acumuladas 2023-2025 en USD FOB. Ese archivo se agrega a
HS4 (primeros 4 dígitos del NCM) y se multiplica por el total OPEX
acumulado del período, obteniendo **exportaciones promedio anuales en
USD por HS4**. Cobertura: 592 HS4 con exportaciones positivas, top 30 =
93 % del total, top 100 = 99 %.

El set de **anclas** hoy es el conjunto de esos 592 HS4 con
exportaciones reales, restringido por dos sliders del sidebar en la
página *Análisis de Proximidad*: umbral de exportación promedio anual
(USD por HS4) y mínimo número de firmas identificadas. Un HS4 que no
entra al set ancla puede reaparecer como candidato marcado como
*posible ancla*. Los ~80 HS4 evidenciados por firmas pero sin
exportaciones DPE registradas quedan en el registro pero **no
participan** del análisis de diversificación.
    """)

    st.subheader("Glosario")
    st.markdown(r"""
| Variable | Significado |
|---|---|
| **HS4** | Sistema Armonizado a 4 dígitos, revisión 1992 (convención Atlas / Growth Lab). |
| **Ancla** | HS4 con **exportaciones reales de Córdoba** en 2023-2025 según DPE. 592 HS4 en total; el set ancla activo se restringe con los sliders exportación promedio anual y # firmas del sidebar en *Análisis de Proximidad*. |
| **Candidato** | HS4 que aparece en el top-1% de proximidad de al menos un ancla, **o** un HS4 exportado por Córdoba cuya exportación promedio anual cayó por debajo del umbral (se flaguea como *posible ancla*). |
| **Exportación promedio anual (USD)** | Valor exportado por Córdoba en el HS4 específico, promedio anual 2023-2025. Fuente: DPE (Dirección de Estadística de la Provincia), archivo participación NCM 2023-2025 × total OPEX acumulado. El slider correspondiente en el sidebar filtra el set de anclas. |
| **OPEX (legacy)** | Exportaciones de Córdoba por rubro INDEC (CCOD_RUBRO), promedio 2023-2025. Ya no filtra el análisis principal — reemplazado por la exportación por HS4 desde DPE. Se conserva en la página *Firmas y Rubros [Legacy]* para comparación. |
| **Rubro (legacy)** | "Grandes Rubros / Capítulos" de INDEC (clasificación ICA); 100 rubros en el panel OPEX. Ya no es la base cuantitativa del análisis — el nuevo pipeline usa HS4 directamente. |
| **Proximidad** | Probabilidad condicional `min(P(p₁\|p₂), P(p₂\|p₁))` de que un país exporte ambos productos con RCA, suavizada con `rca / (rca + 1)`. Está en [0, 1]. |
| **PCI** *(Product Complexity Index)* | Sofisticación productiva implícita de un HS4 — más alto = más complejo. |
| **DAI** *(Índice de alineación de demanda)* | Qué tanto la demanda externa por un producto se alinea con la canasta exportadora del país. Ver fórmula abajo. |
| **Distancia recorrida** *(distance_travelled)* | Distancia geográfica promedio (km) que recorre cada HS4 a nivel global, ponderada por valor exportado en cada par bilateral. **Atributo del producto** — no depende del país exportador. Ver fórmula abajo. |
| **Mercado accesible** *(accessible_market_size)* | Suma de las importaciones mundiales del producto por parte de los destinos que están dentro de la distancia recorrida del producto, o que ya reciben flujo grande desde el origen. Ver fórmula abajo. |
| **Factibilidad** | Promedio ponderado del DAI, del percentil de distancia recorrida y del número normalizado de anclas del candidato. Todos en [0, 1]. |
| **Atractivo** | Promedio ponderado del PCI, del tamaño del mercado accesible y del crecimiento a 5 años del mercado accesible. |
| **Puntaje combinado** | `(1 − balance) · factibilidad + balance · atractivo`, donde `balance` es el dial estratégico del sidebar. |
| **Posible ancla** | Dummy 1/0: el candidato pertenece al set DPE de 592 HS4 con exportación real, pero su exportación promedio anual no llegó al umbral del slider. |
| **Anclas del candidato** | HS4 (separados por ·) de las anclas que tienen al candidato en su top-1% de proximidad. |
    """)

    st.subheader("Fórmulas — variables clave")

    st.markdown("**Distancia recorrida** — atributo del producto")
    st.markdown(r"""
Distancia geográfica promedio que recorre el producto $p$ a nivel global,
ponderada por el valor exportado en cada par bilateral. **Se calcula sobre
el panel bilateral BACI 2020-2024 (promediado)** y es un atributo del
producto: no depende del país exportador de origen. Fuente:
`data_processing.ipynb` (celda 7).
    """)
    st.latex(r"""
\mathrm{DistanceTravelled}_p \;=\; \frac{\sum_{(c,c')} d_{c,c'} \cdot X_{c,c',p}}{\sum_{(c,c')} X_{c,c',p}}
""")
    st.markdown(r"""
donde $X_{c,c',p}$ son las exportaciones bilaterales del producto $p$ del
país $c$ al país $c'$ y $d_{c,c'}$ la distancia geográfica entre ambos.
**Mayor distancia = el producto viaja lejos globalmente = es un bien
tradeable**, y en este tablero se interpreta como mayor factibilidad
para Córdoba.
    """)

    st.markdown("**Mercado accesible**")
    st.markdown(r"""
Suma de las importaciones mundiales del producto $p$ por parte de los
destinos $c'$ que satisfacen al menos una de dos condiciones — cercanía
respecto de la distancia típica del producto, **o** flujo existente grande
desde el origen $c$:
    """)
    st.latex(r"""
\mathrm{AccessibleMarket}_{c,p} \;=\; \sum_{c' \in \mathcal{A}_{c,p}} M_{c',p}
""")
    st.latex(r"""
\mathcal{A}_{c,p} \;=\; \bigl\{\, c' : d_{c,c'} \le \mathrm{DistanceTravelled}_p \;\;\lor\;\; X_{c,c',p} \ge 100\,\mathrm{M\,USD} \,\bigr\}
""")
    st.markdown(r"""
donde $M_{c',p} = \sum_c X_{c,c',p}$ son las importaciones totales del
producto $p$ por parte del país $c'$, y $X_{c,c',p}$ son las exportaciones
bilaterales del origen $c$ al destino $c'$. **El umbral es binario, no un
decay continuo**: el destino entra al conjunto accesible sí o no, y su
importación total se agrega en su totalidad. El **crecimiento a 5 años**
del mercado accesible se calcula como CAGR entre 2020 y 2024:
    """)
    st.latex(r"""
\mathrm{AccessibleMarketGrowth5y}_p \;=\; \left(\frac{\mathrm{AccessibleMarket}_{p,2024}}{\mathrm{AccessibleMarket}_{p,2020}}\right)^{1/5} - 1
""")

    st.markdown("**DAI (Índice de Alineación de la Demanda)**")
    st.latex(r"""
\mathrm{DAI}_{z,i} \;=\; \sum_{y} C_{z,y} \, \omega_{i,y}
""")
    st.latex(r"""
C_{z,y} \;=\; \frac{X_{z,y} / M_y}{X_z / WT}
""")
    st.latex(r"""
\omega_{i,y} \;=\; \frac{M_{i,y}}{\sum_{y'} M_{i,y'}}
""")
    st.markdown(r"""
- `z`: exportador (Argentina en este tablero), `i`: producto, `y`: mercado socio.
- `C_{z,y}` mide **afinidad comercial revelada**: compara la participación
  de Argentina en las importaciones totales del mercado `y` con la
  participación global de Argentina en el comercio mundial.
- `ω_{i,y}` mide el **peso de demanda específico del producto**: la
  proporción de las importaciones mundiales del producto `i` que compra
  el mercado `y`.
- **Interpretación**: el DAI es un promedio ponderado por demanda de las
  afinidades comerciales de Argentina. Valores mayores que 1 significan
  que la demanda del producto `i` se concentra en mercados donde Argentina
  tiene una presencia importadora relativamente superior a su peso global;
  valores menores que 1 significan que la demanda se concentra donde
  Argentina tiene una presencia relativamente débil.
- **Ventana temporal**: todos los flujos ($X_{z,y}$, $M_{i,y}$, $X_z$, $WT$)
  corresponden a **BACI 2024** exclusivamente — no es un promedio de años,
  a diferencia de la distancia recorrida.
- **Set de comparación para el percentil**: `dai_percentile` no es un
  percentil global sobre ~200 países — es un **benchmark competitivo**.
  Argentina se rankea contra los **top-30 exportadores** del mismo producto
  (por valor exportado en BACI 2024) + Argentina, dedupeado, y el ranking
  se expresa como percentil sobre ese set. La pregunta que responde es:
  *"¿Qué tan bien alineada está la red comercial de Argentina para este
  producto frente a sus principales competidores globales?"*. Incluir a
  todos los países diluiría la señal con exportadores marginales o nulos.
  Fuente: `data_processing.ipynb` (celda 13).
    """)

    st.caption(
        "Fuentes: BACI HS92 (CEPII) 2020–2024 · Growth Lab Atlas · "
        "registro `exportadoresdecordoba.com` · panel OPEX provincial."
    )


@st.cache_data
def load_hs4_sector_map(_signature: str = ""):
    """HS4 → sector (Atlas sector classification) used to colour the
    page-3 OPEX treemap. Sourced from cordoba_anchored_proximity.csv —
    that file already attaches `anchor_sector` and `candidate_sector` to
    each HS4 (product_hs92.csv has no sector column)."""
    df = pd.read_csv(
        _data("output", "cordoba_anchored_proximity.csv"),
        dtype={"anchor_hs4": str, "candidate_hs4": str},
        usecols=["anchor_hs4", "anchor_sector", "candidate_hs4", "candidate_sector"],
    )
    cand = dict(zip(df["candidate_hs4"].astype(str).str.zfill(4), df["candidate_sector"].astype(str)))
    anch = dict(zip(df["anchor_hs4"].astype(str).str.zfill(4), df["anchor_sector"].astype(str)))
    # anchor side wins on conflict (consistent values, but anchor coverage is
    # what the firms table joins to).
    return {**cand, **anch}


def page_firmas():
    _page_header(
        "Firmas → anclas",
        "Firmas de Córdoba con su atribución HS4. Dos fuentes: "
        "**código NCM declarado** en la ficha del registro Procórdoba "
        "(la evidencia más fuerte — código aduanero oficial) y "
        "**curado manual** de los grandes exportadores que no figuran en "
        "el registro más verificaciones a mano con URL fuente."
    )

    firm_ev, opex = load_firms_data(_data_signature() if "_data_signature" in globals() else "")

    merged = firm_ev.merge(
        opex[["CCOD_RUBRO", "DESCRIP_RUBRO", "2023_2025_avg", "2024"]].rename(
            columns={
                "DESCRIP_RUBRO": "rubro_indec_nombre_opex",
                "2023_2025_avg": "opex_avg_2023_2025_usd",
                "2024": "opex_2024_usd",
            }
        ),
        left_on="rubro_indec",
        right_on="CCOD_RUBRO",
        how="left",
    )

    merged["rubro_indec_nombre_final"] = merged["rubro_indec_nombre_opex"].fillna(merged["rubro_indec_nombre"])

    merged["hs4_es"] = merged["hs4"].astype(str).str.zfill(4).map(
        lambda h: SPANISH_OVERRIDES.get(h, "")
    )
    merged["hs4_label"] = merged["hs4"].astype(str).str.zfill(4) + (
        " - " + merged["hs4_es"]
    ).where(merged["hs4_es"] != "", "")
    merged["opex_avg_m"] = merged["opex_avg_2023_2025_usd"] / 1e6
    merged["opex_2024_m"] = merged["opex_2024_usd"] / 1e6

    hs4_options = sorted(merged["hs4_label"].dropna().astype(str).unique().tolist())
    confidence_options = sorted(merged["confidence"].dropna().astype(str).unique().tolist())
    # Order profiles from real exporters to aspirationals so the default view
    # leads with hard evidence.
    _profile_order = ["Exportadora Habitual", "Exportadora Ocasional", "No Exporta / Próxima a Exportar"]
    _profiles_present = merged["exporter_profile"].fillna("").astype(str).unique().tolist()
    profile_options = [p for p in _profile_order if p in _profiles_present] + \
                      sorted(p for p in _profiles_present if p and p not in _profile_order)
    # Default selection: exclude aspirationals so numbers match the diversification
    # analysis on the Proximidad page (which only counts real exporters).
    profile_default = [p for p in profile_options if p != "No Exporta / Próxima a Exportar"]

    def _reset_firmas_filters():
        st.session_state["firms_sel_hs4"] = hs4_options
        st.session_state["firms_sel_conf"] = confidence_options
        st.session_state["firms_sel_profile"] = profile_default
        # Clear the treemap click selection so the rubro/sector filter also resets.
        st.session_state.pop("firms_treemap_select", None)

    with st.sidebar:
        st.header("Filtros — Firmas")
        st.button(
            "Restablecer filtros",
            on_click=_reset_firmas_filters,
            use_container_width=True,
            help="Limpia HS4 ancla, Confianza, Perfil exportador y la selección del treemap.",
        )
        sel_profile = st.multiselect(
            "Perfil exportador",
            options=profile_options,
            default=profile_default,
            key="firms_sel_profile",
            help=(
                "Perfil declarado por la firma en el registro Procórdoba. Las "
                "firmas 'curated' (grandes exportadores fuera del registro) se "
                "asignan por defecto a **Exportadora Habitual**. Por defecto se "
                "excluye 'No Exporta / Próxima a Exportar' para que los conteos "
                "coincidan con el set de anclas."
            ),
        )
        sel_hs4 = st.multiselect(
            "HS4 ancla", options=hs4_options, default=hs4_options, key="firms_sel_hs4"
        )
        sel_conf = st.multiselect(
            "Confianza",
            options=confidence_options,
            default=confidence_options,
            key="firms_sel_conf",
        )
        st.caption(
            "Para filtrar por **rubro INDEC** o **capa de evidencia**, clickeá "
            "una baldosa del treemap. Click en el fondo o ESC para limpiar."
        )

    f = merged.copy()
    if sel_profile:
        f = f[f["exporter_profile"].astype(str).isin(sel_profile)]
    if sel_hs4:
        f = f[f["hs4_label"].isin(sel_hs4)]
    if sel_conf:
        f = f[f["confidence"].astype(str).isin(sel_conf)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Firmas únicas", f["firm_id"].nunique())
    c2.metric("Filas (pares firma-HS4)", len(f))
    c3.metric("HS4 ancla cubiertos", f["hs4"].nunique())
    c4.metric("Rubros INDEC", f["rubro_indec_nombre_final"].nunique())

    # ----- OPEX treemap por rubro INDEC (clona el formato de página 2) -----
    # The treemap doubles as a filter: clicking a tile narrows the table to
    # the firms in that rubro (leaf) or sector (parent). Empty = show all.
    visible_rubros = set(f["rubro_indec"].dropna().astype(str).str.strip().unique())
    opex_tm = opex.copy()
    opex_tm["CCOD_RUBRO"] = opex_tm["CCOD_RUBRO"].astype(str).str.strip()
    opex_tm = opex_tm[opex_tm["CCOD_RUBRO"].isin(visible_rubros)]
    opex_tm = opex_tm.dropna(subset=["2023_2025_avg"])
    opex_tm = opex_tm[opex_tm["2023_2025_avg"] > 0].copy()
    if len(opex_tm):
        # Per HS4 → sector. Then per (rubro, sector) split each rubro's OPEX
        # proportionally to its share of HS4s in that sector. Rubros that span
        # multiple sectors (eg the confidentials 39899) emit multiple tiles —
        # one per sector — so the treemap shows the true sector composition.
        hs4_to_sector = load_hs4_sector_map(_data_signature() if "_data_signature" in globals() else "")
        f_with_sector = f.assign(sector=f["hs4"].astype(str).str.zfill(4).map(hs4_to_sector).fillna("Other"))

        n_hs4_by_pair = (
            f_with_sector.drop_duplicates(["rubro_indec", "hs4", "sector"])
            .groupby(["rubro_indec", "sector"])["hs4"].nunique()
            .reset_index(name="n_hs4_in_sector")
        )
        n_hs4_per_rubro = (
            f_with_sector.drop_duplicates(["rubro_indec", "hs4"])
            .groupby("rubro_indec")["hs4"].nunique()
            .reset_index(name="total_hs4")
        )
        shares = n_hs4_by_pair.merge(n_hs4_per_rubro, on="rubro_indec")
        shares["share"] = shares["n_hs4_in_sector"] / shares["total_hs4"]

        opex_split = shares.merge(
            opex_tm[["CCOD_RUBRO", "DESCRIP_RUBRO", "2023_2025_avg"]],
            left_on="rubro_indec", right_on="CCOD_RUBRO", how="inner",
        )
        opex_split["opex_avg_m"] = opex_split["share"] * opex_split["2023_2025_avg"] / 1e6

        # Confidentials (codes ending in '899') and resto (ending in 'Z')
        # are intentionally ambiguous — INDEC doesn't disclose internal
        # composition, so any sector split we'd compute is a guess. Collapse
        # those rubros to a single 'Other' tile with the full OPEX, instead
        # of showing potentially misleading splits.
        def _is_mixed(code: str) -> bool:
            s = str(code).strip()
            return s.endswith("899") or s.endswith("Z")

        mixed_mask = opex_split["CCOD_RUBRO"].apply(_is_mixed)
        clean_part = opex_split[~mixed_mask].copy()
        mixed_part = opex_split[mixed_mask].copy()
        if len(mixed_part):
            mixed_collapsed = (
                mixed_part.drop_duplicates("CCOD_RUBRO")[
                    ["rubro_indec", "CCOD_RUBRO", "DESCRIP_RUBRO", "2023_2025_avg", "total_hs4"]
                ]
                .copy()
            )
            mixed_collapsed["sector"] = "Other"
            mixed_collapsed["n_hs4_in_sector"] = mixed_collapsed["total_hs4"]
            mixed_collapsed["share"] = 1.0
            mixed_collapsed["opex_avg_m"] = mixed_collapsed["2023_2025_avg"] / 1e6
        else:
            mixed_collapsed = mixed_part

        opex_split = pd.concat([clean_part, mixed_collapsed], ignore_index=True)
        opex_split = opex_split[opex_split["opex_avg_m"] > 0].copy()
        opex_split["rubro_label"] = opex_split["CCOD_RUBRO"] + " - " + opex_split["DESCRIP_RUBRO"].astype(str)
        opex_split["rubro_label_wrapped"] = opex_split["rubro_label"].map(_wrap_label)
        # Set of rubros we won't split by sector — used when interpreting clicks
        mixed_rubro_set = set(opex_split.loc[opex_split["sector"] == "Other", "CCOD_RUBRO"].astype(str))

        st.metric(
            label="OPEX total mostrado (USD millones)",
            value=f"{opex_split['opex_avg_m'].sum():,.1f}",
        )

        fig_tm = px.treemap(
            opex_split,
            path=["sector", "rubro_label_wrapped"],
            values="opex_avg_m",
            color="sector",
            color_discrete_map=SECTOR_COLORS,
            hover_data={
                "opex_avg_m": ":.1f",
                "CCOD_RUBRO": True,
                "DESCRIP_RUBRO": True,
                "n_hs4_in_sector": True,
                "total_hs4": True,
                "share": ":.0%",
                "sector": False,
                "rubro_label_wrapped": False,
            },
            title=(
                f"Exportaciones promedio por rubro INDEC "
                f"(n = {opex_tm['CCOD_RUBRO'].nunique()} rubros | "
                f"total = {opex_split['opex_avg_m'].sum():,.1f} USD M · "
                f"promedio 2023-2025) | tamaño = OPEX"
            ),
        )
        fig_tm.update_traces(
            textinfo="label",
            textfont=dict(size=18, color="#ffffff"),
            marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
        )
        fig_tm.update_layout(
            margin=dict(t=60, l=10, r=10, b=95),
            height=720,
            legend=dict(
                orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                title_text="Sector",
            ),
        )
        tm_state = st.plotly_chart(
            fig_tm,
            use_container_width=True,
            on_select="rerun",
            key="firms_treemap_select",
        )
        st.caption(
            "Nota: cuando el rubro INDEC tiene atribución firme, su OPEX se "
            "divide entre los sectores presentes en proporción a los HS4 que "
            "cada sector aporta. Los rubros confidenciales (códigos terminados "
            "en `899`) y los rubros 'resto' (terminados en `Z`) no se dividen "
            "— la composición interna no está publicada por INDEC — y van "
            "enteros a la categoría **Other**."
        )

        # Decode selection — leaves are (sector, rubro) pairs now.
        selected_rubros: set = set()
        selected_sectors: set = set()
        selected_pairs: set = set()  # (rubro_code, sector) for leaf clicks
        try:
            pts = tm_state.selection.points if hasattr(tm_state, "selection") else (tm_state.get("selection", {}) or {}).get("points", [])
        except Exception:
            pts = []
        # Build reverse lookup (wrapped_label, sector) → CCOD_RUBRO
        pair_lookup = {
            (str(r["rubro_label_wrapped"]), str(r["sector"])): str(r["CCOD_RUBRO"])
            for _, r in opex_split.iterrows()
        }
        for p in pts or []:
            try:
                pid = p.get("id", "") if isinstance(p, dict) else getattr(p, "id", "")
                label = p.get("label", "") if isinstance(p, dict) else getattr(p, "label", "")
                parent = p.get("parent", "") if isinstance(p, dict) else getattr(p, "parent", "")
            except Exception:
                pid, label, parent = "", "", ""
            if not pid:
                continue
            if "/" in pid:
                # leaf — parent is the sector
                rubro_code = pair_lookup.get((str(label), str(parent)))
                if rubro_code:
                    if rubro_code in mixed_rubro_set:
                        # Confidential/resto: filter by rubro only (the
                        # 'Other' sector tag is a UI label, not a real
                        # sector — firms in this rubro can have any sector).
                        selected_rubros.add(rubro_code)
                    else:
                        selected_pairs.add((rubro_code, str(parent)))
            else:
                selected_sectors.add(str(label))

        if selected_pairs or selected_sectors or selected_rubros:
            mask = pd.Series(False, index=f_with_sector.index)
            if selected_sectors:
                mask |= f_with_sector["sector"].astype(str).isin(selected_sectors)
            if selected_rubros:
                mask |= f_with_sector["rubro_indec"].astype(str).str.strip().isin(selected_rubros)
            if selected_pairs:
                pair_mask = pd.Series(False, index=f_with_sector.index)
                for rubro, sec in selected_pairs:
                    pair_mask |= (
                        (f_with_sector["rubro_indec"].astype(str).str.strip() == rubro)
                        & (f_with_sector["sector"].astype(str) == sec)
                    )
                mask |= pair_mask
            f = f_with_sector[mask].copy()
            parts = []
            if selected_sectors:
                parts.append("sectores: " + ", ".join(sorted(selected_sectors)))
            if selected_rubros:
                parts.append("rubros: " + ", ".join(sorted(selected_rubros)))
            if selected_pairs:
                parts.append("pares rubro/sector: " + ", ".join(f"{r}/{s}" for r, s in sorted(selected_pairs)))
            st.caption("Filtro activo del treemap — " + " · ".join(parts))
    else:
        st.info("Ningún rubro INDEC con OPEX > 0 en el filtro actual.")

    cols = [
        "firm_name", "razon_social", "hs4_label", "rubro_indec_nombre_final",
        "evidence_layer", "exporter_profile", "confidence",
        "opex_avg_m",
        "attribution_type", "evidence_text", "evidence_url", "source_url",
    ]
    display = f[[c for c in cols if c in f.columns]].copy()
    display = display.sort_values(
        ["evidence_layer", "rubro_indec_nombre_final", "opex_avg_m", "firm_name"],
        ascending=[True, True, False, True],
        na_position="last",
    )

    with st.expander("Diccionario de columnas — Tabla de firmas"):
        st.markdown(r"""
| Columna | Significado |
|---|---|
| **Firma (alias)** | Nombre comercial de la firma según el registro `exportadoresdecordoba.com`. |
| **Razón social** | Nombre legal de la firma. |
| **HS4 ancla** | HS4 (HS 1992) + nombre corto en español al que la firma está atribuida. Todos los HS4 aquí pertenecen al set de 462 HS4 evidenciados. |
| **Rubro INDEC** | Rubro CCOD_RUBRO (clasificación INDEC "Grandes Rubros / Capítulos") al que el HS4 fold-up en el panel OPEX provincial. |
| **Capa** | Fuente del vínculo firma↔HS4. **`declared-ncm`**: la firma cargó explícitamente un código NCM en su ficha del registro Procórdoba (sección "Oferta exportable"). Los primeros 4 dígitos del NCM son el HS4 — es el mismo código con el que la firma exporta ante Aduana, así que hay poca ambigüedad. **`curated`** (66 firmas): revisión manual — cubre los grandes exportadores que no están en el registro (Renault, Stellantis, VW, Iveco, Quilmes, plantas petroquímicas de Río Tercero…) más firmas del registro donde se verificó la atribución HS4 con URL fuente. |
| **Perfil exportador** | Perfil declarado por la firma en el registro Procórdoba. **`Exportadora Habitual`** — exporta regularmente. **`Exportadora Ocasional`** — exporta esporádicamente. **`No Exporta / Próxima a Exportar`** — figura en el registro con canasta exportable declarada pero no exporta actualmente (aspirante). Por defecto la vista excluye este último grupo para que los conteos coincidan con el set de anclas de la página Análisis de Proximidad. Las firmas `curated` se asignan a **Exportadora Habitual** por defecto. |
| **Confianza** | Nivel de certeza sobre el vínculo firma↔HS4. **`high`**: evidencia clara y explícita (código NCM oficialmente declarado, o análisis manual concluyente). **`medium`**: match plausible pero con ambigüedad. **`low`**: señal débil. La capa `declared-ncm` es siempre `high` (código aduanero). Para `curated` casi todo es `high` porque sólo se registran filas donde hay certeza. |
| **OPEX rubro (USD M, prom 2023-2025)** | Monto exportado por Córdoba en el rubro INDEC, promedio anual 2023-2025 en USD millones. Es del **rubro entero**, no de la firma individual — una firma en un rubro grande no representa necesariamente una porción grande del monto. |
| **Tipo de atribución** | Cómo el HS4 evidenciado en la firma fold-up al rubro INDEC. Ordenados de más a menos preciso: **`clean`** — el rubro mapea limpiamente a 1-2 HS4 (ej. `106B Maíz → HS 1005`), sin ambigüedad; **`named-aggregate`** — rubro nombrado que agrupa varios HS4 del mismo dominio (ej. `313BB Vehículos automóviles terrestres → HS 8702/8703/8704`); **`broad-chapter`** — rubro cubre un capítulo HS entero (ej. `312B Máq. eléctricas → HS 8501-8548`), el HS4 específico viene del lado firm; **`resto`** — rubro residual dentro de un grupo (ej. `107Z Resto semillas y frutos oleaginosos`), cubre HS4 no clasificados en categorías nombradas; **`confidential`** — rubro INDEC censurado por Ley 17.622 (códigos terminados en `899`): el monto agregado está publicado pero la composición interna no, así que el HS4 sólo puede establecerse desde el lado firm. |
| **Evidencia (texto)** | Justificación del vínculo firma↔HS4. En **`declared-ncm`**: el código NCM completo + descripción del producto tal como la firma lo declaró en su ficha del registro (ej. *"NCM 1601.00.00 — CHORIZO PARRILLERO · perfil: Habitual"*). En **`curated`**: texto humano explicando por qué se asignó ese HS4 (ej. *"Products CHORIZO DE CERDO, MORCILLA, SALCHICHA, SALAME → HS 1601 'sausages and similar products'"*). Sirve para auditar: si el texto no parece justificar el HS4, hay razón para dudar. |
| **Evidencia URL** | URL pública que evidencia el vínculo (notas de prensa, sitio corporativo, cámara, etc.). |
| **Página registro** | Link a la ficha de la firma en `exportadoresdecordoba.com`. |

**Nota sobre el filtro del treemap**: al clickear una baldosa de sector o
rubro en el treemap de arriba, la tabla se restringe a las firmas cuyo
sector/rubro coincida. Click en el fondo del treemap para limpiar.
        """)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "firm_name": st.column_config.TextColumn("Firma (alias)", width="medium"),
            "razon_social": st.column_config.TextColumn("Razón social", width="medium"),
            "hs4_label": st.column_config.TextColumn("HS4 ancla", width="medium"),
            "rubro_indec_nombre_final": st.column_config.TextColumn("Rubro INDEC", width="medium"),
            "evidence_layer": st.column_config.TextColumn(
                "Capa",
                help="`declared-ncm` = código NCM oficial declarado por la firma en el registro · `curated` = evidencia manual con URL",
            ),
            "exporter_profile": st.column_config.TextColumn(
                "Perfil exportador",
                help=(
                    "Perfil declarado por la firma en el registro Procórdoba: "
                    "**Exportadora Habitual**, **Exportadora Ocasional** o "
                    "**No Exporta / Próxima a Exportar**. Firmas 'curated' "
                    "(grandes exportadores fuera del registro) se marcan como "
                    "Exportadora Habitual por defecto."
                ),
            ),
            "confidence": st.column_config.TextColumn("Confianza"),
            "opex_avg_m": st.column_config.NumberColumn(
                "OPEX rubro (USD M, prom 2023-2025)", format="%.1f",
                help="Monto exportado por Córdoba en el rubro INDEC del lado de la firma, en millones de USD.",
            ),
            "attribution_type": st.column_config.TextColumn("Tipo de atribución"),
            "evidence_text": st.column_config.TextColumn("Evidencia (texto)", width="large"),
            "evidence_url": st.column_config.LinkColumn("Evidencia URL", display_text="↗", width="small"),
            "source_url": st.column_config.LinkColumn("Página registro", display_text="↗", width="small"),
        },
    )

    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Descargar tabla (CSV)",
        csv_bytes,
        "cordoba_firmas.csv",
        "text/csv",
    )

    with st.expander("Notas metodológicas"):
        st.markdown("""
- **`declared-ncm`** (1.587 firmas totales): la firma cargó códigos
  NCM específicos en su ficha del registro Procórdoba (sección "Oferta
  exportable"). El HS4 son los primeros 4 dígitos del NCM. Se muestran
  los tres perfiles con un filtro dedicado: **857** son exportadoras
  reales (Habitual u Ocasional) y **730** son aspirantes ("No Exporta
  / Próxima a Exportar"). Sólo las primeras entran al set de anclas.
- **`curated`** (66 firmas): revisión manual con URL fuente. Cubre
  grandes exportadores fuera del registro (plantas automotrices,
  cervecería, petroquímica, etc.) y verificaciones a mano. Asignadas
  a Exportadora Habitual por defecto.
- Curated tiene prioridad sobre declared-ncm en caso de conflicto.
- El monto OPEX corresponde al rubro INDEC entero (no a la firma).
        """)

def page_analisis():
    _page_header(
        "Córdoba — Análisis de proximidad anclada",
        "Explorá productos candidatos conectados por proximidad en el espacio "
        "de productos a los HS4 donde Córdoba tiene presencia evidenciada."
    )

    df, presence, umap, trade_2024, cluster_color, names = load_data()

    if df.empty:
        st.warning("No anchor proximity data available.")
        st.stop()

    # Cross-page filter: propagate the exporter_profile selection from the
    # Firmas y Rubros page into the anchor universe. Recompute per-HS4 firm
    # counts using only the selected profiles; presence rows with 0 filtered
    # firms fall out of the anchor set.
    _selected_profiles = _selected_anchor_profiles()
    _n_firms_dyn = anchor_firms_per_hs4("", _selected_profiles)
    presence = presence.copy()
    presence["n_firms"] = (
        presence["hs4"].astype(str).str.zfill(4).map(_n_firms_dyn).fillna(0).astype(int)
    )
    if set(_selected_profiles) != set(DEFAULT_ANCHOR_PROFILES):
        st.info(
            "🔎 Filtro cruzado activo: **Perfil exportador = "
            f"{', '.join(_selected_profiles)}** (heredado de la página "
            "*Firmas y Rubros*). El set de anclas y todos los paneles se "
            "recalculan con este filtro."
        )

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
    top_n_default = min(30, max(10, int(df["candidate_hs4"].nunique())))
    # Only hand-curated non-realistic candidates go through the multiselect
    # label mechanism. Natural-resource exclusion is now controlled by a
    # dedicated toggle (see 'Excluir recursos naturales' below) and applied
    # to BOTH anchors and candidates instead of only candidates.
    excluded_hs4_preset_codes = {x for x in PRESET_EXCLUDED_HS4}
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
    st.session_state.setdefault("c4_min_firmas_ancla", 1)
    st.session_state.setdefault("c4_accessible_market_min", 0.0)
    st.session_state.setdefault("c4_am_cagr_only", False)
    st.session_state.setdefault("c4_strategic_balance", 0.50)
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
    st.session_state.setdefault("c4_exclude_nat_resources", True)
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
        st.session_state["c4_exclude_nat_resources"] = True
        st.session_state["c4_proximity_rank_range"] = (1, min(100, max(1, proximity_rank_max)))

        if profile_name == "top_candidates":
            st.session_state["c4_strategic_balance"] = 0.50
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
            st.session_state["c4_exclude_nat_resources"] = True
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
        st.session_state["c4_exclude_nat_resources"] = True
        st.session_state["c4_proximity_rank_range"] = (1, min(100, max(1, proximity_rank_max)))


    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("Perfiles predefinidos")
        st.button(
            "Recomendado",
            on_click=_apply_profile,
            args=("top_candidates",),
            use_container_width=True,
        )

        st.header("Presencia (set de anclas)")
        opex_threshold = st.select_slider(
            "Umbral exportación promedio anual (USD)",
            options=OPEX_OPTIONS,
            value=st.session_state["c4_opex_threshold"],
            format_func=lambda v: OPEX_LABELS[v],
            key="c4_opex_threshold",
            help=(
                "Define el set de anclas (anchors) para Córdoba. Sólo los HS4 "
                "cuya exportación promedio anual 2023-2025 (Dirección de "
                "Estadística de la Provincia, promediada del acumulado 3 años) "
                "es ≥ a este umbral se consideran anclas."
            ),
        )
        _presence_max_firms = int(pd.to_numeric(presence.get("n_firms"), errors="coerce").fillna(0).max()) if "n_firms" in presence.columns else 1
        _presence_max_firms = max(_presence_max_firms, 1)
        min_firmas_ancla = st.slider(
            "Mínimo # de firmas por HS4",
            min_value=1,
            max_value=_presence_max_firms,
            value=min(int(st.session_state["c4_min_firmas_ancla"]), _presence_max_firms),
            step=1,
            key="c4_min_firmas_ancla",
            help=(
                "Refuerza el set de anclas: sólo entran al universo los HS4 con "
                "al menos esta cantidad de firmas evidenciando (declared-ncm + "
                "curated combinadas). Subir el mínimo elimina HS4 con evidencia "
                "frágil (una sola firma) y se propaga a Sankey, tabla, treemap "
                "y product space."
            ),
        )
        exclude_nat_resources = st.checkbox(
            "Excluir recursos naturales",
            value=bool(st.session_state.get("c4_exclude_nat_resources", True)),
            key="c4_exclude_nat_resources",
            help=(
                "Excluye del análisis (tanto anclas como candidatos) los "
                f"{len(NATURAL_RESOURCE_HS4)} HS4 de recursos naturales: "
                "capítulos 25 (sal, azufre, cementos, cal, piedras), "
                "26 (minerales, escorias), 27 (combustibles minerales, "
                "petróleo, gas) y códigos 7101–7103 y 7108 (piedras y "
                "metales preciosos). Estos productos son commodities de "
                "extracción — no representan diversificación productiva."
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
            "Factibilidad (100%) = 0 | Atractivo (100%) = 1",
            0.0, 1.0,
            float(st.session_state["c4_strategic_balance"]),
            0.05,
            key="c4_strategic_balance",
            help=(
                "Score combinado = (1 − valor)·Factibilidad + valor·Atractivo. "
                "0 = sólo factibilidad; 1 = sólo atractivo."
            ),
        )

        st.header("Pesos de los componentes")
        st.caption("Factibilidad: DAI + distancia + # de anclas. Atractivo: PCI + tamaño + crecimiento (sin COG).")
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
        with st.expander("Componentes de atractivo", expanded=True):
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
    # 1. Derive anchor universe from export-value threshold + apply filters
    # ---------------------------------------------------------------------------
    # evidenced_set is the DPE-ground-truth HS4 universe (592 HS4). The anchor
    # universe is the subset whose average annual export value clears the
    # threshold. HS4 that fall below the threshold can resurface as candidates
    # of the surviving anchors — they're flagged `posible_ancla = 1`.
    evidenced_set = set(presence["hs4"].astype(str).str.zfill(4))
    _n_firms_series = pd.to_numeric(presence.get("n_firms"), errors="coerce").fillna(0)
    _nat_res_set = set(NATURAL_RESOURCE_HS4) if exclude_nat_resources else set()
    anchor_universe = set(
        presence.loc[
            (presence["usd_promedio_anual"].fillna(0) >= opex_threshold)
            & (_n_firms_series >= min_firmas_ancla)
            & (~presence["hs4"].astype(str).str.zfill(4).isin(_nat_res_set)),
            "hs4",
        ]
    )
    flt = df[
        df["anchor_hs4"].isin(anchor_universe)
        & ~df["candidate_hs4"].isin(anchor_universe)
    ].copy()
    if _nat_res_set:
        # Also drop natural-resource candidates on the right-hand side.
        flt = flt[~flt["candidate_hs4"].astype(str).str.zfill(4).isin(_nat_res_set)]
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
            anchors=("anchor_hs4", lambda s: " · ".join(sorted(set(s.astype(str).str.zfill(4))))),
            posible_ancla=("posible_ancla", "max"),
        )
    )

    # Córdoba's own export value for the candidate HS4 (DPE promedio anual
    # 2023-2025). Non-zero only for `posible ancla` candidates — pure
    # candidates have no evidenced presence, so the value is 0.
    _cordoba_export_usd = dict(
        zip(
            presence["hs4"].astype(str).str.zfill(4),
            pd.to_numeric(presence["usd_promedio_anual"], errors="coerce").fillna(0.0),
        )
    )
    candidate_scores["cordoba_export_usd_m"] = (
        candidate_scores["candidate_hs4"]
        .astype(str)
        .str.zfill(4)
        .map(_cordoba_export_usd)
        .fillna(0.0)
        / 1e6
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
    # Rubro INDEC + Match lookup (Directo / Residual) per anchor HS4
    _presence_indexed = presence.copy()
    _presence_indexed["hs4"] = _presence_indexed["hs4"].astype(str).str.zfill(4)
    _presence_indexed["primary_ccod_rubro"] = _presence_indexed["primary_ccod_rubro"].astype(str).str.strip()
    rubro_code_lookup = dict(zip(_presence_indexed["hs4"], _presence_indexed["primary_ccod_rubro"]))
    rubro_name_lookup = dict(zip(_presence_indexed["hs4"], _presence_indexed["primary_rubro_name"].astype(str)))

    _attr_map_hover_df = pd.read_csv(
        _data("output", "05_unified_hs4_presence.csv"),
        dtype={"hs4": str},
        usecols=["hs4", "attribution_type"],
    )
    _attr_map_hover = dict(zip(
        _attr_map_hover_df["hs4"].astype(str).str.zfill(4),
        _attr_map_hover_df["attribution_type"].astype(str),
    ))
    def _match_type(hs4: str) -> str:
        return "Residual" if _attr_map_hover.get(hs4, "") == "confidential" else "Directo"

    def _anchor_hover(h: str, name: str) -> str:
        ccod = rubro_code_lookup.get(h, "")
        rname = rubro_name_lookup.get(h, "")
        lines = [f"<b>HS {h}</b> · {name}", "Ancla"]
        if ccod:
            lines.append(f"Rubro INDEC: {ccod} — {rname}")
            lines.append(f"Match: {_match_type(h)}")
        return "<br>".join(lines)

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
    # Background: HS4 NOT in the current anchor set — light grey
    bg = umap_plot[~umap_plot["in_anchor"]]
    fig_ps.add_trace(go.Scatter(
        x=bg["product_space_x"], y=bg["product_space_y"], mode="markers",
        marker=dict(size=bg["radius"], color="#eeeef2",
                    line=dict(width=0.4, color="#d0d0d6")),
        text=[f"HS {h} · {n} · no es ancla" for h, n in zip(bg["product_hs92_code"], bg["product_name_es"])],
        hoverinfo="text", name="Sin evidencia (no ancla)", showlegend=True,
    ))
    # Anchors split into one trace per cluster so Plotly renders an
    # interactive legend below the plot.
    ank = umap_plot[umap_plot["in_anchor"]].copy()
    ank_by_cluster = ank.groupby("cluster_std", sort=True)
    for cluster_name, group in ank_by_cluster:
        display_name = CLUSTER_ES_LABEL.get(cluster_name, cluster_name) if cluster_name else "Otros"
        color = cluster_color.get(cluster_name, "#e0e0e3") if cluster_name else "#e0e0e3"
        fig_ps.add_trace(go.Scatter(
            x=group["product_space_x"], y=group["product_space_y"], mode="markers",
            marker=dict(size=group["radius"], color=color,
                        line=dict(width=0.6, color="#ffffff"), opacity=0.92),
            text=[_anchor_hover(h, n) for h, n in zip(group["product_hs92_code"], group["product_name_es"])],
            hoverinfo="text", name=display_name, showlegend=True,
        ))
    fig_ps.update_layout(
        height=680, margin=dict(l=20, r=20, t=20, b=110),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="rgba(16,24,44,0.95)", font_color="white"),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.02,
            xanchor="center", x=0.5,
            title_text="Sector del espacio de productos",
            title_font=dict(size=13),
            font=dict(size=12),
            itemsizing="constant",
        ),
    )
    st.plotly_chart(fig_ps, use_container_width=True)

    # ---------------------------------------------------------------------------
    # 4b. HS4 anclas del set actual — tabla vinculada al umbral OPEX
    # ---------------------------------------------------------------------------
    st.subheader("HS4 anclas del set actual")
    st.caption(
        f"HS4 con presencia evidenciada cuya OPEX supera el umbral seleccionado "
        f"({OPEX_LABELS[opex_threshold]}) — {len(anchor_universe)} códigos de un total de {len(presence)}."
    )
    _hs4_sector = load_hs4_sector_map(_data_signature() if "_data_signature" in globals() else "")
    _anchor_tbl = (
        presence[presence["hs4"].isin(anchor_universe)]
        .assign(
            hs4=lambda d: d["hs4"].astype(str).str.zfill(4),
        )
        .copy()
    )
    _anchor_tbl["Producto"] = _anchor_tbl["hs4"].map(lambda h: SPANISH_OVERRIDES.get(h, ""))
    _anchor_tbl["Sector"] = _anchor_tbl["hs4"].map(lambda h: _hs4_sector.get(h, "Other"))
    _anchor_tbl["Exportación promedio anual (USD M)"] = _anchor_tbl["usd_promedio_anual"] / 1e6
    # Match = 'Residual' cuando la atribución OPEX del HS4 es a un rubro
    # confidencial (INDEC Ley 17.622, códigos terminados en 899); 'Directo'
    # cuando el rubro es clean / named-aggregate / broad-chapter / resto.
    # Fuente: attribution_type en 05_unified_hs4_presence.csv, no el
    # primary_ccod_rubro de la presence-file (que muestra el rubro nombrado
    # equivalente aún cuando la data cae en un rubro confidencial).
    _attribution = pd.read_csv(_data("output", "05_unified_hs4_presence.csv"), dtype={"hs4": str}, usecols=["hs4","attribution_type"])
    _attribution["hs4"] = _attribution["hs4"].str.zfill(4)
    _attr_map = dict(zip(_attribution["hs4"], _attribution["attribution_type"].astype(str)))
    _anchor_tbl["_attr"] = _anchor_tbl["hs4"].astype(str).str.zfill(4).map(_attr_map).fillna("")
    _anchor_tbl["Match"] = _anchor_tbl["_attr"].apply(
        lambda a: "Residual" if a == "confidential" else "Directo"
    )
    _anchor_tbl = _anchor_tbl.drop(columns=["_attr"])
    _anchor_tbl = _anchor_tbl.rename(columns={
        "hs4": "HS4",
        "primary_ccod_rubro": "CCOD_RUBRO",
        "primary_rubro_name": "Rubro INDEC",
        "n_firms": "# firmas",
        "evidencing_firms_sample": "Firmas ejemplo",
    }).sort_values("Exportación promedio anual (USD M)", ascending=False)

    if min_firmas_ancla > 1:
        st.caption(
            f"→ Mostrando **{len(_anchor_tbl)}** HS4 con ≥ {min_firmas_ancla} firmas "
            f"(controlado por el slider en el sidebar)."
        )

    with st.expander("Diccionario de columnas — HS4 anclas"):
        st.markdown(r"""
| Columna | Significado |
|---|---|
| **HS4** | Código HS4 (HS 1992) del ancla. |
| **Producto** | Nombre corto del HS4 en español (curado). |
| **Sector** | Sector Atlas / Growth Lab del HS4. |
| **CCOD_RUBRO** | Rubro INDEC (código) al que el HS4 fold-up en el panel OPEX. |
| **Rubro INDEC** | Nombre del rubro INDEC. |
| **Match** | Tipo de correspondencia entre el HS4 y su rubro INDEC. **`Directo`**: el rubro es específico y publicado — clean (1-2 HS4), named-aggregate, broad-chapter o resto. **`Residual`**: el rubro es confidencial INDEC (código terminado en `899`); el HS4 sólo puede establecerse desde el lado firm porque la composición interna no está publicada por Ley 17.622. |
| **Exportación promedio anual (USD M)** | Monto exportado por Córdoba en **este HS4 específico**, promedio anual 2023-2025 (USD millones). Fuente: Dirección de Estadística de la Provincia (DPE), acumulado 2023-2025 dividido por 3. No es el rubro entero — es el producto HS4 concreto. |
| **# firmas** | Cantidad de firmas que evidencian el HS4 (declared-ncm + curated combinadas). |
| **Firmas ejemplo** | Sample de hasta 5 nombres de firmas evidenciando este HS4. |

Al mover el slider **Umbral exportación promedio anual** del sidebar, la tabla se restringe a los HS4 cuyo valor exportado cumple el umbral.
        """)

    st.dataframe(
        _anchor_tbl[[
            "HS4", "Producto", "Sector", "CCOD_RUBRO", "Rubro INDEC", "Match",
            "Exportación promedio anual (USD M)", "# firmas", "Firmas ejemplo",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "HS4": st.column_config.TextColumn("HS4", width="small"),
            "Producto": st.column_config.TextColumn("Producto", width="medium"),
            "Sector": st.column_config.TextColumn("Sector"),
            "CCOD_RUBRO": st.column_config.TextColumn("CCOD_RUBRO", width="small"),
            "Rubro INDEC": st.column_config.TextColumn("Rubro INDEC", width="medium"),
            "Match": st.column_config.TextColumn(
                "Match",
                help="`Directo` cuando el rubro INDEC es específico (clean / named-aggregate / broad-chapter / resto). `Residual` cuando el rubro es confidencial INDEC (código terminado en 899): el HS4 sólo se establece desde el lado firm.",
                width="small",
            ),
            "Exportación promedio anual (USD M)": st.column_config.NumberColumn(
                "Exportación promedio anual (USD M)", format="%.1f",
                help="Exportación del HS4 en Córdoba, promedio anual 2023-2025 según DPE (Dirección de Estadística de la Provincia). Es específico del HS4, no del rubro entero.",
            ),
            "# firmas": st.column_config.NumberColumn("# firmas", format="%.0f"),
            "Firmas ejemplo": st.column_config.TextColumn("Firmas ejemplo", width="large"),
        },
    )
    _csv = _anchor_tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar tabla de anclas (CSV)", _csv,
                       "cordoba_anclas.csv", "text/csv")

    # ---------------------------------------------------------------------------
    # 5. Sankey diagram (anchor → candidate)
    # ---------------------------------------------------------------------------
    st.subheader("Sankey — anclas → candidatos")

    _sankey_anchor_options = (
        flt[["anchor_hs4", "anchor_product_name_es"]]
        .drop_duplicates()
        .assign(
            anchor_hs4=lambda d: d["anchor_hs4"].astype(str).str.zfill(4),
            label=lambda d: d["anchor_hs4"].astype(str).str.zfill(4)
                            + " - " + d["anchor_product_name_es"].astype(str),
        )
        .sort_values("anchor_hs4")
    )
    _sankey_label_to_hs4 = dict(
        zip(_sankey_anchor_options["label"], _sankey_anchor_options["anchor_hs4"])
    )
    selected_sankey_anchor_labels = st.multiselect(
        "Filtrar anclas en el diagrama Sankey",
        options=_sankey_anchor_options["label"].tolist(),
        default=[],
        help=(
            "Vacío = mostrar todas las anclas del set filtrado. Elegí una o más "
            "para ver sólo sus links y los candidatos asociados."
        ),
        key="c4_sankey_anchor_selection",
    )
    if selected_sankey_anchor_labels:
        _sankey_anchor_hs4 = {_sankey_label_to_hs4[l] for l in selected_sankey_anchor_labels}
        _candidate = flt[flt["anchor_hs4"].astype(str).str.zfill(4).isin(_sankey_anchor_hs4)]
        if _candidate.empty:
            st.info("La selección actual no produce links — mostrando todas las anclas filtradas.")
            flt_sankey = flt
        else:
            flt_sankey = _candidate.copy()
    else:
        flt_sankey = flt

    anchor_labels = (
        flt_sankey[["anchor_hs4", "anchor_product_name_es", "anchor_sector"]]
        .drop_duplicates()
        .assign(node_label=lambda d: d["anchor_hs4"] + " - " + d["anchor_product_name_es"].astype(str))
    )
    candidate_labels = (
        flt_sankey[["candidate_hs4", "candidate_product_name_es", "candidate_sector"]]
        .drop_duplicates()
        .assign(node_label=lambda d: d["candidate_hs4"] + " - " + d["candidate_product_name_es"].astype(str))
    )

    anchor_node_ids = {k: i for i, k in enumerate(anchor_labels["node_label"].tolist())}
    candidate_offset = len(anchor_node_ids)
    candidate_node_ids = {k: candidate_offset + i for i, k in enumerate(candidate_labels["node_label"].tolist())}

    links_built = flt_sankey.assign(
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
            pad=24, thickness=18,
            line=dict(color="rgba(15,23,42,0.15)", width=0.6),
            label=node_labels, color=node_colors,
            hoverlabel=dict(font=dict(size=14)),
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
    _n_sankey_nodes = max(len(anchor_node_ids), len(candidate_node_ids))
    _sankey_height = max(760, min(1800, 22 * _n_sankey_nodes + 200))
    sankey.update_traces(textfont=dict(size=14, color="#0f172a", family="Inter, system-ui, sans-serif"))
    sankey.update_layout(
        title="Sankey — anclas → candidatos por proximidad",
        font=dict(size=14, color="#0f172a"),
        margin=dict(t=60, l=10, r=10, b=10),
        height=_sankey_height,
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
            distance_pctile=lambda d: d["distance_pctile"] * 100,
        )
    )
    # Publish the exact top-N displayed here so Página 4 (Mercado Accesible)
    # can mirror this ranking rather than recompute a preset. Preserves the
    # user's slider/weight choices — the list stays live-linked.
    st.session_state["p3_displayed_candidates"] = (
        candidate_display["candidate_hs4"].astype(str).str.zfill(4).tolist()
    )

    with st.expander("Diccionario de columnas — Tabla de candidatos"):
        st.markdown(r"""
| Columna | Significado |
|---|---|
| **Ranking** | Posición del candidato ordenado por Puntaje combinado descendente. |
| **HS4** | Código HS4 del candidato (HS 1992). |
| **Producto** | Nombre corto del HS4 en español (curado, ~1.240 entradas). |
| **Sector** | Sector Atlas / Growth Lab del HS4. |
| **Posible ancla** | Dummy 1/0. `1` = el candidato pertenece al set de 462 HS4 evidenciados pero su OPEX o su # firmas quedó por debajo del umbral del slider — es una ex-ancla reaparecida como candidato. Ver Inicio → glosario. |
| **Exportación Córdoba (USD M)** | Exportaciones actuales de Córdoba en este HS4, promedio anual 2023-2025 (USD millones, fuente DPE). `> 0` sólo para los candidatos flagueados como *posible ancla*; `0` para candidatos puros (sin evidencia previa de exportación desde Córdoba). |
| **Puntaje combinado** | `(1 − balance) · Factibilidad + balance · Atractivo`, normalizado 0-1 dentro del set filtrado. `balance` es el dial del sidebar. |
| **Índice de atractivo** | Promedio ponderado del PCI, tamaño del mercado accesible y crecimiento a 5 años del mercado accesible, normalizado 0-1. |
| **Índice de factibilidad** | Promedio ponderado del DAI, percentil de distancia recorrida y # de anclas normalizado, normalizado 0-1. |
| **DAI (crudo)** | Índice de alineación de demanda del HS4 (valor sin percentilar). Fórmula en Inicio. |
| **PCI** | Product Complexity Index del HS4 (más alto = más complejo). |
| **DAI (percentil)** | Percentil del DAI de Argentina contra el set top-30 exportadores + Argentina (0-100). Mayor = mejor posición competitiva. |
| **Distancia recorrida** | Km promedio que recorre el HS4 globalmente, ponderado por valor bilateral (atributo del producto). |
| **Distancia (percentil)** | Percentil de `distancia recorrida` dentro del set filtrado (0-100). Mayor = producto más tradeable → más factible. |
| **Crecimiento del mercado accesible % (5 años)** | CAGR 2020-2024 del mercado accesible del HS4. |
| **Mercado accesible (USD mil M)** | Tamaño del mercado accesible en miles de millones USD (2024). |
| **Proximidad promedio** | Media de las proximidades HS4↔HS4 entre el candidato y las anclas que lo tienen en su top-1%. En [0, 1]. |
| **# anclas** | Cantidad de anclas del set filtrado que tienen a este candidato en su top-1% de proximidad. |
| **Anclas (HS4)** | HS4 (separados por · ) de esas anclas. |
        """)

    st.dataframe(
        candidate_display[[
            "rank", "candidate_hs4", "candidate_product_name_es", "candidate_sector",
            "posible_ancla", "cordoba_export_usd_m",
            "combined_score", "attractiveness_index", "feasibility_index",
            "dai_index", "pci", "dai_percentile", "distance_travelled", "distance_pctile",
            "accessible_market_growth_5y", "accessible_market_size_b",
            "avg_proximity", "anchor_count", "anchors",
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
            "cordoba_export_usd_m": st.column_config.NumberColumn(
                "Exportación Córdoba (USD M)",
                format="%.2f",
                help=(
                    "Exportaciones actuales de Córdoba en este HS4, promedio "
                    "anual 2023-2025 (USD millones, fuente DPE). >0 sólo para "
                    "candidatos flagueados como posible ancla; 0 para "
                    "candidatos puros sin evidencia previa."
                ),
            ),
            "combined_score": st.column_config.NumberColumn("Puntaje combinado", format="%.4f"),
            "attractiveness_index": st.column_config.NumberColumn("Índice de atractivo", format="%.4f"),
            "feasibility_index": st.column_config.NumberColumn("Índice de factibilidad", format="%.4f"),
            "dai_index": st.column_config.NumberColumn("DAI (crudo)", format="%.3f"),
            "pci": st.column_config.NumberColumn("PCI", format="%.3f"),
            "dai_percentile": st.column_config.NumberColumn("DAI (percentil)", format="%.1f"),
            "distance_travelled": st.column_config.NumberColumn("Distancia recorrida", format="%.1f"),
            "distance_pctile": st.column_config.NumberColumn(
                "Distancia (percentil)",
                format="%.1f",
                help=(
                    "Percentil del HS4 en la distribución de `distance_travelled` "
                    "dentro del set filtrado. Mayor = el producto viaja más lejos "
                    "globalmente = más tradeable = más factible."
                ),
            ),
            "accessible_market_growth_5y": st.column_config.NumberColumn("Crecimiento del mercado accesible % (5 años)", format="%.2f%%"),
            "accessible_market_size_b": st.column_config.NumberColumn("Mercado accesible (USD mil M)", format="%.3f"),
            "avg_proximity": st.column_config.NumberColumn("Proximidad promedio", format="%.4f"),
            "anchor_count": st.column_config.NumberColumn("# anclas", format="%.0f"),
            "anchors": st.column_config.TextColumn(
                "Anclas (HS4)",
                help=(
                    "HS4 de las anclas que tienen este candidato en su top-1% de "
                    "proximidad (separadas por ·)."
                ),
                width="medium",
            ),
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
    treemap.update_layout(margin=dict(t=60, l=10, r=10, b=95), height=720)
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

# ---------------------------------------------------------------------------
# Multi-page navigation
# ---------------------------------------------------------------------------
@st.cache_data
def _recomendado_top_candidates(
    _signature: str = "",
    top_n: int = 30,
    profiles: tuple[str, ...] = DEFAULT_ANCHOR_PROFILES,
) -> list[str]:
    """Return the candidate HS4s (zfilled) that would appear in the
    Recomendado preset ranking on the Análisis de Proximidad page, top-N
    by combined score.

    Mirrors _apply_profile("top_candidates"): anchor OPEX ≥ USD 10 M,
    sections 1/2/3 excluded on both sides, natural-resource + manually
    excluded HS4 dropped, proximity rank ∈ [1, 10], market ≥ USD 0.5 B,
    growth > 0, strategic balance 0.50, weights (0.40/0.40/0.20 feas;
    0.50/0.25/0.25 attr).

    `profiles` propagates the exporter_profile filter set on the Firmas
    y Rubros page — HS4 evidenced only by firms outside this set drop
    out of the anchor universe.
    """
    df, presence, _, _, _, _ = load_data()

    # Restrict the anchor universe to HS4 that have at least one firm in
    # the selected exporter_profile set (cross-page filter). Natural
    # resources are excluded from BOTH sides — the Recomendado preset
    # semantics dictate it, and this fallback should mirror them.
    n_firms_dyn = anchor_firms_per_hs4(_signature, profiles)
    profile_ok = {h for h, n in n_firms_dyn.items() if n > 0}
    _nat_res_set = set(NATURAL_RESOURCE_HS4)
    anchor_universe = (
        set(
            presence.loc[
                presence["usd_promedio_anual"].fillna(0) >= 10_000_000,
                "hs4",
            ].astype(str).str.zfill(4)
        )
        & profile_ok
    ) - _nat_res_set
    za = df["anchor_hs4"].astype(str).str.zfill(4)
    zc = df["candidate_hs4"].astype(str).str.zfill(4)
    flt = df[za.isin(anchor_universe) & ~zc.isin(anchor_universe)].copy()
    for col in ("anchor_hs_section_name", "candidate_hs_section_name"):
        flt = flt[~flt[col].astype(str).str.match(r"^[123]\.")]
    _excluded = set(NATURAL_RESOURCE_HS4) | set(PRESET_EXCLUDED_HS4)
    flt = flt[~flt["candidate_hs4"].astype(str).str.zfill(4).isin(_excluded)]
    flt = flt[
        pd.to_numeric(flt["proximity_rank"], errors="coerce").between(1, 10)
        & (pd.to_numeric(flt["accessible_market_size_b"], errors="coerce") >= 0.5)
        & (pd.to_numeric(flt["accessible_market_growth_5y"], errors="coerce") > 0)
    ]
    if flt.empty:
        return []

    cs = flt.groupby("candidate_hs4", as_index=False).agg(
        accessible_market_size=("accessible_market_size", "first"),
        accessible_market_growth_5y=("accessible_market_growth_5y", "first"),
        dai_percentile=("dai_percentile", "first"),
        pci=("pci", "first"),
        distance_travelled=("distance_travelled", "first"),
        anchor_count=("anchor_hs4", "nunique"),
    )
    cs["dai_mm"] = normalize_0_1(cs["dai_percentile"])
    cs["distance_pctile"] = (
        pd.to_numeric(cs["distance_travelled"], errors="coerce").rank(pct=True).fillna(0.5)
    )
    cs["pci_mm"] = normalize_0_1(cs["pci"])
    cs["growth_mm"] = normalize_0_1(cs["accessible_market_growth_5y"])
    cs["market_mm"] = normalize_0_1(cs["accessible_market_size"])
    cs["anchor_mm"] = normalize_0_1(cs["anchor_count"])

    feas = 0.40 * cs["dai_mm"] + 0.40 * cs["distance_pctile"] + 0.20 * cs["anchor_mm"]
    attr = 0.50 * cs["pci_mm"] + 0.25 * cs["growth_mm"] + 0.25 * cs["market_mm"]
    cs["combined"] = 0.5 * feas + 0.5 * attr
    top = cs.sort_values("combined", ascending=False).head(top_n)
    return top["candidate_hs4"].astype(str).str.zfill(4).tolist()


def page_mercado_accesible():
    _page_header(
        "Mercado Accesible por Producto",
        "Composición geográfica del mercado accesible para cada HS4 (2024). "
        "Argentina puede alcanzar cada destino porque está dentro de la "
        "distancia recorrida del producto o porque ya exporta ≥ USD 100 M."
    )

    am = load_accessible_market(_data_signature() if "_data_signature" in globals() else "")

    # Cross-page filter: inherit the exporter_profile selection from Firmas y
    # Rubros so the top-30 shown here matches the anchor universe upstream.
    selected_profiles = _selected_anchor_profiles()
    if set(selected_profiles) != set(DEFAULT_ANCHOR_PROFILES):
        st.info(
            "🔎 Filtro cruzado activo: **Perfil exportador = "
            f"{', '.join(selected_profiles)}** (heredado de la página "
            "*Firmas y Rubros*)."
        )

    # Selector de fuente: (1) Análisis de Proximidad live/preset,
    # (2) Margen Intensivo top-15, (3) Margen Extensivo top-30.
    # Top-N por preset viene del propio OPP_PRESETS (single source of truth).
    _n_int = int(OPP_PRESETS["margen_intensivo"]["top_n"])
    _n_ext = int(OPP_PRESETS["margen_extensivo"]["top_n"])
    hs4_in_am = set(am["hs92"].unique())
    label_intensivo = f"Margen Intensivo (top {_n_int})"
    label_extensivo = f"Margen Extensivo (top {_n_ext})"
    fuente_opciones = [
        "Análisis de Proximidad (vivo)",
        label_intensivo,
        label_extensivo,
    ]
    st.session_state.setdefault("mercado_fuente", fuente_opciones[0])
    # If the stored label doesn't match current top_n (e.g. after preset changes),
    # reset to default to avoid a ValueError in options.index().
    if st.session_state["mercado_fuente"] not in fuente_opciones:
        st.session_state["mercado_fuente"] = fuente_opciones[0]
    fuente = st.selectbox(
        "Fuente del universo de productos",
        options=fuente_opciones,
        index=fuente_opciones.index(st.session_state["mercado_fuente"]),
        key="mercado_fuente",
        help=(
            "Elegí qué lista de HS4 mostrar acá: la tabla viva de la página "
            "*Análisis de Proximidad*, o el top-N de uno de los presets de "
            "Oportunidades (Margen Intensivo top-15 · Margen Extensivo top-30, "
            "según el algoritmo Growth Lab)."
        ),
    )

    if fuente == label_intensivo:
        source_label = f"preset **Margen Intensivo** (top {_n_int} fijo)"
        candidates_ordered = _preset_top_n_candidates(
            _data_signature() if "_data_signature" in globals() else "",
            "margen_intensivo",
        )
    elif fuente == label_extensivo:
        source_label = f"preset **Margen Extensivo** (top {_n_ext} fijo)"
        candidates_ordered = _preset_top_n_candidates(
            _data_signature() if "_data_signature" in globals() else "",
            "margen_extensivo",
        )
    else:
        # Fuente por defecto: replica lo que hay live en Análisis de Proximidad,
        # o cae al preset Recomendado si el usuario no visitó esa página aún.
        p3_live = st.session_state.get("p3_displayed_candidates")
        if p3_live:
            source_label = "vivo desde **Análisis de Proximidad**"
            candidates_ordered = [str(h).zfill(4) for h in p3_live]
        else:
            source_label = "preset **Recomendado**"
            candidates_ordered = _recomendado_top_candidates(
                _data_signature() if "_data_signature" in globals() else "",
                profiles=selected_profiles,
            )

    st.caption(
        f"Universo actual: {source_label}. Cambiá la fuente en el selector de arriba "
        f"para alternar entre la lista viva de Análisis de Proximidad y los top-30 "
        f"de los presets de la página *Oportunidades*."
    )

    # Preserve the ranking order, but only keep HS4 with accessible-market data
    product_universe = [h for h in candidates_ordered if h in hs4_in_am]
    if not product_universe:
        st.info("La tabla de candidatos de Análisis de Proximidad no tiene "
                "productos con datos de mercado accesible bajo los filtros actuales.")
        return
    n_dropped = len(candidates_ordered) - len(product_universe)
    if n_dropped:
        st.caption(
            f"({n_dropped} HS4 del ranking no tienen datos de mercado accesible "
            f"y quedaron fuera del selector.)"
        )

    # Label as "HS4 - Producto (Spanish)"
    label_by_hs4 = {
        h: f"{h} - {SPANISH_OVERRIDES.get(h, '')}".rstrip(" -")
        for h in product_universe
    }
    ALL_LABEL = f"— Todos los productos ({len(product_universe)}) —"
    per_product_labels = [label_by_hs4[h] for h in product_universe]
    labels = [ALL_LABEL] + per_product_labels

    st.selectbox(
        "Producto (HS4)",
        options=labels,
        key="c4_am_product_label",
        index=0,  # default: vista agregada sobre todos los HS4 del universo
        help=(
            "Por defecto se muestra el mercado accesible agregado sobre todos "
            "los HS4 del universo seleccionado arriba. Elegí un producto "
            "específico para ver su composición por destino individualmente."
        ),
    )
    picked_label = st.session_state.get("c4_am_product_label")
    if not picked_label:
        st.info("Sin productos con datos de mercado accesible.")
        return

    is_all = (picked_label == ALL_LABEL)
    if is_all:
        picked_hs4 = None
        sub_raw = am[am["hs92"].isin(product_universe)].copy()
        # Un HS4 aporta una fila por destino → agregamos total_imports por iso3_d
        # para que el treemap tenga una baldosa única por destino.
        sub = (
            sub_raw.groupby("iso3_d", as_index=False)["total_imports"]
            .sum()
        )
        n_products = int(sub_raw["hs92"].nunique())
    else:
        picked_hs4 = picked_label.split(" - ", 1)[0]
        sub = am[am["hs92"] == picked_hs4].copy()
        n_products = 1
    sub["mercado_b"] = sub["total_imports"] / 1e9  # miles de millones USD
    sub["mercado_m"] = sub["total_imports"] / 1e6  # millones USD
    total_b = float(sub["mercado_b"].sum())
    n_dest = int(sub["iso3_d"].nunique())

    c1, c2, c3 = st.columns(3)
    if is_all:
        c1.metric("Productos agregados", f"{n_products}")
    else:
        c1.metric("Producto seleccionado", picked_hs4)
    c2.metric("Mercado accesible total (miles de millones USD)", f"{total_b:.1f}")
    c3.metric("Destinos accesibles", f"{n_dest}")

    if sub.empty or total_b == 0:
        st.info(
            "Sin destinos accesibles para este universo."
            if is_all else
            "Sin destinos accesibles para este producto."
        )
        return

    # Treemap — tile per (continente, destino), colored por continente
    sub["participacion"] = sub["mercado_b"] / total_b
    sub["continente"] = sub["iso3_d"].map(_ISO3_TO_CONTINENT).fillna("Otros")
    _title_lead = (
        f"Mercado accesible agregado — {n_products} HS4 del universo"
        if is_all else f"Mercado accesible para {picked_label}"
    )
    fig = px.treemap(
        sub,
        path=["continente", "iso3_d"],
        values="mercado_b",
        color="continente",
        color_discrete_map=CONTINENT_COLORS,
        hover_data={
            "mercado_b": ":.3f",
            "mercado_m": ":.1f",
            "participacion": ":.1%",
            "iso3_d": False,
            "continente": False,
        },
        title=(
            f"{_title_lead} "
            f"(n = {n_dest} destinos | total = {total_b:,.2f} USD mil M) "
            f"| tamaño = importaciones totales del destino · color = continente"
        ),
    )
    fig.update_traces(
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>$%{value:,.2f} mil M",
        textfont=dict(size=16, color="#ffffff"),
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
    )
    fig.update_layout(
        margin=dict(t=60, l=10, r=10, b=95),
        height=620,
        legend=dict(
            orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
            title_text="Continente",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Competidores — top-20 exportadores hacia el mercado accesible
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Top-20 competidores")
    st.caption(
        "Quiénes están exportando este producto hacia los mercados a los que "
        "Argentina también puede llegar. Datos BACI 2024. Podés restringir el "
        "cálculo a un destino específico o dejarlo sobre todos los destinos "
        "accesibles del producto."
    )

    competitors_all = load_competitors_bilateral(
        _data_signature() if "_data_signature" in globals() else ""
    )
    if is_all:
        comp_sub = competitors_all[competitors_all["hs92"].isin(product_universe)].copy()
    else:
        comp_sub = competitors_all[competitors_all["hs92"] == picked_hs4].copy()

    # Destination filter: 'Todos los mercados accesibles' + one entry per iso3_d
    accessible_dests = sorted(sub["iso3_d"].dropna().unique().tolist())
    OPTION_ALL = "Todos los mercados accesibles"
    dest_options = [OPTION_ALL] + accessible_dests
    _dest_key_suffix = "ALL_UNIVERSE" if is_all else str(picked_hs4)
    picked_dest = st.selectbox(
        "Mercado para treemap de competidores",
        options=dest_options,
        index=0,
        key=f"c4_am_competitor_dest_{_dest_key_suffix}",
    )
    st.caption(f"Mercado seleccionado para treemap de competidores: **{picked_dest}**")

    if picked_dest != OPTION_ALL:
        comp_sub = comp_sub[comp_sub["iso3_d"] == picked_dest]
    # Also restrict to accessible destinations only when 'All' is picked
    else:
        comp_sub = comp_sub[comp_sub["iso3_d"].isin(accessible_dests)]

    # Argentina is not a competitor of itself — remove.
    comp_sub = comp_sub[comp_sub["iso3_o"] != "ARG"]

    # Aggregate per exporter
    per_exp = (
        comp_sub.groupby("iso3_o", as_index=False)["export_value"].sum()
        .rename(columns={"export_value": "value_usd"})
        .sort_values("value_usd", ascending=False)
    )
    if per_exp.empty or per_exp["value_usd"].sum() == 0:
        st.info(
            "No hay exportaciones desde competidores hacia este mercado en 2024 "
            "según los datos BACI disponibles."
        )
        return

    total_market_usd = float(per_exp["value_usd"].sum())
    top20 = per_exp.head(20).copy()
    top20_total = float(top20["value_usd"].sum())
    coverage_top20 = top20_total / total_market_usd if total_market_usd > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Exportadores mostrados", f"{len(top20)}")
    c2.metric("Exportaciones acumuladas (M USD)", f"{top20_total/1e6:,.1f}")
    c3.metric("Cobertura del top 20", f"{coverage_top20:.1%}")

    # Color: continente for consistency with the destinations treemap
    top20["continente"] = top20["iso3_o"].map(_ISO3_TO_CONTINENT).fillna("Otros")
    top20["value_m"] = top20["value_usd"] / 1e6

    fig_comp = px.treemap(
        top20,
        path=["continente", "iso3_o"],
        values="value_m",
        color="continente",
        color_discrete_map=CONTINENT_COLORS,
        hover_data={
            "value_m": ":.1f",
            "iso3_o": False,
            "continente": False,
        },
        title=(
            f"Treemap de competidores para {picked_dest} | "
            f"{'Universo agregado (' + str(n_products) + ' HS4)' if is_all else 'Producto ' + str(picked_label)} · "
            f"tamaño = exportaciones a {picked_dest if picked_dest != OPTION_ALL else 'destinos accesibles'} "
            f"en 2024 (M USD) · color = continente"
        ),
    )
    fig_comp.update_traces(
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>$%{value:,.0f}M",
        textfont=dict(size=16, color="#ffffff"),
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
    )
    fig_comp.update_layout(
        margin=dict(t=60, l=10, r=10, b=95),
        height=620,
        legend=dict(
            orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
            title_text="Continente",
        ),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("Cómo se calcula"):
        st.markdown(r"""
Para el producto seleccionado, tomamos todas las **exportaciones de cada
país** (BACI 2024, valor FOB) hacia los destinos que forman parte del
**mercado accesible** de Argentina para ese HS4. Argentina queda excluida
de la lista de competidores (se excluye a sí misma).

- **"Todos los mercados accesibles"** suma los flujos exportados por
  cada país hacia todos los destinos accesibles.
- **Un destino específico** restringe la suma a ese único mercado —
  útil para ver quiénes son los rivales concretos en, por ejemplo, USA.

**Cobertura del top 20**: qué porcentaje del total exportado por
competidores hacia el mercado accesible cubren los 20 países mostrados.
Un número alto (>80%) indica un mercado concentrado; un número bajo
indica muchos exportadores marginales relevantes.
        """)

    st.download_button(
        "⬇ Descargar datos de competidores (CSV)",
        per_exp.to_csv(index=False).encode("utf-8"),
        f"competidores_{'universo' if is_all else picked_hs4}_{picked_dest.replace(' ', '_')}.csv",
        "text/csv",
    )


def page_exportaciones_producto():
    """Exportaciones por producto (HS4). Ground truth: Dirección de Estadística
    de la Provincia (DPE), archivo 'ExpoCba participacion NCM_23_25.xlsx',
    escalado por el total absoluto OPEX 2023-2025.
    Treemap sector Atlas → HS4, size = share, tooltip con USD promedio anual.
    Click en un HS4 → tabla de firmas identificadas para ese HS4."""
    _page_header(
        "Exportaciones por producto",
        "Composición de las exportaciones cordobesas 2023-2025 por HS4, "
        "agrupadas por sector Atlas. Fuente: Dirección de Estadística de la "
        "Provincia (DPE). Cada baldosa representa un HS4 — tamaño = "
        "participación %; el USD promedio anual está en el tooltip. "
        "Clickeá un HS4 para ver las firmas identificadas que lo exportan."
    )

    df_prox, presence, umap, trade, cluster_color, names = load_data()

    # ------------------------------------------------------------------
    # Attach Atlas sector via HS4 → sector map (from proximity file)
    # ------------------------------------------------------------------
    hs4_to_sector = load_hs4_sector_map(_data_signature() if "_data_signature" in globals() else "")

    presence = presence.copy()
    presence["hs4"] = presence["hs4"].astype(str).str.zfill(4)
    presence["sector"] = presence["hs4"].map(hs4_to_sector).fillna("Otros")
    presence["hs4_es"] = presence["hs4"].map(lambda h: SPANISH_OVERRIDES.get(h, ""))
    presence["hs4_label"] = presence["hs4"] + presence["hs4_es"].where(
        presence["hs4_es"] == "", " - " + presence["hs4_es"]
    )
    presence["usd_promedio_anual"] = pd.to_numeric(
        presence["usd_promedio_anual"], errors="coerce"
    ).fillna(0)
    presence["usd_promedio_m"] = presence["usd_promedio_anual"] / 1e6
    presence["share_pct"] = pd.to_numeric(presence.get("share_2023_2025"),
                                           errors="coerce").fillna(0) * 100

    # Only HS4 with positive exports go into the treemap
    tm_data = presence[presence["usd_promedio_anual"] > 0].copy()
    total_usd = float(tm_data["usd_promedio_anual"].sum())

    # ------------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exportación total (USD M · prom anual)", f"{total_usd/1e6:,.0f}")
    c2.metric("HS4 mostrados", f"{len(tm_data):,}")
    c3.metric("Sectores Atlas", f"{tm_data['sector'].nunique()}")
    c4.metric("Firmas identificadas", f"{int(tm_data['n_firms'].sum()):,}")

    st.caption(
        "Los HS4 se ordenan por su participación en las exportaciones "
        "totales de Córdoba. Cobertura acumulada: top 30 HS4 = "
        f"{tm_data.sort_values('usd_promedio_anual', ascending=False).head(30)['share_pct'].sum():.1f} % · "
        f"top 100 HS4 = {tm_data.sort_values('usd_promedio_anual', ascending=False).head(100)['share_pct'].sum():.1f} %."
    )

    # ------------------------------------------------------------------
    # Treemap por sector Atlas → HS4
    # ------------------------------------------------------------------
    tm_data["descripcion_corta"] = tm_data["descripcion_top_ncm"].fillna("").astype(str).str[:60]
    fig_tm = px.treemap(
        tm_data,
        path=["sector", "hs4_label"],
        values="usd_promedio_m",
        color="sector",
        color_discrete_map=SECTOR_COLORS,
        hover_data={
            "usd_promedio_m": ":.1f",
            "share_pct": ":.2f",
            "n_firms": True,
            "descripcion_corta": True,
            "hs4_label": False,
            "sector": False,
        },
        title=(
            f"Exportaciones por HS4 (sector Atlas → HS4) | "
            f"n = {len(tm_data)} HS4 · total = {total_usd/1e6:,.0f} USD M/año · "
            f"tamaño = exportación promedio anual · color = sector Atlas"
        ),
    )
    fig_tm.update_traces(
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>$%{value:,.0f} M",
        textfont=dict(size=15, color="#ffffff"),
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
    )
    fig_tm.update_layout(
        margin=dict(t=60, l=10, r=10, b=95),
        height=720,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                    title_text="Sector Atlas"),
    )
    tm_state = st.plotly_chart(
        fig_tm,
        use_container_width=True,
        on_select="rerun",
        key="expo_prod_treemap",
    )

    st.caption(
        "Fuente: DPE (Dirección de Estadística de la Provincia de Córdoba), "
        "archivo participación NCM 2023-2025, escalado por total OPEX anual. "
        "Cada NCM 8-dígitos se agrega a su HS4 (primeros 4 dígitos)."
    )

    # ------------------------------------------------------------------
    # Decode treemap click → HS4 filter for the firms table below
    # ------------------------------------------------------------------
    selected_hs4: set = set()
    selected_sectors: set = set()
    try:
        pts = tm_state.selection.points if hasattr(tm_state, "selection") else (tm_state.get("selection", {}) or {}).get("points", [])
    except Exception:
        pts = []
    label_to_hs4 = dict(zip(tm_data["hs4_label"].astype(str), tm_data["hs4"].astype(str)))
    for p in pts or []:
        pid = p.get("id", "") if isinstance(p, dict) else getattr(p, "id", "")
        label = p.get("label", "") if isinstance(p, dict) else getattr(p, "label", "")
        if not pid:
            continue
        if "/" in pid:
            code = label_to_hs4.get(str(label))
            if code:
                selected_hs4.add(code)
        else:
            selected_sectors.add(str(label))

    # ------------------------------------------------------------------
    # Firms table for selected HS4 (or all if nothing selected)
    # ------------------------------------------------------------------
    firm_ev, _ = load_firms_data(_data_signature() if "_data_signature" in globals() else "")
    firm_ev["hs4"] = firm_ev["hs4"].astype(str).str.zfill(4)
    firm_ev["sector"] = firm_ev["hs4"].map(hs4_to_sector).fillna("Otros")

    # Apply exporter_profile filter inherited from Firmas y Rubros page.
    selected_profiles = _selected_anchor_profiles()
    firm_ev = firm_ev[firm_ev["exporter_profile"].astype(str).isin(selected_profiles)]

    st.subheader("Firmas identificadas")
    if selected_hs4:
        firm_filt = firm_ev[firm_ev["hs4"].isin(selected_hs4)].copy()
        hs4_labels = ", ".join(sorted(selected_hs4))
        st.caption(f"Filtrando por HS4 seleccionados: **{hs4_labels}** · "
                   f"click en el fondo del treemap para limpiar.")
    elif selected_sectors:
        firm_filt = firm_ev[firm_ev["sector"].astype(str).isin(selected_sectors)].copy()
        st.caption(f"Filtrando por sector: **{', '.join(sorted(selected_sectors))}** · "
                   f"click en el fondo del treemap para limpiar.")
    else:
        firm_filt = firm_ev.copy()
        st.caption("Mostrando todas las firmas identificadas. Clickeá un HS4 "
                   "o sector en el treemap para filtrar.")

    if firm_filt.empty:
        st.info("No hay firmas identificadas para este filtro.")
    else:
        firm_disp = firm_filt.copy()
        firm_disp["HS4"] = firm_disp["hs4"] + firm_disp["hs4"].map(
            lambda h: (" - " + SPANISH_OVERRIDES.get(h, "")) if SPANISH_OVERRIDES.get(h) else ""
        )
        firm_disp = firm_disp[[
            "firm_name", "razon_social", "HS4", "sector",
            "evidence_layer", "exporter_profile", "confidence",
            "evidence_text", "source_url",
        ]].rename(columns={
            "firm_name": "Firma", "razon_social": "Razón social",
            "sector": "Sector", "evidence_layer": "Capa",
            "exporter_profile": "Perfil exportador",
            "confidence": "Confianza", "evidence_text": "Evidencia",
            "source_url": "Página registro",
        }).sort_values(["Firma", "HS4"])

        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        c_kpi1.metric("Firmas únicas", f"{firm_disp['Firma'].nunique():,}")
        c_kpi2.metric("Filas (firma × HS4)", f"{len(firm_disp):,}")
        c_kpi3.metric("HS4 cubiertos", f"{firm_disp['HS4'].nunique():,}")

        st.dataframe(
            firm_disp, use_container_width=True, hide_index=True,
            column_config={
                "Firma": st.column_config.TextColumn("Firma", width="medium"),
                "Razón social": st.column_config.TextColumn("Razón social", width="medium"),
                "HS4": st.column_config.TextColumn("HS4", width="medium"),
                "Sector": st.column_config.TextColumn("Sector"),
                "Capa": st.column_config.TextColumn("Capa"),
                "Perfil exportador": st.column_config.TextColumn("Perfil exportador"),
                "Confianza": st.column_config.TextColumn("Confianza"),
                "Evidencia": st.column_config.TextColumn("Evidencia", width="large"),
                "Página registro": st.column_config.LinkColumn("Registro", display_text="↗", width="small"),
            },
        )
        st.download_button(
            "⬇ Descargar tabla de firmas (CSV)",
            firm_disp.to_csv(index=False).encode("utf-8"),
            "cordoba_firmas_por_hs4.csv",
            "text/csv",
        )


# ============================================================================
# Página: Oportunidades — Feasibility × Attractiveness (adaptación de ARG_V2)
# ============================================================================

@st.cache_data
def load_oportunidades_dataset(_signature: str = "") -> pd.DataFrame:
    """Loader del CSV opportunity_metrics_hs4_cordoba.csv.
    Contiene 1,241 HS4 con:
      - Componentes de complejidad de Córdoba (nuestro panel): RCA raw+transformed,
        PCI, COG, density, density_percentile.
      - Componentes heredados de ARG_V2: accessible_market_size, growth, share,
        DAI (index/percentile/lead), distance_travelled, eff_num_exp, market_growth_5y.
      - Rankings globales de Córdoba y ARG-sin-Córdoba por HS4.
      - Percentiles de cada componente (bounded [0,1]).
    """
    df = pd.read_csv(_data("output", "opportunity_metrics_hs4_cordoba.csv"), dtype={"hs4": str})
    df["hs4"] = df["hs4"].str.zfill(4)
    # Override product names with Spanish overrides (SPANISH_OVERRIDES combines
    # hs4_names_es.py — ~1,240 curated Spanish HS4 short names — with the
    # script-13 curated dict). Fallback to the ARG_V2 English name if no
    # Spanish override exists.
    _en_backup = df["product_name_short"].fillna("").astype(str)
    df["product_name_short"] = df["hs4"].map(
        lambda h: SPANISH_OVERRIDES.get(h, "")
    )
    df["product_name_short"] = df["product_name_short"].where(
        df["product_name_short"].astype(str).str.len() > 0, _en_backup
    )
    # Merge n_anchors_linking from cordoba_candidates_ranked (candidates in the
    # anchored-proximity output). HS4 that never appear as a candidate get 0.
    # n_anchors_linking_pct is a min-max normalization used as a feasibility
    # component for Margen Extensivo (imagen: Factibilidad = DAI 50% + # anclas 50%).
    try:
        _cand = pd.read_csv(
            _data("output", "cordoba_candidates_ranked.csv"),
            dtype={"candidate_hs4": str},
        )
        _cand["candidate_hs4"] = _cand["candidate_hs4"].str.zfill(4)
        df = df.merge(
            _cand[["candidate_hs4", "n_anchors_linking"]].rename(
                columns={"candidate_hs4": "hs4"}
            ),
            on="hs4", how="left",
        )
    except FileNotFoundError:
        df["n_anchors_linking"] = np.nan
    df["n_anchors_linking"] = pd.to_numeric(df["n_anchors_linking"], errors="coerce").fillna(0.0)
    _max_anchors = float(df["n_anchors_linking"].max() or 0.0)
    df["n_anchors_linking_pct"] = (
        df["n_anchors_linking"] / _max_anchors if _max_anchors > 0 else 0.0
    )
    return df


# Preset definitions (RCA filter + root filters + weights + balance + top_n)
# reusable across pages. Sigue el algoritmo del Growth Lab (Identificación de
# Oportunidades): Mercado Accesible ≥ 500M USD + CAGR 5y > 0, luego split RCA
# en Intensivo (0.7-10, top 15) y Extensivo (0-0.25, top 30).
OPP_PRESETS = {
    "margen_intensivo": {
        "label": "Margen Intensivo",
        "rca_range": (0.7, 10.0),
        "market_min_b": 0.5,
        "require_growth_positive": True,
        "top_n": 15,
        "balance": 0.50,
        "feas": {
            "rca_transformed_cba_pct":  0.00,
            "density_pct_cba_pct":      0.50,
            "eff_num_exp_pct":          0.00,
            "dai_pct_norm":             0.50,
            "distance_travelled_pct":   0.00,
            "n_anchors_linking_pct":    0.00,
        },
        "attr": {
            "pci_pct":                          0.50,
            "cog_pct":                          0.00,
            "accessible_market_size_share_pct": 0.25,
            "accessible_market_growth_5y_pct":  0.25,
        },
    },
    "margen_extensivo": {
        "label": "Margen Extensivo",
        "rca_range": (0.0, 0.25),
        "market_min_b": 0.5,
        "require_growth_positive": True,
        "top_n": 30,
        "balance": 0.50,
        "feas": {
            "rca_transformed_cba_pct":  0.00,
            "density_pct_cba_pct":      0.00,
            "eff_num_exp_pct":          0.00,
            "dai_pct_norm":             0.50,
            "distance_travelled_pct":   0.00,
            "n_anchors_linking_pct":    0.50,
        },
        "attr": {
            "pci_pct":                          0.35,
            "cog_pct":                          0.35,
            "accessible_market_size_share_pct": 0.15,
            "accessible_market_growth_5y_pct":  0.15,
        },
    },
}


@st.cache_data
def _preset_top_n_candidates(_signature: str, preset_name: str, top_n: int | None = None) -> list[str]:
    """Return top-N HS4 codes (zfilled strings) for an Oportunidades preset.

    Aplica el algoritmo Growth Lab en orden:
      1. Mercado Accesible ≥ market_min_b (default 500M USD si el preset lo declara).
      2. Mercado Accesible CAGR 5y > 0 (si require_growth_positive=True).
      3. RCA en el rango del preset (Intensivo 0.7-10 · Extensivo 0-0.25).
      4. Score = (1-balance)·feas + balance·attr con los pesos del preset.
      5. Retorna top-N (default = preset["top_n"] si no se pasa argumento).

    Single source of truth para lo que la página Mercado Accesible y la página
    Oportunidades entienden por "top Margen Intensivo/Extensivo".
    """
    if preset_name not in OPP_PRESETS:
        return []
    p = OPP_PRESETS[preset_name]
    if top_n is None:
        top_n = int(p.get("top_n", 30))
    df = load_oportunidades_dataset(_signature)

    # Root filters (Identificación de Oportunidades)
    market_min_b = float(p.get("market_min_b", 0.0))
    if market_min_b > 0:
        df = df[pd.to_numeric(df["accessible_market_size_b"], errors="coerce") >= market_min_b]
    if p.get("require_growth_positive", False):
        df = df[pd.to_numeric(df["accessible_market_growth_5y"], errors="coerce") > 0]

    # RCA-margin split
    rca_lo, rca_hi = p["rca_range"]
    df = df[(df["raw_rca_cba"] >= rca_lo) & (df["raw_rca_cba"] <= rca_hi)].copy()

    def _wavg(comp_weights):
        cols = [c for c, w in comp_weights.items() if w > 0 and c in df.columns]
        weights = [w for c, w in comp_weights.items() if w > 0 and c in df.columns]
        if not cols:
            return pd.Series(0.0, index=df.index)
        arr = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in cols])
        weight_sum = sum(weights)
        return pd.Series((arr * np.array(weights)).sum(axis=1) / weight_sum, index=df.index)

    df["_feas"] = _wavg(p["feas"])
    df["_attr"] = _wavg(p["attr"])
    bal = p["balance"]
    df["_score"] = (1 - bal) * df["_feas"] + bal * df["_attr"]
    top = df.sort_values("_score", ascending=False).head(top_n)
    return top["hs4"].astype(str).str.zfill(4).tolist()


def page_oportunidades_cordoba():
    _page_header(
        "Oportunidades — Feasibility × Attractiveness",
        "Ranking de HS4 combinando factibilidad (proximidad al know-how cordobés) "
        "y atractivo (complejidad del producto + mercado accesible). Adapta la "
        "metodología del Growth Lab / ARG_Dashboard_V2 al caso Córdoba: se usan "
        "métricas de Córdoba+BACI para complejidad y densidad, y se heredan de "
        "Argentina las métricas de mercado (accessible market, DAI, distance)."
    )

    df = load_oportunidades_dataset(_data_signature() if "_data_signature" in globals() else "")
    if df.empty:
        st.warning("El dataset de oportunidades está vacío.")
        st.stop()

    # ------------------------------------------------------------------------
    # Constantes de la página
    # ------------------------------------------------------------------------
    FEAS_COMPS = [
        ("w_rca",        "RCA transformada",           "rca_transformed_cba_pct"),
        ("w_density",    "Density (percentil)",        "density_pct_cba_pct"),
        ("w_eff",        "Effective exporters (inv.)", "eff_num_exp_pct"),
        ("w_dai",        "DAI percentil (heredado)",   "dai_pct_norm"),
        ("w_dist",       "Distancia recorrida %",      "distance_travelled_pct"),
        ("w_n_anchors",  "Número de anclas (norm.)",   "n_anchors_linking_pct"),
    ]
    ATTR_COMPS = [
        ("w_pci",       "PCI",                         "pci_pct"),
        ("w_cog",       "COG",                         "cog_pct"),
        ("w_tma",       "Tamaño Mercado Accesible",    "accessible_market_size_share_pct"),
        ("w_cma",       "Crecimiento Mercado Accesible", "accessible_market_growth_5y_pct"),
    ]
    SECTOR_COLORS_OPP = {
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

    # Techo dinámico del slider RCA = ceil(max) + 1
    rca_max_data = float(pd.to_numeric(df["raw_rca_cba"], errors="coerce").max() or 0.0)
    RCA_MAX_UI = int(np.ceil(rca_max_data)) + 1

    sector_options = sorted(df["sector"].dropna().astype(str).unique().tolist())

    hs4_labels_df = (
        df[["hs4", "product_name_short"]].copy()
        .assign(hs4=lambda d: d["hs4"].astype(str).str.zfill(4))
        .assign(product_name_short=lambda d: d["product_name_short"].fillna("").astype(str))
    )
    hs4_labels_df["hs4_label"] = hs4_labels_df["hs4"] + " - " + hs4_labels_df["product_name_short"]
    hs4_label_to_code = dict(zip(hs4_labels_df["hs4_label"], hs4_labels_df["hs4"]))

    # Session-state defaults
    defaults = {
        "opp_rca_min": 0.0,
        "opp_rca_max": float(RCA_MAX_UI),
        "opp_tma_min_b": 0.0,          # Tamaño Mercado Accesible mínimo (BUSD)
        "opp_dens_pct_range": (0.0, 1.0),
        "opp_sectors": sector_options,
        "opp_excluded_hs4": [],
        "opp_toggle_market_cagr_pos": False,
        "opp_toggle_export_cagr_pos": False,
        "opp_toggle_accessible_growth_pos": False,
        # Feasibility weights (default = Margen Intensivo)
        "opp_w_rca": 0.00,
        "opp_w_density": 0.50,
        "opp_w_eff": 0.00,
        "opp_w_dai": 0.50,
        "opp_w_dist": 0.00,
        "opp_w_n_anchors": 0.00,
        # Attractiveness weights (default = Margen Intensivo)
        "opp_w_pci": 0.50,
        "opp_w_cog": 0.00,
        "opp_w_tma": 0.25,
        "opp_w_cma": 0.25,
        # Balance
        "opp_balance": 0.50,
        # Display
        "opp_top_n": 60,
        "opp_size_var": "Tamaño Mercado Accesible (B USD)",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    def _apply_preset_margen_intensivo():
        # Algoritmo Growth Lab · Margen Intensivo (0.7<RCA<10) · Top 15
        # Root filters: Mercado Accesible ≥ 500M USD + CAGR 5y > 0
        st.session_state["opp_rca_min"] = 0.7
        st.session_state["opp_rca_max"] = 10.0
        st.session_state["opp_tma_min_b"] = 0.5
        st.session_state["opp_dens_pct_range"] = (0.0, 1.0)
        st.session_state["opp_sectors"] = sector_options
        st.session_state["opp_excluded_hs4"] = []
        st.session_state["opp_toggle_market_cagr_pos"] = False
        st.session_state["opp_toggle_export_cagr_pos"] = False
        st.session_state["opp_toggle_accessible_growth_pos"] = True
        # Feasibility · DAI 50 + Densidad 50
        st.session_state["opp_w_rca"] = 0.00
        st.session_state["opp_w_density"] = 0.50
        st.session_state["opp_w_eff"] = 0.00
        st.session_state["opp_w_dai"] = 0.50
        st.session_state["opp_w_dist"] = 0.00
        st.session_state["opp_w_n_anchors"] = 0.00
        # Attractiveness · PCI 50 + Crec 25 + Tamaño 25 (sin COG)
        st.session_state["opp_w_pci"] = 0.50
        st.session_state["opp_w_cog"] = 0.00
        st.session_state["opp_w_tma"] = 0.25
        st.session_state["opp_w_cma"] = 0.25
        st.session_state["opp_balance"] = 0.50
        st.session_state["opp_top_n"] = 15

    def _apply_preset_margen_extensivo():
        # Algoritmo Growth Lab · Margen Extensivo (0<RCA<0.25) · Top 30
        # Root filters: Mercado Accesible ≥ 500M USD + CAGR 5y > 0
        st.session_state["opp_rca_min"] = 0.0
        st.session_state["opp_rca_max"] = 0.25
        st.session_state["opp_tma_min_b"] = 0.5
        st.session_state["opp_dens_pct_range"] = (0.0, 1.0)
        st.session_state["opp_sectors"] = sector_options
        st.session_state["opp_excluded_hs4"] = []
        st.session_state["opp_toggle_market_cagr_pos"] = False
        st.session_state["opp_toggle_export_cagr_pos"] = False
        st.session_state["opp_toggle_accessible_growth_pos"] = True
        # Feasibility · DAI 50 + Número de anclas 50 (imagen: sin densidad/distancia)
        st.session_state["opp_w_rca"] = 0.00
        st.session_state["opp_w_density"] = 0.00
        st.session_state["opp_w_eff"] = 0.00
        st.session_state["opp_w_dai"] = 0.50
        st.session_state["opp_w_dist"] = 0.00
        st.session_state["opp_w_n_anchors"] = 0.50
        # Attractiveness · PCI 35 + COG 35 + Crec 15 + Tamaño 15
        st.session_state["opp_w_pci"] = 0.35
        st.session_state["opp_w_cog"] = 0.35
        st.session_state["opp_w_tma"] = 0.15
        st.session_state["opp_w_cma"] = 0.15
        st.session_state["opp_balance"] = 0.50
        st.session_state["opp_top_n"] = 30

    def _reset_opp_filters():
        for k, v in defaults.items():
            st.session_state[k] = v

    # ------------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.header("Perfiles predefinidos")
        st.button("Margen Intensivo", on_click=_apply_preset_margen_intensivo,
                  use_container_width=True,
                  help="Algoritmo Growth Lab · RCA ∈ [0.7, 10] · "
                       "Root: Mercado Accesible ≥ 500M USD + CAGR > 0 · "
                       "Feasibility (DAI 50 · Densidad 50) · "
                       "Attractiveness (PCI 50 · Crec 25 · Tamaño 25) · "
                       "Balance 0.50 · Top 15")
        st.button("Margen Extensivo", on_click=_apply_preset_margen_extensivo,
                  use_container_width=True,
                  help="Algoritmo Growth Lab · RCA ∈ [0, 0.25] · "
                       "Root: Mercado Accesible ≥ 500M USD + CAGR > 0 · "
                       "Feasibility (DAI 50 · # anclas 50) · "
                       "Attractiveness (PCI 35 · COG 35 · Crec 15 · Tamaño 15) · "
                       "Balance 0.50 · Top 30")
        st.button("Reiniciar filtros", on_click=_reset_opp_filters, use_container_width=True)

        st.header("Filtros de universo")
        rca_min = st.number_input(
            "RCA raw mínima",
            min_value=0.0,
            max_value=float(RCA_MAX_UI),
            value=float(st.session_state["opp_rca_min"]),
            step=0.05,
            format="%.3f",
            key="opp_rca_min",
        )
        rca_max = st.number_input(
            "RCA raw máxima",
            min_value=float(rca_min),
            max_value=float(RCA_MAX_UI),
            value=max(float(st.session_state["opp_rca_max"]), float(rca_min)),
            step=0.05,
            format="%.3f",
            key="opp_rca_max",
            help=f"Techo dinámico basado en el RCA máximo de Córdoba (~{rca_max_data:.1f}).",
        )
        tma_max_data = float(df["accessible_market_size_b"].max() or 0.0)
        tma_min_b = st.number_input(
            "Tamaño Mercado Accesible mínimo (B USD)",
            min_value=0.0,
            max_value=max(tma_max_data, 0.1),
            value=float(st.session_state["opp_tma_min_b"]),
            step=0.1,
            format="%.2f",
            key="opp_tma_min_b",
        )
        dens_range = st.slider(
            "Percentil de densidad — Córdoba",
            0.0, 1.0,
            tuple(st.session_state["opp_dens_pct_range"]),
            0.01,
            key="opp_dens_pct_range",
            help="Filtra HS4 por la posición de Córdoba en la distribución de density del HS4.",
        )
        selected_sectors = st.multiselect(
            "Sectores Atlas",
            options=sector_options,
            default=st.session_state["opp_sectors"],
            key="opp_sectors",
        )
        excluded_labels = st.multiselect(
            "Excluir HS4 específicos",
            options=list(hs4_label_to_code.keys()),
            default=st.session_state["opp_excluded_hs4"],
            key="opp_excluded_hs4",
        )
        excluded_hs4_codes = {hs4_label_to_code[l] for l in excluded_labels}

        st.header("Filtros de crecimiento")
        st.toggle("Market growth 5y > 0 (mundial)", key="opp_toggle_market_cagr_pos")
        st.toggle("Accessible market growth 5y > 0", key="opp_toggle_accessible_growth_pos")

        st.header("Balance estratégico")
        st.slider(
            "Factibilidad ← 0    |    Atractivo → 1",
            0.0, 1.0, float(st.session_state["opp_balance"]), 0.05,
            key="opp_balance",
        )

        st.header("Pesos — Factibilidad")
        st.caption("Componentes del índice (percentiles). Suman implícitamente al normalizar.")
        for key, label, _col in FEAS_COMPS:
            st.slider(label, 0.0, 1.0, float(st.session_state[f"opp_{key}"]),
                      0.05, key=f"opp_{key}")

        st.header("Pesos — Atractivo")
        for key, label, _col in ATTR_COMPS:
            st.slider(label, 0.0, 1.0, float(st.session_state[f"opp_{key}"]),
                      0.05, key=f"opp_{key}")

    # ------------------------------------------------------------------------
    # Filtro + Cálculo de índices
    # ------------------------------------------------------------------------
    flt = df.copy()
    flt = flt[(flt["raw_rca_cba"] >= rca_min) & (flt["raw_rca_cba"] <= rca_max)]
    flt = flt[flt["accessible_market_size_b"].fillna(0) >= tma_min_b]
    flt = flt[
        (flt["density_pct_cba"].fillna(0) >= dens_range[0])
        & (flt["density_pct_cba"].fillna(0) <= dens_range[1])
    ]
    if selected_sectors:
        flt = flt[flt["sector"].isin(selected_sectors)]
    if excluded_hs4_codes:
        flt = flt[~flt["hs4"].isin(excluded_hs4_codes)]
    if st.session_state["opp_toggle_market_cagr_pos"]:
        flt = flt[flt["market_growth_5y"].fillna(0) > 0]
    if st.session_state["opp_toggle_accessible_growth_pos"]:
        flt = flt[flt["accessible_market_growth_5y"].fillna(0) > 0]

    def _weighted(frame, comps, weight_prefix):
        cols, weights = [], []
        for key, _label, col in comps:
            w = float(st.session_state[f"opp_{key}"])
            if w > 0:
                cols.append(col)
                weights.append(w)
        if not cols:
            return pd.Series(0.0, index=frame.index)
        weight_sum = sum(weights)
        arr = np.column_stack([pd.to_numeric(frame[c], errors="coerce").fillna(0).to_numpy() for c in cols])
        return pd.Series((arr * np.array(weights)).sum(axis=1) / weight_sum, index=frame.index)

    flt["feasibility_index"] = _weighted(flt, FEAS_COMPS, "opp_")
    flt["attractiveness_index"] = _weighted(flt, ATTR_COMPS, "opp_")
    bal = float(st.session_state["opp_balance"])
    flt["combined_score"] = (1 - bal) * flt["feasibility_index"] + bal * flt["attractiveness_index"]

    if flt.empty:
        st.warning("Ningún HS4 cumple los filtros actuales. Ajustá el sidebar.")
        st.stop()

    # ------------------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("HS4 mostrados", f"{len(flt):,} / {len(df):,}")
    k2.metric("Feasibility media", f"{flt['feasibility_index'].mean():.3f}")
    k3.metric("Attractiveness media", f"{flt['attractiveness_index'].mean():.3f}")
    k4.metric("Score combinado top", f"{flt['combined_score'].max():.3f}")

    # ------------------------------------------------------------------------
    # Scatter Feasibility × Attractiveness
    # ------------------------------------------------------------------------
    size_choices = {
        "Tamaño Mercado Accesible (B USD)": "accessible_market_size_b",
        "Exportación Córdoba (USD)": "cordoba_value_usd",
        "PCI": "pci",
        "RCA raw Córdoba": "raw_rca_cba",
        "Distancia recorrida (%)": "distance_travelled_pct",
    }
    size_label = st.selectbox(
        "Tamaño del punto",
        options=list(size_choices.keys()),
        index=list(size_choices.keys()).index(st.session_state["opp_size_var"])
              if st.session_state["opp_size_var"] in size_choices else 0,
        key="opp_size_var",
    )
    size_col = size_choices[size_label]
    size_raw = pd.to_numeric(flt[size_col], errors="coerce").fillna(0).clip(lower=0)
    size_min, size_max = 6, 26
    if size_raw.max() > size_raw.min():
        flt["_dot_size"] = size_min + ((size_raw - size_raw.min()) / (size_raw.max() - size_raw.min())) * (size_max - size_min)
    else:
        flt["_dot_size"] = (size_min + size_max) / 2

    fig = px.scatter(
        flt,
        x="feasibility_index", y="attractiveness_index",
        color="sector", color_discrete_map=SECTOR_COLORS_OPP,
        size="_dot_size", size_max=size_max,
        hover_name="product_name_short",
        hover_data={
            "hs4": True,
            "raw_rca_cba": ":.3f",
            "pci": ":.2f",
            "cog_cba": ":.2f",
            "density_cba": ":.3f",
            "density_pct_cba": ":.2f",
            "dai_percentile": ":.1f",
            "distance_travelled": ":.0f",
            "accessible_market_size_b": ":.2f",
            "accessible_market_growth_5y": ":.2%",
            "combined_score": ":.3f",
            "_dot_size": False,
            "sector": False,
        },
        labels={
            "feasibility_index": "Factibilidad (Feasibility)",
            "attractiveness_index": "Atractivo (Attractiveness)",
            "raw_rca_cba": "RCA raw",
            "density_pct_cba": "Density pct",
            "dai_percentile": "DAI pct",
            "distance_travelled": "Distancia recorrida (km)",
            "accessible_market_size_b": "TMA (B USD)",
            "accessible_market_growth_5y": "CMA (5y)",
            "sector": "Sector Atlas",
        },
        template="plotly_white",
        title=f"Feasibility × Attractiveness  ·  {len(flt):,} HS4 mostrados",
    )
    fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0.4, color="#1f2937")))
    fig.update_layout(margin=dict(t=60, l=20, r=20, b=20), height=580, legend_title="Sector Atlas")
    # Diagonal referencia
    dmin = float(min(flt["feasibility_index"].min(), flt["attractiveness_index"].min()))
    dmax = float(max(flt["feasibility_index"].max(), flt["attractiveness_index"].max()))
    fig.add_shape(type="line", x0=dmin, y0=dmin, x1=dmax, y1=dmax,
                  line=dict(color="rgba(180,60,60,0.5)", width=1.5, dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------------
    # Tabla top-N
    # ------------------------------------------------------------------------
    st.subheader("Ranking de HS4 — top oportunidades")
    top_n = st.slider("Filas a mostrar", 10, 300, int(st.session_state["opp_top_n"]), 10, key="opp_top_n")

    table = flt.sort_values("combined_score", ascending=False).head(top_n).copy()
    table.insert(0, "rank", range(1, len(table) + 1))

    display = table[[
        "rank", "hs4", "product_name_short", "sector",
        "raw_rca_cba", "cordoba_value_usd", "pci", "cog_cba",
        "density_pct_cba", "dai_percentile", "distance_travelled",
        "accessible_market_size_b", "accessible_market_growth_5y",
        "rank_cordoba", "rank_arg_ex_cba",
        "feasibility_index", "attractiveness_index", "combined_score",
    ]].rename(columns={
        "product_name_short": "Producto",
        "sector": "Sector",
        "raw_rca_cba": "RCA Cba",
        "cordoba_value_usd": "Exp. Cba (USD)",
        "pci": "PCI",
        "cog_cba": "COG",
        "density_pct_cba": "Density pct",
        "dai_percentile": "DAI pct",
        "distance_travelled": "Dist. (km)",
        "accessible_market_size_b": "TMA (B USD)",
        "accessible_market_growth_5y": "CMA 5y",
        "rank_cordoba": "Rank Cba",
        "rank_arg_ex_cba": "Rank ARG-Cba",
        "feasibility_index": "Feas.",
        "attractiveness_index": "Attr.",
        "combined_score": "Score",
    })

    st.dataframe(
        display,
        use_container_width=True, hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
            "hs4": st.column_config.TextColumn("HS4", width="small"),
            "Producto": st.column_config.TextColumn("Producto", width="medium"),
            "Sector": st.column_config.TextColumn("Sector"),
            "RCA Cba": st.column_config.NumberColumn("RCA Cba", format="%.3f"),
            "Exp. Cba (USD)": st.column_config.NumberColumn("Exp. Cba (USD)", format="$%.0f"),
            "PCI": st.column_config.NumberColumn("PCI", format="%.2f"),
            "COG": st.column_config.NumberColumn("COG", format="%.2f"),
            "Density pct": st.column_config.NumberColumn("Density pct", format="%.2f"),
            "DAI pct": st.column_config.NumberColumn("DAI pct", format="%.1f"),
            "Dist. (km)": st.column_config.NumberColumn("Dist. (km)", format="%.0f"),
            "TMA (B USD)": st.column_config.NumberColumn("TMA (B USD)", format="%.2f"),
            "CMA 5y": st.column_config.NumberColumn("CMA 5y", format="%.2%%"),
            "Rank Cba": st.column_config.NumberColumn("Rank Cba", format="%.0f"),
            "Rank ARG-Cba": st.column_config.NumberColumn("Rank ARG-Cba", format="%.0f"),
            "Feas.": st.column_config.NumberColumn("Feas.", format="%.3f"),
            "Attr.": st.column_config.NumberColumn("Attr.", format="%.3f"),
            "Score": st.column_config.NumberColumn("Score", format="%.3f"),
        },
    )

    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar ranking (CSV)", csv_bytes,
                       "cordoba_oportunidades_ranking.csv", "text/csv")

    # ------------------------------------------------------------------------
    # Treemap de oportunidades seleccionadas (top-N)
    # ------------------------------------------------------------------------
    st.subheader("Treemap — top oportunidades seleccionadas")
    st.caption(
        f"Muestra los top {len(table)} HS4 del ranking, agrupados por sector Atlas. "
        "Elegí la variable que define el tamaño de cada baldosa y la variable de color."
    )

    tm_size_choices = {
        "Score combinado":                   "combined_score",
        "Factibilidad":                      "feasibility_index",
        "Atractivo":                         "attractiveness_index",
        "Tamaño Mercado Accesible (B USD)":  "accessible_market_size_b",
        "Exportación Córdoba (USD)":         "cordoba_value_usd",
        "PCI":                               "pci",
    }
    tm_color_choices = ["Sector Atlas", "PCI (raw)"]
    PCI_COLOR_SCALE = [
        [0.000, "rgb(227, 159, 96)"],
        [0.279, "rgb(231, 173, 120)"],
        [0.339, "rgb(235, 188, 143)"],
        [0.398, "rgb(240, 202, 168)"],
        [0.448, "rgb(244, 217, 191)"],
        [0.494, "rgb(248, 231, 215)"],
        [0.495, "rgb(192, 228, 225)"],
        [0.534, "rgb(154, 211, 207)"],
        [0.571, "rgb(116, 195, 189)"],
        [0.607, "rgb(77, 178, 171)"],
        [0.662, "rgb(40, 162, 153)"],
        [1.000, "rgb(2, 146, 135)"],
    ]

    c_tm1, c_tm2 = st.columns(2)
    with c_tm1:
        st.session_state.setdefault("opp_tm_size", "Tamaño Mercado Accesible (B USD)")
        tm_size_label = st.selectbox(
            "Tamaño de la baldosa",
            options=list(tm_size_choices.keys()),
            index=list(tm_size_choices.keys()).index(st.session_state["opp_tm_size"])
                  if st.session_state["opp_tm_size"] in tm_size_choices else 0,
            key="opp_tm_size",
        )
    with c_tm2:
        st.session_state.setdefault("opp_tm_color", "Sector Atlas")
        tm_color_label = st.selectbox(
            "Color de la baldosa",
            options=tm_color_choices,
            index=tm_color_choices.index(st.session_state["opp_tm_color"])
                  if st.session_state["opp_tm_color"] in tm_color_choices else 0,
            key="opp_tm_color",
        )

    treemap_df = table.copy()
    tm_size_col = tm_size_choices[tm_size_label]
    tm_size_values = pd.to_numeric(treemap_df[tm_size_col], errors="coerce").fillna(0)
    # Some size cols can be non-positive; treemap needs positive values → floor at a tiny eps
    tm_size_values = tm_size_values.clip(lower=0)
    if tm_size_values.sum() == 0:
        # Fallback to score if the picked col is all-zero for the filtered set
        tm_size_values = pd.to_numeric(treemap_df["combined_score"], errors="coerce").fillna(0).clip(lower=0)
    treemap_df["_size_val"] = tm_size_values

    def _wrap(text: str, width: int = 20) -> str:
        words = str(text).split()
        if not words:
            return str(text)
        lines, cur = [], words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) <= width:
                cur = f"{cur} {w}"
            else:
                lines.append(cur); cur = w
        lines.append(cur)
        return "<br>".join(lines)

    treemap_df["product_label"] = (
        treemap_df["hs4"].astype(str).str.zfill(4)
        + " - " + treemap_df["product_name_short"].fillna("").astype(str)
    )
    treemap_df["product_label_wrapped"] = treemap_df["product_label"].map(_wrap)

    common_hover = {
        "combined_score": ":.3f",
        "feasibility_index": ":.3f",
        "attractiveness_index": ":.3f",
        "raw_rca_cba": ":.3f",
        "pci": ":.2f",
        "accessible_market_size_b": ":.2f",
        "accessible_market_growth_5y": ":.2%",
        "product_label": True,
        "sector": False,
        "product_label_wrapped": False,
        "_size_val": False,
    }
    # Mercado accesible agregado del top-N mostrado (siempre visible en el
    # título, independiente de la variable de tamaño seleccionada).
    _ma_total_b = float(
        pd.to_numeric(treemap_df["accessible_market_size_b"], errors="coerce")
        .fillna(0).sum()
    )
    tm_kwargs = dict(
        data_frame=treemap_df,
        path=["sector", "product_label_wrapped"],
        values="_size_val",
        hover_data=common_hover,
        title=(
            f"Top {len(treemap_df)} oportunidades  ·  tamaño = {tm_size_label}  ·  color = {tm_color_label}"
            f"<br><span style='font-size:0.8em'>Mercado accesible agregado (top {len(treemap_df)}) = "
            f"USD {_ma_total_b:,.2f} mil M</span>"
        ),
    )

    if tm_color_label == "PCI (raw)":
        fig_tm = px.treemap(
            color="pci",
            color_continuous_scale=PCI_COLOR_SCALE,
            range_color=(-2.0, 2.0),
            labels={"pci": "PCI"},
            **tm_kwargs,
        )
        text_color = "#1f2937"
        fig_tm.update_layout(
            coloraxis_colorbar=dict(
                title=dict(text="PCI (raw)", side="top"),
                orientation="h", x=0.5, xanchor="center", y=-0.14, yanchor="top",
                len=0.6, thickness=14,
                tickmode="array", tickvals=[-2, -1, 0, 1, 2],
                ticktext=["≤ -2", "-1", "0", "1", "≥ 2"],
            ),
            margin=dict(t=60, l=10, r=10, b=80),
        )
    else:
        fig_tm = px.treemap(
            color="sector",
            color_discrete_map=SECTOR_COLORS_OPP,
            **tm_kwargs,
        )
        text_color = "#ffffff"
        fig_tm.update_layout(margin=dict(t=60, l=10, r=10, b=10))

    fig_tm.update_traces(
        textinfo="label",
        textfont=dict(size=15, color=text_color),
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
    )
    fig_tm.update_layout(height=620)
    st.plotly_chart(fig_tm, use_container_width=True)

    # ------------------------------------------------------------------------
    # Nota metodológica
    # ------------------------------------------------------------------------
    with st.expander("Nota metodológica"):
        st.markdown("""
**Factibilidad (Feasibility)** — promedio ponderado (por pesos del sidebar) de percentiles:
- **RCA transformada** (`rca/(rca+1)`) de Córdoba en el panel Córdoba+BACI.
- **Density (percentil)** — cercanía de Córdoba al producto en el espacio de productos, recomputado dentro del panel Córdoba+BACI.
- **Effective exporters (invertido)** — inversa del percentil de número efectivo de exportadores; más competencia mundial = menos feasibility.
- **DAI percentil** — heredado de Argentina (ARG_Dashboard_V2).
- **Distancia recorrida (percentil)** — producto que viaja lejos globalmente = mercado accesible más amplio.

**Atractivo (Attractiveness)** — promedio ponderado de percentiles:
- **PCI** — complejidad del producto en el panel Córdoba+BACI.
- **COG** — Complexity Outlook Gain de Córdoba en el HS4.
- **TMA (Tamaño Mercado Accesible)** — participación del HS4 en el mercado accesible total de Argentina (heredado).
- **CMA (Crecimiento Mercado Accesible)** — CAGR 5 años del mercado accesible (heredado).

**Combined Score** = (1 − balance) × Feasibility + balance × Attractiveness.

**Presets (Algoritmo Growth Lab · Identificación de Oportunidades)**:

Filtros raíz aplicados a ambos márgenes:
- **Mercado Accesible ≥ 500M USD**
- **Mercado Accesible CAGR 5y > 0**

- **Margen Intensivo** — RCA ∈ [0.7, 10] · Top 15 · Feas (DAI 50 · Densidad 50) · Attr (PCI 50 · TMA 25 · CMA 25) · Balance 0.50.
- **Margen Extensivo** — RCA ∈ [0, 0.25] · Top 30 · Feas (DAI 50 · # anclas 50) · Attr (PCI 35 · COG 35 · TMA 15 · CMA 15) · Balance 0.50.

**Número de anclas** en Margen Extensivo = `n_anchors_linking` normalizado a [0, 1] (min-max sobre `cordoba_candidates_ranked.csv`). Cuenta cuántas anclas de Córdoba referencian al HS4 candidato en su top-K de proximidad.

**Fuentes**:
- Complejidad + RCA + PCI + COG + density: nuestro panel `complexity_cordoba_full.csv` (BACI 2020-2024 con Córdoba como location adicional).
- Métricas de mercado y DAI: `data/reference/arg_v2_inherited/opportunity_metrics_hs4_arg.csv` (snapshot desde ARG_Dashboard_V2 del Growth Lab).
- Nombres y sectores HS4: `data/reference/arg_v2_inherited/hs92_4digits.csv`.
        """)


inicio = st.Page(page_inicio, title="Inicio", icon=":material/home:", default=True)
expo_producto = st.Page(page_exportaciones_producto, title="Exportaciones por producto",
                        icon=":material/analytics:")
analisis = st.Page(page_analisis, title="Análisis de Proximidad", icon=":material/insights:")
oportunidades = st.Page(page_oportunidades_cordoba, title="Oportunidades",
                        icon=":material/target:")
mercado = st.Page(page_mercado_accesible, title="Mercado Accesible por Producto", icon=":material/public:")
firmas = st.Page(page_firmas, title="Firmas y Rubros [Legacy]", icon=":material/business:")
st.navigation([inicio, expo_producto, analisis, oportunidades, mercado, firmas]).run()
