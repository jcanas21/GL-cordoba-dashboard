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

# ---------------------------------------------------------------------------
# Loaders + page functions
# ---------------------------------------------------------------------------
@st.cache_data
def load_firms_data(_signature: str = ""):
    """Union 03 (curated) + 04 (registry-keyword) firm-HS4 evidence."""
    curated = pd.read_csv(
        DATA_DIR / "03_firm_hs4_evidence.csv",
        dtype={"firm_id": "string", "hs4": str, "supports_top50_line": str},
    )
    curated["hs4"] = curated["hs4"].astype(str).str.zfill(4)
    curated["supports_top50_line"] = curated["supports_top50_line"].astype(str).str.strip()

    reg = pd.read_csv(
        DATA_DIR / "04_registry_full.csv",
        dtype={"firm_id": "string", "code": str, "opex_line": str},
    )
    reg = reg[reg["classification"] == "HS4"].copy()
    reg["code"] = reg["code"].astype(str).str.zfill(4)
    reg["opex_line"] = reg["opex_line"].astype(str).str.strip()

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
    })

    reg_norm = pd.DataFrame({
        "firm_id": reg["firm_id"].astype(str),
        "firm_name": reg["firm_name"].astype(str),
        "razon_social": reg["razon_social"].astype(str),
        "hs4": reg["code"],
        "rubro_indec": reg["opex_line"],
        "rubro_indec_nombre": reg["opex_line_name"].astype(str),
        "attribution_type": reg["attribution_type"].astype(str),
        "confidence": reg["confidence"].astype(str),
        "evidence_text": reg["evidence"].astype(str),
        "evidence_url": reg["source_url"].astype(str),
        "source_url": reg["source_url"].astype(str),
        "evidence_layer": "registry-keyword",
    })

    pair_key = curated_norm["firm_id"] + "|" + curated_norm["hs4"]
    reg_norm_dedup = reg_norm[~(reg_norm["firm_id"] + "|" + reg_norm["hs4"]).isin(set(pair_key))].copy()
    firm_ev = pd.concat([curated_norm, reg_norm_dedup], ignore_index=True)

    opex = pd.read_csv(
        DATA_DIR / "exportaciones_opex_cordoba.csv",
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
def load_accessible_market(_signature: str = ""):
    df = pd.read_csv(DATA_DIR / "accessible_market_arg.csv", dtype={"hs92": str})
    df["hs92"] = df["hs92"].str.zfill(4)
    df["continente"] = df["iso3_d"].map(_ISO3_TO_CONTINENT).fillna("Otros")
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
partiendo del conjunto de productos **HS4 (HS 1992)** donde la provincia ya
tiene presencia exportadora evidenciada por firmas reales. El núcleo
metodológico es la **proximidad en el espacio de productos** (Hidalgo, Hausmann
et al.) construida con datos BACI 2020–2024 y la implementación
[`ecomplexity`](https://github.com/cid-harvard/py-ecomplexity) del Growth Lab.

Cada candidato se rankea combinando dos dimensiones:

- **Factibilidad** — qué tan cerca está el candidato de la base productiva
  actual (proximidad), si su demanda global se alinea con el patrón
  exportador (DAI), y si es un producto que viaja lejos en el mundo
  (`distance_travelled`).
- **Atractivo** — qué tan complejo es el producto (PCI), qué tan grande
  es su mercado accesible, y a qué tasa crece ese mercado.

Un dial estratégico balancea ambas dimensiones, y los pesos internos de cada
componente son configurables en el sidebar.
    """)

    st.markdown("""
### Tres páginas

- **Inicio** (acá): contexto, glosario, fórmulas.
- **Análisis de Proximidad**: el tablero interactivo principal con todos
  los filtros, el espacio de productos, el Sankey, la tabla de
  candidatos rankeados y el treemap.
- **Firmas y Rubros**: catálogo de las firmas con evidencia explícita de
  exportar productos ancla, con el rubro INDEC al que se atribuyen y el
  monto de exportaciones del rubro en OPEX.
- **Mercado Accesible por Producto**: para cada uno de los top-30
  candidatos del preset Recomendado, la composición geográfica del
  mercado accesible (países destino con sus importaciones).
    """)

    st.subheader("De dónde salen los 125 HS4 anclas")
    st.markdown(r"""
El set de **125 HS4 anclas** es el resultado de una cadena de filtros
sobre el registro provincial. Cada paso descarta o consolida firmas hasta
quedarse con lo verificable:

| Etapa | # firmas | # HS4 |
|---|---|---|
| 1. Registro Procórdoba scrapeado | 2.678 | — |
| 2. Menos servicios (EBOPS) — bancos, consultoras, marketing… | −1.035 | — |
| 3. Firmas de bienes con `products_text` que matchea al menos un pattern del script 07 | **943** | **114** |
| 4. Firmas curadas manualmente (URL + evidencia textual) | 66 | 95 |
| **Unión anclas = HS4 en (3) ∪ (4)** | **~1.000** | **125** |

Las 943 firmas de la etapa 3 se concentran en la canasta real de Córdoba
(soja, maíz, maní, carnes, muebles, polímeros, autopartes…), por eso 943
firmas producen sólo 114 HS4 distintos — muchas firmas comparten el mismo
puñado de productos dominantes.

**Los 125 son ~10% del universo HS 1992** (1.243 códigos). El otro ~40%
se analiza como **candidatos** vía proximidad al set ancla. El ~50%
restante son productos alejados de la base productiva provincial.
    """)

    st.subheader("¿Qué significa `curated` vs `registry-keyword`?")
    st.markdown(r"""
Cada fila firma↔HS4 viene de una de dos capas de evidencia:

**`curated`** (66 firmas, 152 pares en `03_firm_hs4_evidence.csv`)
Un analista humano revisó firma por firma su ficha en el registro, sitio
web, notas de prensa. Para cada par firma-HS4 confirmado registró:
- **texto de evidencia** en prosa (ej. *"Products CHORIZO DE CERDO,
  MORCILLA, SALCHICHA → HS 1601 sausages and similar products"*)
- **URL fuente** específica donde verificó
- **nivel de confianza** (casi siempre `high` — sólo se registra cuando
  hay certeza)

**`registry-keyword`** (~899 firmas adicionales, en `04_registry_full.csv`)
Atribución automática por regex sobre el `products_text` de la firma en
el registro. El pipeline (`scripts/07_full_registry_pass.py`) tiene ~130
patterns tipo `\btrigo\b → HS 1001`, `\baceite de soja\b → HS 1507`,
`\bautomóvil\b → HS 8703`. Si el texto matchea, se asigna el HS4.

| | curated | registry-keyword |
|---|---|---|
| **Quién decide** | Humano con lupa | Regex sobre texto |
| **Puede desambiguar?** | Sí (aceite motor ≠ aceite soja) | No |
| **Cobertura** | 66 firmas curadas a mano | 943 firmas matchean patterns |
| **Costo** | Alto (minutos por firma) | Cero |

Las dos capas se **unen** para producir el set de anclas (125 HS4). Cada
fila en Página 3 muestra su `Capa` para que puedas auditar el nivel de
evidencia detrás.
    """)

    st.subheader("Glosario")
    st.markdown(r"""
| Variable | Significado |
|---|---|
| **HS4** | Sistema Armonizado a 4 dígitos, revisión 1992 (convención Atlas / Growth Lab). |
| **Ancla** | HS4 donde Córdoba tiene presencia exportadora **evidenciada por firmas reales** (registry + cámaras). 125 HS4 en total (set actual). |
| **Candidato** | HS4 sin presencia evidenciada que aparece en el top-1% de proximidad de al menos un ancla, **o** un HS4 evidenciado cuyo OPEX cayó por debajo del umbral del slider (se flaguea como *posible ancla*). |
| **OPEX** | Exportaciones de Córdoba por rubro INDEC (CCOD_RUBRO), promedio 2023–2025. El slider de umbral OPEX filtra el set de anclas. |
| **Rubro** | "Grandes Rubros / Capítulos" de INDEC (clasificación ICA); 100 rubros en el panel OPEX. **No es lo mismo que NCM ni que Complejos Exportadores Rev. 2018**. |
| **Proximidad** | Probabilidad condicional `min(P(p₁\|p₂), P(p₂\|p₁))` de que un país exporte ambos productos con RCA, suavizada con `rca / (rca + 1)`. Está en [0, 1]. |
| **PCI** *(Product Complexity Index)* | Sofisticación productiva implícita de un HS4 — más alto = más complejo. |
| **DAI** *(Índice de alineación de demanda)* | Qué tanto la demanda externa por un producto se alinea con la canasta exportadora del país. Ver fórmula abajo. |
| **Distancia recorrida** *(distance_travelled)* | Distancia geográfica promedio (km) que recorre cada HS4 a nivel global, ponderada por valor exportado en cada par bilateral. **Atributo del producto** — no depende del país exportador. Ver fórmula abajo. |
| **Mercado accesible** *(accessible_market_size)* | Suma de las importaciones mundiales del producto por parte de los destinos que están dentro de la distancia recorrida del producto, o que ya reciben flujo grande desde el origen. Ver fórmula abajo. |
| **Factibilidad** | Promedio ponderado del DAI, del percentil de distancia recorrida y del número normalizado de anclas del candidato. Todos en [0, 1]. |
| **Atractivo** | Promedio ponderado del PCI, del tamaño del mercado accesible y del crecimiento a 5 años del mercado accesible. |
| **Puntaje combinado** | `(1 − balance) · factibilidad + balance · atractivo`, donde `balance` es el dial estratégico del sidebar. |
| **Posible ancla** | Dummy 1/0: el candidato pertenece al set de HS4 evidenciados pero su OPEX no llegó al umbral. |
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
        DATA_DIR / "cordoba_anchored_proximity.csv",
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
        "Firmas del registro de Córdoba con evidencia de atribución HS4. "
        "Dos capas: curated (rows con evidencia HS4 explícita + URL fuente, "
        "curados a mano) y registry-keyword (atribución vía match de keyword "
        "en products_text, post-fix sin fallback ciego)."
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

    with st.sidebar:
        st.header("Filtros — Firmas")
        hs4_options = sorted(merged["hs4_label"].dropna().astype(str).unique().tolist())
        sel_hs4 = st.multiselect("HS4 ancla", options=hs4_options, default=hs4_options)
        confidence_options = sorted(merged["confidence"].dropna().astype(str).unique().tolist())
        sel_conf = st.multiselect("Confianza", options=confidence_options, default=confidence_options)
        st.caption(
            "Para filtrar por **rubro INDEC** o **capa de evidencia**, clickeá "
            "una baldosa del treemap. Click en el fondo o ESC para limpiar."
        )

    f = merged.copy()
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
                f"Exportaciones OPEX por rubro INDEC (n = {opex_tm['CCOD_RUBRO'].nunique()} rubros | "
                f"OPEX total = {opex_split['opex_avg_m'].sum():,.1f} USD M) "
                f"| tamaño = OPEX (split por sector cuando hay atribución firme; rubros confidenciales/resto van enteros a 'Other')"
            ),
        )
        fig_tm.update_traces(
            textinfo="label",
            textfont=dict(size=18, color="#ffffff"),
            marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
        )
        fig_tm.update_layout(
            margin=dict(t=60, l=10, r=10, b=95),
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
        "evidence_layer", "confidence",
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
| **HS4 ancla** | HS4 (HS 1992) + nombre corto en español al que la firma está atribuida. Todos los HS4 aquí pertenecen al set de 125 anclas evidenciadas. |
| **Rubro INDEC** | Rubro CCOD_RUBRO (clasificación INDEC "Grandes Rubros / Capítulos") al que el HS4 fold-up en el panel OPEX provincial. |
| **Capa** | Origen y grado de curación del vínculo firma↔HS4. **`curated`** (66 firmas, 152 pares): revisión analítica manual — un curador inspeccionó `products_text` + sitio web + notas de prensa y asignó el HS4 explícitamente, dejando el razonamiento en `Evidencia (texto)` y una URL primaria en `Evidencia URL`. Es el subset más confiable. **`registry-keyword`** (~899 firmas): atribución automática por match de regex sobre `products_text` en el pipeline (`scripts/07_full_registry_pass.py`). Sin fallback ciego — si el texto no matchea ningún pattern específico, la firma no recibe HS4. Confiable a nivel individual pero sin curación por analista. |
| **Confianza** | Nivel de certeza sobre el vínculo firma↔HS4. **`high`**: evidencia clara y explícita (texto de producto es un match exacto del HS4, o análisis manual concluyente). **`medium`**: match plausible pero con ambigüedad (ej. firma multiproducto donde el HS4 es sólo uno de varios candidatos). **`low`**: señal débil — históricamente se usaba para atribuciones por rubro amplio; en el pipeline actual sin fallback ciego, este valor debería ser raro. Para la capa `curated` casi todo es `high` porque sólo se registran filas donde hay certeza. |
| **OPEX rubro (USD M, prom 2023-2025)** | Monto exportado por Córdoba en el rubro INDEC, promedio anual 2023-2025 en USD millones. Es del **rubro entero**, no de la firma individual — una firma en un rubro grande no representa necesariamente una porción grande del monto. |
| **Tipo de atribución** | Cómo el HS4 evidenciado en la firma fold-up al rubro INDEC. Ordenados de más a menos preciso: **`clean`** — el rubro mapea limpiamente a 1-2 HS4 (ej. `106B Maíz → HS 1005`), sin ambigüedad; **`named-aggregate`** — rubro nombrado que agrupa varios HS4 del mismo dominio (ej. `313BB Vehículos automóviles terrestres → HS 8702/8703/8704`); **`broad-chapter`** — rubro cubre un capítulo HS entero (ej. `312B Máq. eléctricas → HS 8501-8548`), el HS4 específico viene del lado firm; **`resto`** — rubro residual dentro de un grupo (ej. `107Z Resto semillas y frutos oleaginosos`), cubre HS4 no clasificados en categorías nombradas; **`confidential`** — rubro INDEC censurado por Ley 17.622 (códigos terminados en `899`): el monto agregado está publicado pero la composición interna no, así que el HS4 sólo puede establecerse desde el lado firm. |
| **Evidencia (texto)** | Justificación en prosa del vínculo firma↔HS4. En **`curated`**: texto humano explicando por qué se asignó ese HS4, típicamente citando el producto declarado y su correspondencia con la definición HS4 (ej. *"Products CHORIZO DE CERDO, MORCILLA, SALCHICHA, SALAME → HS 1601 'sausages and similar products'"*). En **`registry-keyword`**: nota generada por el pipeline indicando el pattern regex que matcheó (ej. *"keyword 'trigo' matched in product/description text"*). En ambos casos permite auditar la fila: si el texto no parece justificar el HS4 mostrado, hay razón para dudar de esa atribución. |
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
                help="`curated` = evidencia manual con URL · `registry-keyword` = match de keyword en products_text",
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
- **`curated`** (66 firmas, 152 pares firm-HS4): cada par tiene evidencia
  HS4 manual + URL fuente.
- **`registry-keyword`** (~899 firmas adicionales): atribución vía match
  de keyword en `products_text` (regex sobre términos específicos como
  "soja" → HS 1201, "biodiesel" → HS 3826). Si el products_text no
  matchea ningún pattern, la firma no recibe HS4 — sin fallback ciego.
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
            "Recomendado",
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
    # 1. Derive anchor universe from OPEX threshold + apply filters
    # ---------------------------------------------------------------------------
    # evidenced_set is the firm-evidenced HS4 universe (currently 125 HS4). The anchor universe
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
            anchors=("anchor_hs4", lambda s: " · ".join(sorted(set(s.astype(str).str.zfill(4))))),
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
    # Rubro INDEC + Match lookup (Directo / Residual) per anchor HS4
    _presence_indexed = presence.copy()
    _presence_indexed["hs4"] = _presence_indexed["hs4"].astype(str).str.zfill(4)
    _presence_indexed["primary_ccod_rubro"] = _presence_indexed["primary_ccod_rubro"].astype(str).str.strip()
    rubro_code_lookup = dict(zip(_presence_indexed["hs4"], _presence_indexed["primary_ccod_rubro"]))
    rubro_name_lookup = dict(zip(_presence_indexed["hs4"], _presence_indexed["primary_rubro_name"].astype(str)))

    _attr_map_hover_df = pd.read_csv(
        DATA_DIR / "05_unified_hs4_presence.csv",
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
        text=[_anchor_hover(h, n) for h, n in zip(ank["product_hs92_code"], ank["product_name_es"])],
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
    _anchor_tbl["OPEX rubro (USD M)"] = _anchor_tbl["max_rubro_opex_2023_2025_avg_usd"] / 1e6
    # Match = 'Residual' cuando la atribución OPEX del HS4 es a un rubro
    # confidencial (INDEC Ley 17.622, códigos terminados en 899); 'Directo'
    # cuando el rubro es clean / named-aggregate / broad-chapter / resto.
    # Fuente: attribution_type en 05_unified_hs4_presence.csv, no el
    # primary_ccod_rubro de la presence-file (que muestra el rubro nombrado
    # equivalente aún cuando la data cae en un rubro confidencial).
    _attribution = pd.read_csv(DATA_DIR / "05_unified_hs4_presence.csv", dtype={"hs4": str}, usecols=["hs4","attribution_type"])
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
    }).sort_values("OPEX rubro (USD M)", ascending=False)

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
| **OPEX rubro (USD M)** | Monto exportado por Córdoba en ese rubro, promedio anual 2023-2025 (USD millones). Es del **rubro entero**. |
| **# firmas** | Cantidad de firmas del registro que evidencian el HS4 (curated + registry-keyword combinadas). |
| **Firmas ejemplo** | Sample de hasta 5 nombres de firmas evidenciando este HS4. |

Al mover el slider **Umbral OPEX** del sidebar, la tabla se restringe a los HS4 cuyo rubro cumple el umbral.
        """)

    st.dataframe(
        _anchor_tbl[[
            "HS4", "Producto", "Sector", "CCOD_RUBRO", "Rubro INDEC", "Match",
            "OPEX rubro (USD M)", "# firmas", "Firmas ejemplo",
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
            "OPEX rubro (USD M)": st.column_config.NumberColumn(
                "OPEX rubro (USD M)", format="%.1f",
                help="Monto exportado por Córdoba en el rubro INDEC, promedio 2023-2025.",
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

    with st.expander("Diccionario de columnas — Tabla de candidatos"):
        st.markdown(r"""
| Columna | Significado |
|---|---|
| **Ranking** | Posición del candidato ordenado por Puntaje combinado descendente. |
| **HS4** | Código HS4 del candidato (HS 1992). |
| **Producto** | Nombre corto del HS4 en español (curado, ~1.240 entradas). |
| **Sector** | Sector Atlas / Growth Lab del HS4. |
| **Posible ancla** | Dummy 1/0. `1` = el candidato pertenece al set de 125 HS4 evidenciados pero su OPEX quedó por debajo del umbral del slider — es una ex-ancla reaparecida como candidato. Ver Inicio → glosario. |
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
            "posible_ancla",
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

# ---------------------------------------------------------------------------
# Multi-page navigation
# ---------------------------------------------------------------------------
@st.cache_data
def _recomendado_top_candidates(_signature: str = "", top_n: int = 30) -> list[str]:
    """Return the candidate HS4s (zfilled) that would appear in page 2's
    'Recomendado' preset ranking, top-N by combined score.

    Mirrors _apply_profile("top_candidates"): anchor OPEX ≥ USD 10 M,
    sections 1/2/3 excluded on both sides, natural-resource HS4 excluded,
    proximity rank ∈ [1, 10], market ≥ USD 0.5 B, growth > 0, strategic
    balance 0.50, weights (0.40/0.40/0.20 feas; 0.50/0.25/0.25 attr).
    """
    df, presence, _, _, _, _ = load_data()

    anchor_universe = set(
        presence.loc[
            presence["max_rubro_opex_2023_2025_avg_usd"].fillna(0) >= 10_000_000,
            "hs4",
        ].astype(str).str.zfill(4)
    )
    za = df["anchor_hs4"].astype(str).str.zfill(4)
    zc = df["candidate_hs4"].astype(str).str.zfill(4)
    flt = df[za.isin(anchor_universe) & ~zc.isin(anchor_universe)].copy()
    for col in ("anchor_hs_section_name", "candidate_hs_section_name"):
        flt = flt[~flt[col].astype(str).str.match(r"^[123]\.")]
    flt = flt[~flt["candidate_hs4"].astype(str).str.zfill(4).isin(NATURAL_RESOURCE_HS4)]
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

    # Product universe: top-30 candidates from the 'Recomendado' preset on
    # page 2. Ordered by combined score desc.
    st.caption(
        "Los productos disponibles corresponden a los **top-30 candidatos** que "
        "el preset **Recomendado** de la página 2 (Análisis de Proximidad) "
        "produce con los filtros y pesos por defecto."
    )
    hs4_in_am = set(am["hs92"].unique())
    recomendado_top = _recomendado_top_candidates(
        _data_signature() if "_data_signature" in globals() else ""
    )
    # Preserve Recomendado ranking order, but only keep HS4 with accessible-market data
    product_universe = [h for h in recomendado_top if h in hs4_in_am]
    if not product_universe:
        st.info("El preset Recomendado no produjo candidatos con datos de mercado accesible.")
        return

    # Label as "HS4 - Producto (Spanish)"
    label_by_hs4 = {
        h: f"{h} - {SPANISH_OVERRIDES.get(h, '')}".rstrip(" -")
        for h in product_universe
    }
    labels = [label_by_hs4[h] for h in product_universe]

    default_label = labels[0] if labels else None
    st.selectbox(
        "Producto (HS4)",
        options=labels,
        key="c4_am_product_label",
        index=0 if default_label else None,
    )
    picked_label = st.session_state.get("c4_am_product_label")
    if not picked_label:
        st.info("Sin productos con datos de mercado accesible.")
        return
    picked_hs4 = picked_label.split(" - ", 1)[0]

    sub = am[am["hs92"] == picked_hs4].copy()
    sub["mercado_b"] = sub["total_imports"] / 1e9  # miles de millones USD
    sub["mercado_m"] = sub["total_imports"] / 1e6  # millones USD
    total_b = float(sub["mercado_b"].sum())
    n_dest = int(sub["iso3_d"].nunique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Producto seleccionado", picked_hs4)
    c2.metric("Mercado accesible total (miles de millones USD)", f"{total_b:.1f}")
    c3.metric("Destinos accesibles", f"{n_dest}")

    if sub.empty or total_b == 0:
        st.info("Sin destinos accesibles para este producto.")
        return

    # Treemap — tile per (continente, destino), colored por continente
    sub["participacion"] = sub["mercado_b"] / total_b
    sub["continente"] = sub["iso3_d"].map(_ISO3_TO_CONTINENT).fillna("Otros")
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
            f"Mercado accesible para {picked_label} "
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

    # Table
    tbl = (
        sub.assign(continente=sub["iso3_d"].map(_ISO3_TO_CONTINENT).fillna("Otros"))
        .sort_values("mercado_b", ascending=False)
        [["continente", "iso3_d", "mercado_b", "participacion"]]
        .rename(columns={
            "continente": "Continente",
            "iso3_d": "País (ISO3)",
            "mercado_b": "Mercado accesible (miles de millones USD)",
            "participacion": "Participación",
        })
    )
    with st.expander("Diccionario de columnas — Mercado accesible por país"):
        st.markdown(r"""
| Columna | Significado |
|---|---|
| **Continente** | Región geográfica del país destino (América / Europa / Asia / África / Oceanía). |
| **País (ISO3)** | Código ISO 3166-1 alpha-3 del país destino. |
| **Mercado accesible (miles de millones USD)** | Importaciones totales del país destino para el producto seleccionado en 2024 (BACI). Incluído en el conjunto accesible porque satisface la condición de distancia recorrida o el umbral de flujo existente de USD 100 M desde Argentina. |
| **Participación** | Porcentaje del mercado accesible total del producto que representa ese país destino. |
        """)

    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Continente": st.column_config.TextColumn("Continente"),
            "País (ISO3)": st.column_config.TextColumn("País (ISO3)", width="small"),
            "Mercado accesible (miles de millones USD)": st.column_config.NumberColumn(
                "Mercado accesible (mil M USD)", format="%.3f",
            ),
            "Participación": st.column_config.NumberColumn("Participación", format="%.2f%%"),
        },
    )
    st.download_button(
        "⬇ Descargar tabla (CSV)",
        tbl.to_csv(index=False).encode("utf-8"),
        f"mercado_accesible_{picked_hs4}.csv",
        "text/csv",
    )


inicio = st.Page(page_inicio, title="Inicio", icon=":material/home:", default=True)
analisis = st.Page(page_analisis, title="Análisis de Proximidad", icon=":material/insights:")
firmas = st.Page(page_firmas, title="Firmas y Rubros", icon=":material/business:")
mercado = st.Page(page_mercado_accesible, title="Mercado Accesible por Producto", icon=":material/public:")
st.navigation([inicio, analisis, firmas, mercado]).run()
