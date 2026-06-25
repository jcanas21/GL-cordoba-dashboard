# Córdoba — Anchored Proximity Dashboard

Interactive Streamlit app that explores diversification opportunities for
Córdoba, Argentina, by ranking candidate products on **feasibility** (DAI,
geographic distance travelled, number of anchors) and **attractiveness**
(PCI, accessible market size and growth).

Anchors are the HS4 (HS 1992) codes where Córdoba has firm-evidenced presence
(184 codes, sourced from `exportadoresdecordoba.com` and other registered
chambers). An OPEX threshold slider re-derives the anchor set live, and any
evidenced HS4 that falls below the threshold can resurface as a candidate of
the surviving anchors — flagged in the table as **`posible_ancla = 1`**.

## Run

```bash
pip install -r requirements.txt
streamlit run cordoba_anchored_proximity_app.py
```

Then open <http://localhost:8501>.

## Layout

```
cordoba-dashboard/
├── README.md
├── requirements.txt
├── cordoba_anchored_proximity_app.py   # Streamlit entry point
├── hs4_names_es.py                     # HS4 -> Spanish short names (~1,240 entries)
└── data/
    ├── cordoba_anchored_proximity.csv          # anchor-candidate proximity links + enrichment
    ├── hs4_presence_by_opex_threshold.csv      # the 184 evidenced HS4 with OPEX values
    ├── umap_layout_hs92.csv                    # UMAP coordinates for the product space
    ├── hs92_product_year_4.csv                 # BACI 2024 trade values for dot sizing
    ├── product_space_clusters.csv              # cluster -> hex colour
    └── product_hs92.csv                        # HS4 -> Atlas English product names
```

## What the panels show

| Panel | Driven by |
|---|---|
| **KPI strip** | Anchor universe size · # links · # unique anchors · # unique candidates after filters |
| **Product space** | Plotly scatter — anchors coloured by cluster; everything else grey background |
| **Sankey** | Anchors → top-N candidates, link width = proximity, coloured by anchor sector |
| **Candidate ranking table** | One row per candidate after filters, sortable, downloadable as CSV |
| **Treemap** | Top-N candidates sized by accessible market size, coloured by sector or PCI |

## Controls (sidebar)

- **Preset "Top perfil seleccionado"** — applies a curated screen: $10M OPEX
  threshold, $0.5 B minimum accessible market, growth > 0, top-10 proximity
  per anchor, HS sections 1/2/3 dropped, natural-resource HS4 excluded,
  strategic balance 0.70 (attractiveness-leaning), 30 candidates.
- **OPEX threshold** — single source of truth for the anchor set. Moving it
  re-derives anchors → links → candidates → all panels.
- Sector / HS section multi-selects (anchor side, candidate side).
- HS4 exclusions.
- Proximity rank range, minimum accessible market size, "growth > 0" toggle.
- **Strategic balance dial** + per-component sliders inside Feasibility and
  Attractiveness. Weights are re-normalised to sum 1 within each dimension.
- **Top-N** drives both the treemap and the ranking table.
- **Treemap colour** — sector or PCI.

## Methodology — quick read

- **Anchor set**: HS4 (HS 1992) codes where firm-level evidence ties Córdoba
  exporters to that product, filtered live by the OPEX threshold. Source:
  registry scrapes + chamber data, manually curated.
- **Candidate set**: every other HS4 (including evidenced HS4 below the OPEX
  threshold — these get `posible_ancla = 1`).
- **Proximity matrix**: computed once on BACI 2020–2024 averaged exports, via
  the [`ecomplexity`](https://github.com/cid-harvard/py-ecomplexity) package
  with `presence_test = "manual"` after an `rca / (rca + 1)` transform.
- **Per-anchor cut**: keep the top 1 % of candidates by proximity for each
  anchor — that's the link set the dashboard navigates.
- **Feasibility**: weighted average of DAI percentile, distance-travelled
  percentile (higher = product travels farther globally = more tradeable),
  and a normalised anchor count. All re-normalised within the current filter.
- **Attractiveness**: weighted average of PCI, accessible market size, and
  5-year market growth. COG is intentionally excluded.
- **Combined score**: `(1 − balance) · feasibility + balance · attractiveness`.

## Data provenance

- BACI HS92 trade panel (CEPII) — 2020-2024, averaged.
- DAI, accessible market size & growth — Growth Lab Argentina opportunity
  metrics (HS4, 2024).
- PCI — Growth Lab Argentina complexity output (HS4, 2024).
- Product space UMAP layout + sector colours — Growth Lab Atlas conventions.
- Anchor universe — firm-level evidence retrieved from
  `exportadoresdecordoba.com` and corroborated by chamber sites.

## License

Internal research artefact. Contact the maintainer before redistributing the
underlying data.
