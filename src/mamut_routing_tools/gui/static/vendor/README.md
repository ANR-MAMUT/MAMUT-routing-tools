# Vendored web libraries

Self-hosted so the GUI makes no third-party asset requests besides the basemap tiles. Each directory keeps the upstream license file. The `maplibre-gl/` files are byte-identical to the MAMUT-routing website's `site_assets/vendor/` copy (the website repository checks this).

| Directory | Library | Version | License | Source |
|---|---|---|---|---|
| `maplibre-gl/` | MapLibre GL JS (UMD build `dist/maplibre-gl.js` + `dist/maplibre-gl.css`) | 5.24.0 | BSD-3-Clause | https://github.com/maplibre/maplibre-gl-js |
| `fonts/InterVariable.woff2` | Inter (variable font) | — | OFL-1.1 | https://github.com/rsms/inter |

MapLibre GL JS stays on the 5.x line: 6.x ships ES modules only, and the GUI loads it as a classic script that exposes the `maplibregl` global (read by the shared `mamut-map.js`).
