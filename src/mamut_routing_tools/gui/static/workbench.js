"use strict";

/* Route colours come from nocturne-tokens.css via nocturne.js — see that file. */
const POI_CATEGORIES = [
  "restaurant", "cafe", "bar", "fast_food", "pub", "school", "university",
  "hospital", "clinic", "pharmacy", "dentist", "doctors", "veterinary",
  "bank", "atm", "post_office", "police", "fire_station", "townhall", "courthouse",
  "library", "theatre", "cinema", "arts_centre", "community_centre", "museum",
  "place_of_worship", "marketplace", "fuel", "charging_station", "car_wash", "parking",
  "bus_station", "taxi", "bicycle_rental", "ferry_terminal", "kindergarten", "college",
  "nightclub", "biergarten", "ice_cream", "food_court", "bench", "drinking_water",
  "toilets", "shower", "shelter", "waste_basket", "recycling",
];
const DEFAULT_POI_CATEGORIES = new Set([
  "restaurant", "cafe", "bar", "fast_food", "pub", "school", "university",
]);

/* Mirrors generation/demands.py: demand_distribution_bounds and
   avg_route_size_bounds. Kept in sync by tests/test_gui_server.py. */
const DEMAND_TYPES = [
  { value: 1, label: "Unitary", detail: "every demand = 1" },
  { value: 2, label: "Small, large variance", detail: "1–10" },
  { value: 3, label: "Small, small variance", detail: "5–10" },
  { value: 4, label: "Large, large variance", detail: "1–100" },
  { value: 5, label: "Large, small variance", detail: "50–100" },
  { value: 6, label: "Quadrant-dependent", detail: "51–100 on one diagonal, 1–50 elsewhere" },
  { value: 7, label: "Few large, many small", detail: "50–100 for ~1.5n/r of them, 1–10 for the rest" },
];
const ROUTE_SIZE_BANDS = [
  { value: 1, label: "Ultra short", detail: "3–5 customers per route" },
  { value: 2, label: "Very short", detail: "5–8" },
  { value: 3, label: "Short", detail: "8–12" },
  { value: 4, label: "Medium", detail: "12–16" },
  { value: 5, label: "Long", detail: "16–25" },
  { value: 6, label: "Very long", detail: "25–50" },
  { value: 7, label: "Ultra long", detail: "50–200" },
];
const DEMAND_LABELS = new Map(DEMAND_TYPES.map((entry) => [entry.value, entry.label]));
const BAND_LABELS = new Map(ROUTE_SIZE_BANDS.map((entry) => [entry.value, entry.label]));
const BULK_CSV_COLUMNS = [
  "problemType", "city", "nCustomers", "demandType", "avgRouteSize", "method", "seed",
  "depotMode", "customerMode", "twMethod", "onlyIntersections", "clusterSeeds",
  "clusterDecayMeters", "hybridPoiShare", "categories",
];
const BULK_INT_FIELDS = new Set(["nCustomers", "demandType", "avgRouteSize", "seed", "clusterSeeds"]);
const BULK_FLOAT_FIELDS = new Set(["clusterDecayMeters", "hybridPoiShare"]);
const SAMPLING_METHODS = ["poi_categories", "parametric_attach", "hybrid", "manual"];
const DEPOT_MODES = ["center", "random", "corner"];
const CUSTOMER_MODES = ["random_clustered", "clustered", "random"];
const TW_METHODS = ["route_centered", "reachable_interval"];

const isDark = () => document.documentElement.dataset.theme !== "light";
const routeColor = (index) => window.MamutNocturne.routeColor(index);
const markerColors = () => (isDark()
  ? { depot: "#ff7a6e", poi: "#45e8a5", parametric: "#9d8bff" }
  : { depot: "#e8503f", poi: "#0f9e68", parametric: "#5b43e8" });

const el = (id) => document.getElementById(id);
const status = (text) => { el("status").textContent = text; el("status").title = text; };

/* ── Map ── */
/* MapLibre GL through the shared kit in mamut-map.js (the public website draws with
   the same file). CARTO vector basemaps need the CARTO basemaps API key that the
   server inlines from MAMUT_BASEMAP_API_KEY (https://carto.com/basemaps/apikey);
   without it the map runs on OpenStreetMap only. CARTO's free tier requires CARTO
   and OpenStreetMap to stay credited on the map; mamut-map.js sets both. */
const basemapApiKey = window.__MAMUT_BASEMAP_API_KEY__ || "";
const mapController = window.MamutMap.createMap(el("map"), {
  center: [2.5, 46.5],
  zoom: 5,
  apiKey: basemapApiKey,
  dark: isDark(),
  scale: true,
  // Shift-click picks a POI as the depot; MapLibre's shift-drag box zoom would eat it.
  boxZoom: false,
});
const map = mapController?.map || null;
if (!map) {
  const notice = document.createElement("p");
  notice.className = "map-unavailable";
  notice.textContent = "The map needs WebGL, which this browser does not provide.";
  el("map").appendChild(notice);
}

/* The basemap normally follows the theme. Once the user picks one explicitly that
   link is cut, otherwise flipping the theme would silently undo their choice. */
let basemapPinned = false;
mapController?.onUserBasemapChoice(() => {
  basemapPinned = true;
});

const solutionLayers = map
  ? window.MamutMap.createSolutionLayers(map, {
    interaction: "hover",
    nodeRadius: 5,
    depotRadius: 7,
    nodeStrokeWidth: 2,
    depotStrokeWidth: 3,
    nodeFillOpacity: 0.9,
    describe: describeSolutionFeature,
  })
  : null;

/* What the solution layers currently hold; geometry is pushed to the map only when
   it changes, every view option is feature-state. */
let solutionData = { key: null, routeLines: [], nodes: [], depot: null, routeMode: "straight_line", hasRoutes: false };

/* Route animation (mamut-trips.js). deck.gl is fetched only when one is first
   opened; animated paths are thinned to a vertex budget deck redraws at ~60 fps. */
const DECK_SRC = "/static/vendor/deck.gl/dist.min.js";
const TRIPS_VERTEX_BUDGET = 500000;
let routeAnimation = null;
let routeAnimationLoading = false;

/* The generation preview and the manual POI pool are this GUI's own layers. Their
   ids start with "mamut-" so basemap swaps slide in underneath them. */
const toolLayers = { ready: false, preview: null, poi: null };
if (map) map.once("load", installToolLayers);

function installToolLayers() {
  const empty = { type: "FeatureCollection", features: [] };
  map.addSource("mamut-preview", { type: "geojson", data: empty });
  map.addLayer({
    id: "mamut-preview-circle",
    type: "circle",
    source: "mamut-preview",
    paint: {
      "circle-radius": ["case", ["==", ["get", "role"], "depot"], 7, 4],
      "circle-color": ["to-color", ["get", "color"]],
      "circle-opacity": 0.85,
      "circle-stroke-width": ["case", ["==", ["get", "role"], "depot"], 2, 1],
      "circle-stroke-color": ["to-color", ["get", "color"]],
    },
  });
  map.addSource("mamut-poi", { type: "geojson", data: empty });
  map.addLayer({
    id: "mamut-poi-circle",
    type: "circle",
    source: "mamut-poi",
    paint: {
      "circle-radius": ["coalesce", ["feature-state", "r"], 3.5],
      "circle-color": ["to-color", ["coalesce", ["feature-state", "c"], "#888888"]],
      "circle-opacity": ["coalesce", ["feature-state", "fo"], 0.35],
      "circle-stroke-width": ["coalesce", ["feature-state", "sw"], 1],
      "circle-stroke-color": ["to-color", ["coalesce", ["feature-state", "c"], "#888888"]],
    },
  });
  const tooltips = window.MamutMap.bindLayerTooltip(map, ["mamut-poi-circle", "mamut-preview-circle"], describeToolFeature);
  map.on("click", "mamut-poi-circle", (event) => {
    const feature = event.features?.[0];
    const poi = feature ? state.poiPool[feature.id] : null;
    if (!poi) return;
    const key = poiKey(poi.properties);
    // Shift-click promotes a POI to the depot instead of toggling it.
    if (event.originalEvent && event.originalEvent.shiftKey) setManualDepot(key, poi);
    else toggleManualPick(key, poi);
    // The tooltip names the pick state; show the new one on the next pointer move.
    tooltips.refresh();
  });
  toolLayers.ready = true;
  setPreviewFeatures(toolLayers.preview);
  if (toolLayers.poi) {
    map.getSource("mamut-poi").setData(toolLayers.poi);
    applyPoiStates();
  }
}

/* Layer handles kept from the Leaflet version: the call sites only ever clear them. */
const previewLayers = { clearLayers: () => setPreviewFeatures(null) };
const instanceNodeLayers = {
  clearLayers: () => {
    solutionData = { ...solutionData, key: null, nodes: [], depot: null, hasRoutes: false };
    pushSolution();
  },
};
const poiPickLayers = { clearLayers: () => setPoiFeatures(null) };

function textTooltip(text) {
  const node = document.createElement("div");
  node.className = "map-tooltip";
  node.textContent = text;
  return node;
}

function describeToolFeature(feature) {
  if (feature.layer.id === "mamut-poi-circle") {
    const poi = state.poiPool[feature.id];
    if (!poi) return null;
    const key = poiKey(poi.properties);
    const suffix = state.manualDepotKey === key ? " · DEPOT" : state.manualPicks.has(key) ? " · picked" : "";
    return textTooltip(`${poiLabel(poi.properties)}${suffix}`);
  }
  const previewFeature = state.previewGeojson?.features?.[feature.id];
  return previewFeature ? textTooltip(nodeTooltip(previewFeature.properties)) : null;
}

/* The left and right panels, so a fit keeps the data clear of them. */
function fitMapTo(bounds) {
  if (!map) return;
  window.MamutMap.fitToBounds(map, bounds, [document.querySelector(".panel-left"), document.querySelector(".panel-right")], { base: 40 });
}

/* ── Preferences ── */
/* Durable across GUI restarts: localStorage is keyed by origin including the port, and
   the server binds a new random port each start, so it alone loses every preference. */
function savePreferences(patch) {
  fetch("/api/preferences", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  }).catch(() => { /* preferences are a convenience; never surface a failure */ });
}

/* ── Theme ── */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("mamut-theme", theme); } catch (error) { /* private mode */ }
  savePreferences({ theme });
  if (mapController && !basemapPinned) {
    const themed = window.MamutMap.themedBasemapId(isDark(), basemapApiKey);
    if (themed !== mapController.basemaps.current) mapController.basemaps.set(themed);
  }
  redrawRoutes();
  if (state.poiPool.length) applyPoiStates();
}
document.querySelectorAll("[data-theme-pick]").forEach((button) => {
  button.addEventListener("click", () => applyTheme(button.dataset.themePick));
});

/* ── API ── */
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({ ok: false, error: `HTTP ${response.status}` }));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function apiBlob(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.blob();
}

async function apiDelete(path) {
  const response = await fetch(path, { method: "DELETE" });
  const data = await response.json().catch(() => ({ ok: false, error: `HTTP ${response.status}` }));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function submitJob(kind, payload) {
  const data = await api("/api/jobs", { kind, payload });
  state.submittedJobs.add(data.job.job_id);
  state.jobs = [data.job, ...state.jobs.filter((job) => job.job_id !== data.job.job_id)];
  renderJobs();
  return data.job;
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/* ── State ── */
const state = {
  instances: [],
  selected: null,
  solvers: [],
  hiddenRoutes: new Set(),
  focusedRoute: null,
  routeView: {
    fadedOpacity: 0.2,
    depotLegOpacity: 0.25,
    renderMode: "straight_line",
    customersFollowRoutes: false,
    arrowsEnabled: true,
    depotStar: false,
  },
  lastRendered: null,
  bulkBases: null,
  jobs: [],
  openJobLogId: null,
  openJobLogStatus: null,
  submittedJobs: new Set(),
  handledJobs: new Set(),
  solutionRuns: [],
  solutionReferences: [],
  selectedRunId: null,
  previewGeojson: null,
  // Manual POI picking
  poiPool: [],
  poiPoolCity: null,
  manualPicks: new Map(),
  manualDepotKey: null,
  osmAudit: null,
  // Bulk configuration table
  bulkRows: [],
  bulkNextId: 1,
  // The last run comparison: { instanceId, candidateId, html }.
  comparison: null,
  // Published instances: the listing, the family/problem scope shown, and the panel objects by id.
  instanceSource: "workspace",
  benchmarks: { loaded: false, loading: false, instances: [], hint: null, root: null, scope: null },
  benchmarkObjects: new Map(),
};

/* ── Tabs ── */
function activateTab(name) {
  el("tabVisualize").classList.toggle("tab-active", name === "visualize");
  el("tabGenerate").classList.toggle("tab-active", name === "generate");
  el("visualizePanel").classList.toggle("tab-panel-active", name === "visualize");
  el("generatePanel").classList.toggle("tab-panel-active", name === "generate");
  document.querySelectorAll("[data-rail-target]").forEach((button) => {
    button.classList.toggle("rail-active", button.dataset.railTarget === name);
  });
}
el("tabVisualize").addEventListener("click", () => activateTab("visualize"));
el("tabGenerate").addEventListener("click", () => activateTab("generate"));

/* ── Resizable layout ── */
const layout = window.MamutLayout.initLayout({
  stage: document.documentElement,
  storageKey: window.MamutLayout.STORAGE_KEY,
  defaults: (window.__MAMUT_PREFS__ || {}).layout,
  /* MapLibre caches the container size; resize after a drag so it never draws stale. */
  onResize: () => map?.resize(),
  onPersist: (state) => savePreferences({ layout: state }),
});

/* The markup ships with Generate active; run it once so the rail highlight agrees. */
activateTab(el("tabGenerate").classList.contains("tab-active") ? "generate" : "visualize");

/* ── Inspector tabs ── */
const INSPECTOR_TABS = { routes: "routesPanel", runs: "runsPanel", solve: "solvePanel", details: "detailsPanel" };
function activateInspectorTab(name) {
  if (!INSPECTOR_TABS[name]) return;
  Object.entries(INSPECTOR_TABS).forEach(([tab, panelId]) => {
    el(panelId).classList.toggle("tab-panel-active", tab === name);
    el(`tab${tab[0].toUpperCase()}${tab.slice(1)}`).classList.toggle("tab-active", tab === name);
  });
  document.querySelectorAll('[data-rail-side="right"][data-rail-target]').forEach((button) => {
    button.classList.toggle("rail-active", button.dataset.railTarget === name);
  });
}
Object.keys(INSPECTOR_TABS).forEach((name) => {
  el(`tab${name[0].toUpperCase()}${name.slice(1)}`)
    .addEventListener("click", () => activateInspectorTab(name));
});
activateInspectorTab("routes");

/* ── Instance filters ── */
["inst-search", "inst-city", "inst-sort", "inst-solved"].forEach((id) => {
  el(id).addEventListener("input", renderInstanceList);
});

/* Copy the full workspace path out of the Details tab. */
el("selFacts").addEventListener("click", async (event) => {
  const button = event.target instanceof Element ? event.target.closest("[data-copy]") : null;
  if (!button) return;
  try {
    await navigator.clipboard.writeText(button.dataset.copy);
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = "Copy"; }, 1200);
  } catch (error) {
    status("Could not copy the path to the clipboard.");
  }
});

el("appearance-render-mode").addEventListener("change", (event) => {
  state.routeView.renderMode = event.target.value;
  redrawRoutes();
});
el("appearance-depot-opacity").addEventListener("input", (event) => {
  state.routeView.depotLegOpacity = Number(event.target.value);
  el("appearance-depot-value").textContent = `${Math.round(state.routeView.depotLegOpacity * 100)}%`;
  redrawRoutes();
});
el("appearance-other-opacity").addEventListener("input", (event) => {
  state.routeView.fadedOpacity = Number(event.target.value);
  el("appearance-other-value").textContent = `${Math.round(state.routeView.fadedOpacity * 100)}%`;
  redrawRoutes();
});
el("appearance-customers").addEventListener("change", (event) => {
  state.routeView.customersFollowRoutes = event.target.checked;
  redrawRoutes();
});
el("appearance-arrows").addEventListener("change", (event) => {
  state.routeView.arrowsEnabled = event.target.checked;
  redrawRoutes();
});
el("appearance-depot-star").addEventListener("change", (event) => {
  state.routeView.depotStar = event.target.checked;
  redrawRoutes();
});

/* ── Activity strip ── */
function setActivityOpen(open) {
  el("activity").dataset.open = String(open);
  el("activity-body").hidden = !open;
  el("activity-toggle").setAttribute("aria-expanded", String(open));
}
el("activity-toggle").addEventListener("click", () => {
  setActivityOpen(el("activity-body").hidden);
});
el("jobs-filter").addEventListener("change", renderJobs);

document.documentElement.addEventListener("layout:rail-select", (event) => {
  const target = event.detail.target;
  if (target === "visualize" || target === "generate") {
    activateTab(target);
  } else if (target === "jobs") {
    setActivityOpen(true);
  } else if (INSPECTOR_TABS[target]) {
    activateInspectorTab(target);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  /* Never steal a bracket the user is typing into a field. */
  const target = event.target;
  if (target instanceof Element && target.closest("input, select, textarea, [contenteditable]")) return;
  if (event.key === "[") layout.toggleCollapsed("left");
  else if (event.key === "]") layout.toggleCollapsed("right");
  else return;
  event.preventDefault();
});

/* ── Generation inputs ── */
function selectedPoiCategories() {
  return [...el("poi-list").querySelectorAll("input[type='checkbox']:checked")]
    .map((checkbox) => checkbox.value);
}

function updatePoiCount() {
  const label = `${selectedPoiCategories().length} selected`;
  el("poi-count").textContent = label;
  el("poi-popover-count").textContent = label;
}

function setPoiSelection(categories) {
  const selected = new Set(categories);
  el("poi-list").querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
    checkbox.checked = selected.has(checkbox.value);
  });
  updatePoiCount();
}

function renderPoiCategories() {
  const list = el("poi-list");
  list.innerHTML = "";
  POI_CATEGORIES.forEach((category) => {
    const label = document.createElement("label");
    label.className = "poi-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = category;
    checkbox.checked = DEFAULT_POI_CATEGORIES.has(category);
    checkbox.addEventListener("change", updatePoiCount);
    const text = document.createElement("span");
    text.textContent = category.replaceAll("_", " ");
    label.title = category;
    label.append(checkbox, text);
    list.append(label);
  });
  updatePoiCount();
}

/* ── Labelled dials and inline help ── */
function fillBandSelects() {
  const fill = (id, entries, selected) => {
    const select = el(id);
    select.innerHTML = "";
    entries.forEach((entry) => {
      const option = new Option(`${entry.value} · ${entry.label}`, String(entry.value));
      option.title = entry.detail;
      select.append(option);
    });
    select.value = String(selected);
  };
  fill("demand", DEMAND_TYPES, 7);
  fill("ars", ROUTE_SIZE_BANDS, 4);

  const table = (id, entries) => {
    el(id).innerHTML = entries
      .map((entry) => `<dt>${entry.value}</dt><dd>${entry.label} — ${entry.detail}</dd>`)
      .join("");
  };
  table("help-demand-table", DEMAND_TYPES);
  table("help-band-table", ROUTE_SIZE_BANDS);
}

/* ── Popovers ── */
/* Anchor a body-level popover beside its trigger, nudged back inside the viewport.
   Shared by the category picker and the field help. */
function placePopover(popover, anchor, beside = anchor) {
  popover.hidden = false;
  const box = anchor.getBoundingClientRect();
  // `beside` is what the popover must not cover: the trigger itself, or the whole
  // panel the trigger sits in.
  const side = beside.getBoundingClientRect();
  const size = popover.getBoundingClientRect();
  const margin = 10;
  let left = side.right + margin;
  let top = box.top;
  if (left + size.width > window.innerWidth - margin) {
    left = side.left - size.width - margin;
    if (left < margin && beside !== anchor) {
      // No room on either side of a full-width sheet: sit above it, in the trigger's column.
      left = Math.min(box.right - size.width, window.innerWidth - size.width - margin);
      top = Math.max(margin, side.top - size.height - margin);
    }
    left = Math.max(margin, left);
  }
  if (top + size.height > window.innerHeight - margin) {
    top = Math.max(margin, window.innerHeight - size.height - margin);
  }
  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.round(top)}px`;
}

/* Only one popover is open at a time, and whatever opened it gets told to close. */
let openPopover = null;
function closePopover() {
  if (!openPopover) return;
  openPopover.onClose();
  openPopover = null;
}
function showPopover(popover, anchor, onClose, beside) {
  closePopover();
  placePopover(popover, anchor, beside || anchor);
  openPopover = { popover, onClose: onClose || (() => { popover.hidden = true; }) };
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closePopover();
});
document.addEventListener("pointerdown", (event) => {
  if (!openPopover) return;
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (openPopover.popover.contains(target) || target.closest(".help-btn, #poi-edit, [data-popover-trigger]")) return;
  closePopover();
});
window.addEventListener("resize", closePopover);

/* Field help: the body is moved into the popover and moved back on close, so the
   tables that JS populates keep their identity instead of being cloned stale. */
const helpPopover = el("help-popover");
const helpPopoverBody = el("help-popover-body");
let helpReturn = null;

function restoreHelpBody() {
  if (!helpReturn) return;
  helpReturn.parent.insertBefore(helpReturn.node, helpReturn.next);
  helpReturn.node.hidden = true;
  helpReturn.button.setAttribute("aria-expanded", "false");
  helpReturn = null;
  helpPopover.hidden = true;
}

document.querySelectorAll(".help-btn").forEach((button) => {
  button.addEventListener("click", () => {
    const body = el(button.dataset.help);
    if (!body) return;
    const alreadyOpen = helpReturn && helpReturn.node === body;
    closePopover();
    if (alreadyOpen) return;
    const head = button.closest(".field-head, .step-head");
    const label = head ? head.querySelector("span:not(.step-index)") : null;
    el("help-popover-title").textContent = label ? label.textContent : "About this field";
    helpReturn = { node: body, parent: body.parentNode, next: body.nextSibling, button };
    helpPopoverBody.append(body);
    body.hidden = false;
    button.setAttribute("aria-expanded", "true");
    showPopover(helpPopover, button, restoreHelpBody);
  });
});
el("help-popover-close").addEventListener("click", closePopover);

/* "Explain this form" reveals every help body inline in one step, for reading the
   whole form rather than one field at a time. */
el("explain-toggle").addEventListener("click", () => {
  const button = el("explain-toggle");
  closePopover();
  const on = button.getAttribute("aria-pressed") !== "true";
  button.setAttribute("aria-pressed", on ? "true" : "false");
  button.textContent = on ? "Hide explanations" : "Explain this form";
  el("generatePanel").classList.toggle("explain-all", on);
});

/* Category picker */
el("poi-edit").addEventListener("click", () => {
  const button = el("poi-edit");
  const popover = el("poi-popover");
  if (!popover.hidden) { closePopover(); return; }
  button.setAttribute("aria-expanded", "true");
  showPopover(popover, button, () => {
    popover.hidden = true;
    button.setAttribute("aria-expanded", "false");
  });
  el("poi-search").focus();
});
el("poi-popover-close").addEventListener("click", closePopover);
el("poi-search").addEventListener("input", () => {
  const needle = el("poi-search").value.trim().toLowerCase();
  el("poi-list").querySelectorAll("label").forEach((label) => {
    label.hidden = needle !== "" && !label.textContent.toLowerCase().includes(needle);
  });
});

function updateGenerationControls() {
  const method = el("method").value;
  const hybridPoiPercent = Number(el("hybrid-poi-share").value);
  const manual = method === "manual";
  const usesPoi = method === "poi_categories" || (method === "hybrid" && hybridPoiPercent > 0);
  const usesParametric = method === "parametric_attach" || (method === "hybrid" && hybridPoiPercent < 100);
  // Manual picking reuses the category checkboxes to filter what the map shows.
  el("poi-picker").hidden = !usesPoi && !manual;
  // The attachment controls moved into step 2's Advanced block, so they no longer
  // inherit the picker's visibility and have to be switched on the same condition.
  el("poi-attach-block").hidden = !usesPoi && !manual;
  el("manual-options").hidden = !manual;
  el("hybrid-options").hidden = method !== "hybrid";
  el("parametric-options").hidden = !usesParametric;
  const usesClusters = usesParametric && el("customer-mode").value !== "random";
  el("cluster-options").hidden = !usesClusters;
  el("cluster-note").hidden = !usesClusters;
  // The radius only means anything when POIs are allowed to move.
  el("poi-attach-radius-field").hidden = el("poi-attach-mode").value !== "nearest_vertex";
  // n is derived from the picks in manual mode.
  el("n").disabled = manual;
  if (!manual) clearPoiPool();
}

function updateHybridShare() {
  const poiPercent = Number(el("hybrid-poi-share").value);
  el("hybrid-share-value").textContent = `${poiPercent}% POI / ${100 - poiPercent}% parametric`;
}

el("poi-default").addEventListener("click", () => setPoiSelection(DEFAULT_POI_CATEGORIES));
el("poi-all").addEventListener("click", () => setPoiSelection(POI_CATEGORIES));
el("poi-clear").addEventListener("click", () => setPoiSelection([]));
el("method").addEventListener("change", updateGenerationControls);
el("customer-mode").addEventListener("change", updateGenerationControls);
el("poi-attach-mode").addEventListener("change", updateGenerationControls);
el("hybrid-poi-share").addEventListener("input", () => {
  updateHybridShare();
  updateGenerationControls();
});

function requestBody() {
  const method = el("method").value;
  const categories = selectedPoiCategories();
  const hybridPoiShare = Number(el("hybrid-poi-share").value) / 100;
  if ((method === "poi_categories" || (method === "hybrid" && hybridPoiShare > 0)) && !categories.length) {
    throw new Error("Select at least one POI category for this sampling method.");
  }
  // Sent as {type, id} pairs: an id alone would be ambiguous between a node, a
  // way and a relation carrying the same number.
  const manualPois = [...state.manualPicks.keys()].map((key) => {
    const [osmType, osmId] = key.split("/");
    return { type: osmType, id: Number(osmId) };
  });
  const depotKey = state.manualDepotKey;
  if (method === "manual") {
    const customers = [...state.manualPicks.keys()].filter((key) => key !== depotKey).length;
    if (customers < 2) {
      throw new Error("Pick at least 2 POIs (besides the depot) on the map for manual generation.");
    }
  }
  return {
    city: el("city").value,
    manualPois: method === "manual" ? manualPois : undefined,
    manualDepotPoi: method === "manual" && depotKey
      ? { type: depotKey.split("/")[0], id: Number(depotKey.split("/")[1]) }
      : undefined,
    nCustomers: Number(el("n").value),
    seed: Number(el("seed").value),
    method,
    depotMode: el("depot-mode").value,
    customerMode: el("customer-mode").value,
    clusterSeeds: Number(el("cluster-seeds").value),
    clusterDecayMeters: Number(el("cluster-decay").value),
    hybridPoiShare,
    poiAttachMode: el("poi-attach-mode").value,
    poiAttachRadiusM: Number(el("poi-attach-radius").value) || 50,
    categories,
    demandType: Number(el("demand").value),
    avgRouteSize: Number(el("ars").value),
    deriveVrptw: el("problem").value === "vrptw",
    overwrite: el("overwrite").checked,
  };
}

async function loadInstances(selectBase) {
  const data = await api("/api/workbench/instances");
  const previous = new Map(state.instances.map((instance) => [`${instance.folder}/${instance.base_name}`, instance]));
  state.instances = (data.instances || []).map((entry) => {
    const kept = previous.get(`${entry.folder}/${entry.base_name}`);
    return {
      instance_id: entry.instance_id,
      base_name: entry.base_name,
      folder: entry.folder,
      files: entry.files,
      summary: entry.summary || {},
      vrptw: entry.has_vrptw_twin || null,
      city: entry.city || "?",
      n: entry.n_customers ?? "?",
      seed: entry.seed ?? "?",
      solution_count: entry.solution_count || 0,
      solved: kept?.solved || null,
      rendered: kept?.rendered || null,
      mapData: kept?.mapData || null,
    };
  });
  if (selectBase) {
    const index = state.instances.findIndex((instance) => instance.base_name === selectBase);
    if (index >= 0) selectInstance(index);
  } else if (state.selected && state.selected.source !== "benchmark") {
    const index = state.instances.findIndex((instance) => instance.base_name === state.selected.base_name && instance.folder === state.selected.folder);
    state.selected = index >= 0 ? state.instances[index] : null;
  }
  renderInstanceList();
  renderSelected();
}

async function loadCities() {
  const data = await api("/api/workbench/generation/cities");
  const select = el("city");
  select.innerHTML = "";
  data.cities.forEach((city) => {
    const option = document.createElement("option");
    option.value = city.label;
    option.textContent = city.label;
    select.append(option);
  });
  status(data.cities.length
    ? `${data.cities.length} city extract(s) in the workspace.`
    : "No OSM extracts yet: fetch a city first.");
}

/* ── Solvers ── */
function solverOptionNote(solver) {
  if (!solver) return "";
  if (!solver.available) return [solver.reason, solver.hint, solver.note].filter(Boolean).join(" — ");
  if (solver.supported === false) return (solver.reasons || []).join("; ");
  return (solver.warnings || []).join("; ");
}

function syncSolverNote() {
  const note = el("solve-solver-note");
  const solver = (state.solvers || []).find((entry) => entry.name === el("solve-solver").value);
  const text = solverOptionNote(solver);
  note.hidden = !text;
  note.textContent = text;
}

async function loadSolvers() {
  const select = el("solve-solver");
  const previous = select.value || "pyvrp";
  const params = new URLSearchParams();
  if (state.selected) {
    params.set("instance_id", state.selected.instance_id);
    params.set("metric", el("solve-metric").value);
    params.set("objective_function", el("solve-objective").value);
  }
  const query = params.toString();
  const data = await api(`/api/solvers${query ? `?${query}` : ""}`);
  state.solvers = data.solvers;
  select.innerHTML = "";
  data.solvers.forEach((solver) => {
    const option = document.createElement("option");
    const unsupported = solver.supported === false;
    option.value = solver.name;
    option.textContent = `${solver.label}${solver.version ? ` ${solver.version}` : ""}`
      + (!solver.available ? " — not installed" : unsupported ? " — not for this variant" : "");
    option.disabled = !solver.available || unsupported;
    option.title = solverOptionNote(solver);
    select.append(option);
  });
  const options = [...select.options];
  const keep = options.find((option) => option.value === previous && !option.disabled);
  select.value = keep ? previous : (options.find((option) => !option.disabled)?.value || "pyvrp");
  syncSolverNote();
}

/* ── Map drawing ── */
function clearRouteLayers() {
  solutionData = { ...solutionData, key: null, routeLines: [], routeMode: "straight_line", hasRoutes: false };
  pushSolution();
}

function setPreviewFeatures(features) {
  toolLayers.preview = features;
  if (!toolLayers.ready) return;
  map.getSource("mamut-preview").setData({ type: "FeatureCollection", features: features || [] });
}

function drawPreview(geojson) {
  state.previewGeojson = geojson;
  solutionData = { key: null, routeLines: [], nodes: [], depot: null, routeMode: "straight_line", hasRoutes: false };
  pushSolution();
  const colors = markerColors();
  const points = [];
  // The colour is data here (not feature-state): a preview is small, and a theme
  // change simply redraws it.
  setPreviewFeatures(geojson.features.map((feature, index) => {
    points.push(feature.geometry.coordinates);
    const tag = String(feature.properties.source_tag || "");
    const role = feature.properties.role === "depot"
      ? "depot"
      : (tag.startsWith("poi") ? "poi" : "parametric");
    return { type: "Feature", id: index, properties: { role, color: colors[role] }, geometry: feature.geometry };
  }));
  if (points.length) fitMapTo(window.MamutMap.boundsOf(points));
}

/* What a merged amenity reads as: its name, or its category when unnamed, and
   the element type when it is an outline rather than a point — otherwise a list
   of names gives no clue which of them came from a way or a relation. */
function mergedPoiLabel(entry) {
  const base = entry.name || `(unnamed ${String(entry.category || "poi").replaceAll("_", " ")})`;
  return entry.osm_type && entry.osm_type !== "node" ? `${base} [${entry.osm_type}]` : base;
}

/* A node's label: its OSM name when the pipeline captured one, else the
   amenity, else just the source tag. */
function nodeTooltip(properties) {
  const parts = [`#${properties.instance_node_id}`];
  if (properties.poi_name) parts.push(properties.poi_name);
  else if (properties.poi_category) parts.push(properties.poi_category.replaceAll("_", " "));
  parts.push(String(properties.source_tag || ""));
  // Several amenities on one road point collapse into this single customer;
  // naming them is the only way to see what the node really stands for.
  const merged = properties.poi_merged || [];
  if (merged.length) {
    parts.push(`+${merged.length} here: ${merged.map(mergedPoiLabel).join(", ")}`);
  }
  // Worth showing: a way or relation POI is a building outline reduced to its
  // centre, so its position is approximate in a way a node's is not.
  if (properties.poi_osm_type && properties.poi_osm_type !== "node") {
    parts.push(`${properties.poi_osm_type} centre`);
  }
  if (properties.snap_distance_m > 1) parts.push(`snapped ${Math.round(properties.snap_distance_m)} m`);
  return parts.join(" · ");
}

/* ── Manual POI picking ── */
function poiLabel(properties) {
  const base = properties.name
    || `(unnamed ${String(properties.category || "poi").replaceAll("_", " ")})`;
  // Way and relation POIs are outlines reduced to their centre; saying so also
  // separates two picks that share a name but not an element type.
  const kind = properties.osm_type && properties.osm_type !== "node"
    ? ` [${properties.osm_type}]`
    : "";
  return `${base}${kind}`;
}

function clearPoiPool() {
  poiPickLayers.clearLayers();
  state.poiPool = [];
  state.poiPoolCity = null;
}

function setPoiFeatures(collection) {
  toolLayers.poi = collection;
  if (!toolLayers.ready) return;
  map.getSource("mamut-poi").setData(collection || { type: "FeatureCollection", features: [] });
}

async function loadPoiPool() {
  const city = el("city").value;
  if (!city) throw new Error("Select a city first.");
  const categories = selectedPoiCategories();
  const query = new URLSearchParams({ city, categories: categories.join(",") });
  const data = await api(`/api/workbench/generation/pois?${query.toString()}`);
  // Picks from another city cannot resolve against this extract. Compared
  // against the city the pool was last loaded for, not against data.city, which
  // is the request's own parameter echoed back and so can never differ.
  const previousCity = state.poiPoolCity;
  const dropped = previousCity && previousCity !== city ? state.manualPicks.size : 0;
  if (dropped) {
    state.manualPicks.clear();
    state.manualDepotKey = null;
  }
  state.poiPool = data.geojson.features;
  state.poiPoolCity = city;
  renderPoiMarkers(true);
  el("manual-available").textContent = data.truncated
    ? `${data.returned} of ${data.matching} matching POI(s) shown (capped) · ${data.named} named.`
      + " Narrow the categories to see the rest."
    : `${data.returned} POI(s) shown of ${data.total_in_extract} in the extract · ${data.named} named.`;
  status(`Loaded ${data.returned} POI(s) for ${city}; click markers to pick them.`
    + (dropped ? ` Cleared ${dropped} pick(s) made in ${previousCity}.` : ""));
}

/* OSM ids repeat across element types — node 42 and way 42 are different places
   — so a pick is identified by both halves. */
function poiKey(properties) {
  return `${properties.osm_type || "node"}/${properties.osm_id}`;
}

function poiStyle(key, colors) {
  const picked = state.manualPicks.has(key);
  const isDepot = state.manualDepotKey === key;
  return {
    r: isDepot ? 7 : picked ? 5.5 : 3.5,
    c: isDepot ? colors.depot : picked ? colors.poi : colors.parametric,
    fo: picked || isDepot ? 0.9 : 0.35,
    sw: picked || isDepot ? 2 : 1,
  };
}

/* Picks restyle the pool through feature-state: a click on one of 20 000 POIs no
   longer rebuilds every marker. */
function applyPoiStates() {
  if (!toolLayers.ready || !toolLayers.poi) return;
  const colors = markerColors();
  state.poiPool.forEach((feature, index) => {
    map.setFeatureState({ source: "mamut-poi", id: index }, poiStyle(poiKey(feature.properties), colors));
  });
}

function renderPoiMarkers(fit = false) {
  if (toolLayers.poi?.pool !== state.poiPool) {
    const collection = {
      type: "FeatureCollection",
      features: state.poiPool.map((feature, index) => ({ type: "Feature", id: index, properties: {}, geometry: feature.geometry })),
    };
    // Remembered so the source is only replaced when the pool itself changes.
    Object.defineProperty(collection, "pool", { value: state.poiPool, enumerable: false });
    setPoiFeatures(collection);
  }
  applyPoiStates();
  if (fit && state.poiPool.length) fitMapTo(window.MamutMap.boundsOf(state.poiPool.map((feature) => feature.geometry.coordinates)));
  renderManualPicks();
}

function toggleManualPick(key, feature) {
  if (state.manualPicks.has(key)) {
    state.manualPicks.delete(key);
    if (state.manualDepotKey === key) state.manualDepotKey = null;
  } else {
    state.manualPicks.set(key, feature.properties);
  }
  renderPoiMarkers();
}

function setManualDepot(key, feature) {
  if (!state.manualPicks.has(key)) state.manualPicks.set(key, feature.properties);
  state.manualDepotKey = state.manualDepotKey === key ? null : key;
  renderPoiMarkers();
}

function renderManualPicks() {
  const customers = [...state.manualPicks.keys()].filter((key) => key !== state.manualDepotKey).length;
  el("manual-count").textContent = String(customers);
  const list = el("manual-list");
  list.innerHTML = "";
  if (!state.manualPicks.size) {
    list.innerHTML = '<div class="empty-note">No POI picked yet.</div>';
    return;
  }
  state.manualPicks.forEach((properties, key) => {
    const row = document.createElement("div");
    row.className = "picked-row";
    const name = document.createElement("span");
    name.className = "nm";
    name.textContent = poiLabel(properties);
    row.append(name);
    if (state.manualDepotKey === key) {
      const flag = document.createElement("span");
      flag.className = "depot-flag";
      flag.textContent = "depot";
      row.append(flag);
    }
    const category = document.createElement("span");
    category.className = "cat";
    category.textContent = String(properties.category || "").replaceAll("_", " ");
    row.append(category);
    const remove = document.createElement("button");
    remove.className = "bulk-delete-btn";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Remove this pick";
    remove.addEventListener("click", () => {
      state.manualPicks.delete(key);
      if (state.manualDepotKey === key) state.manualDepotKey = null;
      renderPoiMarkers();
    });
    row.append(remove);
    list.append(row);
  });
}

el("manual-load").addEventListener("click", async () => {
  try {
    await loadPoiPool();
  } catch (error) { status(`Cannot load POIs: ${error.message}`); }
});
el("manual-clear").addEventListener("click", () => {
  state.manualPicks.clear();
  state.manualDepotKey = null;
  renderPoiMarkers();
});
el("manual-depot-clear").addEventListener("click", () => {
  state.manualDepotKey = null;
  renderPoiMarkers();
});

function cachedRoadRenderingAvailable(rendered = state.lastRendered) {
  return ["cached_road", "mixed"].includes(rendered?.summary?.render_mode);
}

function resetRouteViewDefaults(rendered = null) {
  const routeCount = Array.isArray(state.selected?.solved?.routes) ? state.selected.solved.routes.length : 0;
  state.hiddenRoutes.clear();
  state.focusedRoute = null;
  state.routeView.fadedOpacity = 0.2;
  state.routeView.depotLegOpacity = 0.25;
  state.routeView.renderMode = cachedRoadRenderingAvailable(rendered) ? "cached_road" : "straight_line";
  state.routeView.customersFollowRoutes = false;
  state.routeView.arrowsEnabled = routeCount <= 10;
  state.routeView.depotStar = routeCount > 0;
}

function syncAppearanceControls() {
  const visible = Boolean(state.selected?.solved && state.lastRendered);
  // The settings live in a popover; without a drawn run there is nothing to set.
  el("appearance-open").hidden = !visible;
  if (!visible && openPopover?.popover === el("route-appearance")) closePopover();
  if (!visible) return;
  const cachedAvailable = cachedRoadRenderingAvailable();
  const renderSelect = el("appearance-render-mode");
  const cachedOption = renderSelect.querySelector('[value="cached_road"]');
  cachedOption.disabled = !cachedAvailable;
  cachedOption.textContent = cachedAvailable ? "Cached road" : "Cached road (unavailable)";
  if (!cachedAvailable && state.routeView.renderMode === "cached_road") {
    state.routeView.renderMode = "straight_line";
  }
  renderSelect.value = state.routeView.renderMode;
  el("appearance-depot-field").hidden = state.routeView.renderMode !== "straight_line";
  el("appearance-depot-opacity").value = String(state.routeView.depotLegOpacity);
  el("appearance-depot-value").textContent = `${Math.round(state.routeView.depotLegOpacity * 100)}%`;
  el("appearance-other-opacity").value = String(state.routeView.fadedOpacity);
  el("appearance-other-opacity").disabled = state.focusedRoute === null;
  el("appearance-focus-hint").textContent = state.focusedRoute === null ? " (focus a route)" : "";
  el("appearance-other-value").textContent = `${Math.round(state.routeView.fadedOpacity * 100)}%`;
  el("appearance-customers").checked = state.routeView.customersFollowRoutes;
  el("appearance-arrows").checked = state.routeView.arrowsEnabled;
  el("appearance-depot-star").checked = state.routeView.depotStar;
}

function customerMarkerOpacity(routeIndex) {
  if (!state.routeView.customersFollowRoutes || !Number.isInteger(routeIndex)) return 1;
  if (state.hiddenRoutes.has(routeIndex)) return 0;
  if (state.focusedRoute !== null && state.focusedRoute !== routeIndex) return state.routeView.fadedOpacity;
  return 1;
}

function instanceNodeData(routes) {
  const features = state.selected?.mapData?.geojson?.features;
  if (!features) return { nodes: [], depot: null };
  const assignments = new Map();
  (routes || []).forEach((route, routeIndex) => {
    route.forEach((modelNodeId) => assignments.set(Number(modelNodeId), routeIndex));
  });
  const nodes = [];
  let depot = null;
  features.forEach((feature) => {
    const properties = feature.properties || {};
    const index = Number(properties.model_node_id);
    const coord = feature.geometry.coordinates;
    if (properties.role === "depot") {
      depot = { index, coord, properties };
    } else {
      nodes.push({ index, coord, routeIndex: assignments.get(index), properties });
    }
  });
  return { nodes, depot };
}

function pushSolution() {
  // New geometry: an open animation would keep showing the old routes.
  closeRouteAnimation();
  if (!solutionLayers) return;
  solutionLayers.setData({ routeLines: solutionData.routeLines, nodes: solutionData.nodes, depot: solutionData.depot });
  applySolutionView();
}

/* Every appearance option is feature-state on data the map already holds. */
function applySolutionView() {
  if (!solutionLayers) return;
  const routes = state.selected?.solved?.routes || [];
  const visibleRoutes = new Set();
  routes.forEach((_, index) => {
    if (!state.hiddenRoutes.has(index)) visibleRoutes.add(index);
  });
  const colors = markerColors();
  solutionLayers.applyView({
    visibleRoutes,
    focusedRoute: state.focusedRoute,
    fadedOpacity: state.routeView.fadedOpacity,
    depotLegOpacity: state.routeView.depotLegOpacity,
    routeMode: solutionData.routeMode,
    roadOpacity: 0.9,
    customerOpacity: customerMarkerOpacity,
    arrowsEnabled: state.routeView.arrowsEnabled,
    depotStar: state.routeView.depotStar && solutionData.hasRoutes,
    palette: window.MamutNocturne.routeColors(),
    depotColor: colors.depot,
    unassignedColor: (node) => (String(node.properties?.source_tag || "").startsWith("poi") ? colors.poi : colors.parametric),
    dark: isDark(),
    dim: animationPlaying() ? 0.35 : 1,
  });
  routeAnimation?.setView({ palette: window.MamutNocturne.routeColors(), visibleRoutes, focusedRoute: state.focusedRoute });
  syncAnimateButton();
}

/* One Animate button on the map beside "Clear map", whichever tab is open, and a copy in the
   "Displayed solution" card for the narrow layout, where the map's actions are hidden. */
function syncAnimateButton() {
  const shown = Boolean(map && state.selected?.solved && state.lastRendered && state.selectedRunId);
  for (const button of document.querySelectorAll("[data-animate-routes]")) {
    button.hidden = !shown;
    button.disabled = routeAnimationLoading;
    button.textContent = routeAnimationLoading ? "Loading…" : animationPlaying() ? "Stop animation" : "Animate";
    button.setAttribute("aria-pressed", String(animationPlaying()));
  }
}

/* routeAnimation also holds the "needs WebGL2" notice, so that it closes with the run; only
   a real animation dims the routes and turns the buttons into "Stop animation". */
function animationPlaying() {
  return Boolean(routeAnimation && !routeAnimation.unavailable);
}

function closeRouteAnimation() {
  const animation = routeAnimation;
  routeAnimation = null;
  animation?.close();
}

/* Times come from the run's /timeline: the checker's accounting on the instance's
   free-flow fastest arc costs. When that cannot clock the run, the timeline has no
   routes and a note, and the animation runs at constant speed. */
async function toggleRouteAnimation() {
  if (routeAnimation) {
    closeRouteAnimation();
    applySolutionView();
    return;
  }
  const instance = state.selected;
  const runId = state.selectedRunId;
  if (!map || !instance || !runId || routeAnimationLoading) return;
  routeAnimationLoading = true;
  syncAnimateButton();
  const key = solutionData.key;
  try {
    let timeline = null;
    let clockLabels;
    try {
      timeline = await api(`/api/instances/${encodeURIComponent(instance.instance_id)}/solutions/${encodeURIComponent(runId)}/timeline`);
      // The server words the clock when the default does not fit (why it is constant speed, say).
      if (timeline.note) clockLabels = { [timeline.clock]: timeline.note };
    } catch (error) {
      // No schedule for this run: the vehicles still drive, at constant speed.
      clockLabels = { schematic: `Constant speed (25 km/h): no schedule for this run (${error.message})` };
    }
    if (state.selected !== instance || solutionData.key !== key) return;
    const ranked = window.MamutMap.rankRouteLines(solutionData.routeLines, { minTolerance: 0.5 });
    const routes = state.selected?.solved?.routes || [];
    const visibleRoutes = new Set(routes.map((_, index) => index).filter((index) => !state.hiddenRoutes.has(index)));
    const animation = await window.MamutTrips.openAnimation(map, el("tripsBar"), {
      deckSrc: DECK_SRC,
      routeLines: window.MamutMap.thinRouteLines(ranked, TRIPS_VERTEX_BUDGET).routeLines,
      timeline,
      clockLabels,
      palette: window.MamutNocturne.routeColors(),
      visibleRoutes,
      focusedRoute: state.focusedRoute,
      onClose: () => {
        routeAnimation = null;
        applySolutionView();
      },
    });
    if (animation && (state.selected !== instance || solutionData.key !== key)) {
      animation.close();
      return;
    }
    routeAnimation = animation;
  } catch (error) {
    status(`Route animation unavailable: ${error.message}`);
  } finally {
    routeAnimationLoading = false;
    applySolutionView();
  }
}


/* Tooltip text, built as a text node: OSM names are data, never markup. */
function describeSolutionFeature(info) {
  if (info.kind === "route") {
    const routeLine = solutionData.routeLines[info.routeIndex];
    const properties = routeLine?.properties || {};
    return textTooltip(`R${properties.route_index ?? info.routeIndex + 1}: ${properties.stops ?? "?"} stops, load ${properties.load ?? "?"} (${solutionData.routeMode})`);
  }
  const node = info.kind === "depot" ? solutionData.depot : solutionData.nodes.find((candidate) => candidate.index === info.nodeIndex);
  if (!node) return null;
  const properties = node.properties || {};
  const label = info.kind === "depot" ? "Depot" : `Customer ${properties.model_node_id}`;
  const named = properties.poi_name ? ` · ${properties.poi_name}` : "";
  // The amenities that collapsed onto this same road point, so a node holding
  // several places does not read as one arbitrary winner.
  const merged = properties.poi_merged || [];
  const alsoHere = merged.length
    ? ` · +${merged.length} here: ${merged.map(mergedPoiLabel).join(", ")}`
    : "";
  // Non-zero when the amenity was snapped to a routable point rather than
  // sitting on one, so the node is not exactly where the place is.
  const snapped = properties.snap_distance_m > 1
    ? ` · snapped ${Math.round(properties.snap_distance_m)} m`
    : "";
  return textTooltip(`${label}${named} · demand ${properties.demand} · ${properties.source_tag}${snapped}${alsoHere}`);
}

function drawInstanceNodes(routes = null, fit = false) {
  const { nodes, depot } = instanceNodeData(routes);
  solutionData = { ...solutionData, key: null, nodes, depot, hasRoutes: Array.isArray(routes) && routes.length > 0 };
  if (!routes) solutionData.routeLines = [];
  pushSolution();
  if (fit) fitMapTo(solutionLayers?.bounds());
}

function straightRouteLines(rendered) {
  const routes = state.selected?.solved?.routes || [];
  const features = state.selected?.mapData?.geojson?.features || [];
  const nodeCoordinates = new Map(features.map((feature) => [
    Number(feature.properties?.model_node_id),
    feature.geometry?.coordinates,
  ]));
  const depotId = Number(features.find((feature) => feature.properties?.role === "depot")?.properties?.model_node_id ?? 0);
  return routes.map((route, routeIndex) => {
    const sequence = [depotId, ...route.map(Number), depotId];
    const segments = sequence.slice(0, -1).map((fromId, segmentIndex) => [
      nodeCoordinates.get(fromId),
      nodeCoordinates.get(sequence[segmentIndex + 1]),
    ].filter((point) => Array.isArray(point) && point.length >= 2));
    const properties = rendered?.geojson?.features?.[routeIndex]?.properties || {};
    return {
      routeIndex,
      sequence,
      segments,
      coordinates: segments.flatMap((segment, segmentIndex) => segmentIndex === 0 ? segment : segment.slice(1)),
      properties: { ...properties, render_mode: "straight_line" },
    };
  });
}

/* Per-arc segments of a server-rendered road route: arc k spans coordinates
   arc_starts[k] .. arc_starts[k + 1] (sharing the joint point). A payload without
   arc_starts (an older server) stays one merged segment, with no separable legs. */
function roadSegments(feature) {
  const coordinates = feature.geometry?.coordinates || [];
  const starts = feature.properties?.arc_starts;
  if (!Array.isArray(starts) || starts.length === 0) return [coordinates];
  return starts.map((start, arc) => coordinates.slice(start, arc + 1 < starts.length ? starts[arc + 1] + 1 : coordinates.length));
}

function depotModelId() {
  const features = state.selected?.mapData?.geojson?.features || [];
  return Number(features.find((feature) => feature.properties?.role === "depot")?.properties?.model_node_id ?? 0);
}

function activeRouteGeometry(rendered) {
  if (state.routeView.renderMode === "cached_road" && cachedRoadRenderingAvailable(rendered)) {
    return {
      routeMode: rendered.summary.render_mode,
      routeLines: rendered.geojson.features.map((feature, routeIndex) => ({
        routeIndex,
        // Model ids, as the timeline's stops name them.
        sequence: [depotModelId(), ...(state.selected?.solved?.routes?.[routeIndex] || []).map(Number), depotModelId()],
        coordinates: feature.geometry?.coordinates || [],
        segments: roadSegments(feature),
        properties: feature.properties || {},
      })),
    };
  }
  return { routeMode: "straight_line", routeLines: straightRouteLines(rendered) };
}

function drawRoutes(rendered, fit) {
  previewLayers.clearLayers();
  const routes = state.selected?.solved?.routes || [];
  // Geometry is rebuilt (and handed to MapLibre) only when the rendered solution,
  // its instance or the render mode changes; sliders and focus reuse it.
  const key = [rendered, state.selected?.mapData, routes, state.routeView.renderMode];
  const same = Array.isArray(solutionData.key) && solutionData.key.every((value, index) => value === key[index]);
  if (same) {
    applySolutionView();
  } else {
    // Only the render mode changed (the same solution, now on roads or straight lines):
    // an open animation is reopened on the new geometry instead of just closing.
    const reopen = animationPlaying() && Array.isArray(solutionData.key)
      && key.slice(0, 3).every((value, index) => value === solutionData.key[index]);
    const geometry = activeRouteGeometry(rendered);
    const { nodes, depot } = instanceNodeData(routes);
    solutionData = { key, routeLines: geometry.routeLines, nodes, depot, routeMode: geometry.routeMode, hasRoutes: routes.length > 0 };
    pushSolution();
    if (reopen) {
      setTimeout(() => {
        if (!routeAnimation && solutionData.key === key) toggleRouteAnimation();
      }, 0);
    }
  }
  if (fit) fitMapTo(solutionLayers?.bounds());
  syncAppearanceControls();
}

function redrawRoutes() {
  if (state.previewGeojson) {
    drawPreview(state.previewGeojson);
  } else if (state.lastRendered) {
    drawRoutes(state.lastRendered, false);
  } else if (state.selected?.mapData) {
    drawInstanceNodes(null, false);
  }
  renderLegend();
  syncAppearanceControls();
}

/* ── Visualize tab list ── */
/* Instances are addressed by their index in state.instances everywhere else, so the
   filters narrow a list of [instance, index] pairs rather than reindexing. */
function filteredInstances() {
  const needle = el("inst-search").value.trim().toLowerCase();
  const city = el("inst-city").value;
  const solvedOnly = el("inst-solved").checked;
  const pairs = state.instances
    .map((instance, index) => [instance, index])
    .filter(([instance]) => {
      if (city && instance.city !== city) return false;
      if (solvedOnly && !instance.solution_count) return false;
      if (!needle) return true;
      return `${instance.base_name} ${instance.city}`.toLowerCase().includes(needle);
    });
  const sort = el("inst-sort").value;
  const byName = (a, b) => a[0].base_name.localeCompare(b[0].base_name);
  if (sort === "name") pairs.sort(byName);
  else if (sort === "size") pairs.sort((a, b) => a[0].n - b[0].n || byName(a, b));
  else if (sort === "city") {
    pairs.sort((a, b) => a[0].city.localeCompare(b[0].city) || a[0].n - b[0].n || byName(a, b));
  }
  // "recent" is the order the server already returns, so it needs no sort.
  return pairs;
}

function refreshInstanceCityOptions() {
  const select = el("inst-city");
  const cities = [...new Set(state.instances.map((instance) => instance.city))].sort();
  const current = select.value;
  select.innerHTML = '<option value="">All cities</option>' +
    cities.map((city) => `<option value="${city}">${city}</option>`).join("");
  if (cities.includes(current)) select.value = current;
}

function renderInstanceList() {
  const list = el("instList");
  refreshInstanceCityOptions();
  if (!state.instances.length) {
    el("inst-count").textContent = "";
    list.innerHTML = '<div class="empty-note">Nothing generated yet. Use the Generate tab: generated instances appear here, ready to preview, solve and render on the map.</div>';
    return;
  }
  const pairs = filteredInstances();
  el("inst-count").textContent = pairs.length === state.instances.length
    ? `${pairs.length} instance${pairs.length === 1 ? "" : "s"}`
    : `${pairs.length} of ${state.instances.length}`;
  list.innerHTML = "";
  if (!pairs.length) {
    list.innerHTML = '<div class="empty-note">No instance matches these filters.</div>';
    return;
  }
  pairs.forEach(([instance, index]) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "inst-row" + (state.selected === instance ? " selected" : "");
    const twin = instance.vrptw ? ' <span class="chip chip-acc">VRPTW twin</span>' : "";
    const saved = instance.solution_count
      ? ` <span class="chip chip-gr">${instance.solution_count} saved solution${instance.solution_count === 1 ? "" : "s"}</span>`
      : "";
    row.innerHTML = `<span class="inst-head"><span class="inst-name">${instance.base_name}</span>${twin}${saved}</span>` +
      `<span class="inst-meta">${instance.city} · n=${instance.n} · seed ${shortSeed(instance.seed)} · capacity ${instance.summary.capacity}</span>`;
    row.addEventListener("click", () => selectInstance(index));
    list.append(row);
  });
}

// A machine-drawn 64-bit seed is 20 digits of noise in a list row. The row keeps
// both ends, enough to tell two instances apart; the Details tab shows it whole.
function shortSeed(seed) {
  const text = String(seed ?? "?");
  return text.length > 9 ? `${text.slice(0, 4)}…${text.slice(-3)}` : text;
}

/* ── Selected instance panel ── */
const escapeHtml = (value) => String(value).replace(/[&<>"]/g, (character) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[character]));

/* The workspace path is absolute and, on Windows, backslash-separated — splitting on
   "/" alone left the whole `C:\Users\…\instances\osm\lorient\n51` wrapped over four
   lines and dominating the fact list. Show the last two segments, keep the full path
   on the title and behind a copy button. */
function folderFactHtml(folder) {
  const segments = String(folder).split(/[\\/]/).filter(Boolean);
  const tail = segments.slice(-2).join("/") || String(folder);
  return `<span class="fact-path"><code title="${escapeHtml(folder)}">${escapeHtml(tail)}</code>` +
    `<button type="button" class="copy-btn" data-copy="${escapeHtml(folder)}" title="Copy the full path">Copy</button></span>`;
}

/* The panel keeps what the instance is apart from what is drawn on the map: the
   header and the Details tab describe the instance, the "Displayed solution"
   card and the Routes tab describe the run on the map, the Runs tab lists every
   saved run, and Solve only queues a new one. */
function renderSelected() {
  const instance = state.selected;
  el("selEmpty").hidden = Boolean(instance);
  el("selBody").hidden = !instance;
  if (!instance) {
    closePopover();
    syncAppearanceControls();
    return;
  }
  el("selName").textContent = instance.base_name;
  // "n301" in the name counts the depot; say what the number in the header counts.
  const customers = Number.isFinite(Number(instance.n)) ? `${instance.n} customers` : "customers n/a";
  const benchmark = instance.source === "benchmark";
  const lead = benchmark ? `${instance.problem} · ${instance.family}` : "CVRP";
  el("selMeta").innerHTML = `<span>${escapeHtml(lead)} · ${escapeHtml(instance.city)} · ${escapeHtml(customers)} · capacity ${escapeHtml(instance.summary.capacity ?? "n/a")}</span>`
    + (instance.vrptw ? '<span class="chip chip-acc" title="A VRPTW twin was derived from this instance">VRPTW twin</span>' : "")
    + (benchmark ? `<span class="chip chip-acc" title="Published instance, read-only: your solutions are stored in the workspace">benchmark${instance.summary.revision ? ` ${escapeHtml(instance.summary.revision)}` : ""}</span>` : "")
    + (benchmark && instance.tw_set ? `<span class="chip chip-acc" title="Time-window set">${escapeHtml(instance.tw_set)}</span>` : "");
  // The .vrp export is a CVRP format; a benchmark has no workspace folder to zip.
  el("download-open").hidden = benchmark && instance.problem !== "CVRP";
  el("download").hidden = benchmark;
  el("download-rule").hidden = benchmark;
  renderDetails(instance);
  renderRunCard();
  renderLegend();
  syncAppearanceControls();
}

function renderBenchmarkDetails(instance) {
  const summary = instance.summary || {};
  const variants = instance.files?.vrp_json || {};
  const first = Object.values(variants)[0] || "";
  const facts = [
    ["Family", `${escapeHtml(instance.family)}${summary.revision ? ` · revision ${escapeHtml(summary.revision)}` : ""}`],
    ["Problem", `${escapeHtml(instance.problem)}${instance.tw_set ? ` · time windows: ${escapeHtml(instance.tw_set)}` : ""}`],
    ["City", escapeHtml(instance.city)],
    ["Customers", escapeHtml(instance.n)],
    ["Capacity", escapeHtml(summary.capacity ?? "n/a")],
  ];
  if (summary.route_count != null) facts.push(["Vehicles, lower bound", escapeHtml(summary.route_count)]);
  for (const metric of Object.keys(variants)) {
    const published = (summary.bks?.[metric] || []).join(", ") || "no BKS";
    facts.push([`${metric[0].toUpperCase()}${metric.slice(1)} variant`, `cost in ${escapeHtml(summary.units?.[metric] || "units")} · ${escapeHtml(published)}`]);
  }
  if (summary.profile) facts.push(["Checker profile", escapeHtml(summary.profile)]);
  facts.push(["Folder", folderFactHtml(`${instance.folder}/${first.split("/").slice(0, -1).join("/")}`)]);
  facts.push(["My solutions", escapeHtml(instance.solution_count)]);
  el("selFacts").innerHTML = facts.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
}

function renderDetails(instance) {
  if (instance.source === "benchmark") {
    renderBenchmarkDetails(instance);
    return;
  }
  const summary = instance.summary || {};
  const variants = instance.files?.vrp_json || {};
  const facts = [
    ["Problem", instance.vrptw ? "CVRP, with a derived VRPTW twin" : "CVRP"],
    ["City", escapeHtml(instance.city)],
    ["Customers", escapeHtml(instance.n)],
    ["Capacity", escapeHtml(summary.capacity ?? "n/a")],
    ["Expected routes", summary.route_count != null ? `~${escapeHtml(summary.route_count)}` : "n/a"],
  ];
  // The generator stores the codes of the Generate form; say what they stand for.
  const coded = (labels, code) => (labels.has(Number(code)) ? `${escapeHtml(labels.get(Number(code)))} (${escapeHtml(code)})` : escapeHtml(code));
  if (summary.demand_type != null) facts.push(["Demand type", coded(DEMAND_LABELS, summary.demand_type)]);
  if (summary.avg_route_size != null) facts.push(["Route size band", coded(BAND_LABELS, summary.avg_route_size)]);
  facts.push(["Seed", `<span class="fact-seed">${escapeHtml(instance.seed)}</span>`]);
  facts.push(["Folder", folderFactHtml(instance.folder)]);
  for (const metric of ["fastest", "shortest", "euclidean"]) {
    if (variants[metric]) facts.push([`${metric[0].toUpperCase()}${metric.slice(1)} variant`, `<code class="fact-file">${escapeHtml(String(variants[metric]).split(/[\\/]/).pop())}</code>`]);
  }
  facts.push(["Saved solutions", escapeHtml(instance.solution_count)]);
  el("selFacts").innerHTML = facts.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
}

function routeVisibilityIcon(hidden) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z"/><circle cx="12" cy="12" r="2.5"/>${hidden ? '<path class="eye-slash" d="M4 4l16 16"/>' : ""}</svg>`;
}

function renderLegend() {
  const legend = el("legend");
  const instance = state.selected;
  const drawn = Boolean(instance && instance.solved && state.lastRendered);
  el("legendTools").hidden = !drawn;
  if (!drawn) {
    el("legendTitle").textContent = "Routes";
    const saved = instance ? allSolutionRecords().length : 0;
    legend.innerHTML = '<div class="empty-note">No solution is drawn: the map shows the depot and the customers.'
      + (instance
        ? `<div class="row empty-actions">${saved ? '<button type="button" class="mini-btn" data-panel-action="runs">Choose a saved solution</button>' : ""}`
          + '<button type="button" class="mini-btn" data-panel-action="solve">Solve it</button>'
          + '<button type="button" class="mini-btn" data-panel-action="import">Import a solution</button></div>'
        : "")
      + "</div>";
    syncAppearanceControls();
    return;
  }
  const features = state.lastRendered.geojson.features;
  const shown = features.filter((_, index) => !state.hiddenRoutes.has(index)).length;
  el("legendTitle").textContent = shown === features.length ? `Routes · ${features.length}` : `Routes · ${shown} of ${features.length} shown`;
  const capacity = Number(instance.summary.capacity);
  legend.innerHTML = "";
  features.forEach((feature, index) => {
    const hidden = state.hiddenRoutes.has(index);
    const focused = state.focusedRoute === index;
    const row = document.createElement("div");
    row.className = `legend-row${hidden ? " off" : ""}${focused ? " focused" : ""}`;
    const eye = document.createElement("button");
    eye.type = "button";
    eye.className = "legend-eye";
    eye.title = `${hidden ? "Show" : "Hide"} route ${index + 1}`;
    eye.setAttribute("aria-label", eye.title);
    eye.innerHTML = routeVisibilityIcon(hidden);
    eye.addEventListener("click", () => {
      if (state.hiddenRoutes.has(index)) {
        state.hiddenRoutes.delete(index);
      } else {
        state.hiddenRoutes.add(index);
      }
      if (state.focusedRoute === index && state.hiddenRoutes.has(index)) state.focusedRoute = null;
      redrawRoutes();
    });
    const focus = document.createElement("button");
    focus.type = "button";
    focus.className = "legend-focus";
    focus.title = focused ? "Clear the focus" : "Focus this route: the others fade";
    focus.setAttribute("aria-pressed", String(focused));
    focus.setAttribute("aria-label", `${focused ? "Clear focus from" : "Focus"} route ${index + 1}`);
    const load = Number(feature.properties.load);
    // A bar says at a glance how full each vehicle is; the number stays for the exact load.
    const loadHtml = capacity > 0 && Number.isFinite(load)
      ? `<span class="load" title="Load ${load} of capacity ${capacity}"><span class="load-bar"><i style="width:${Math.min(100, (100 * load) / capacity).toFixed(1)}%"></i></span><span class="load-num">${load}</span></span>`
      : `<span class="load"><span class="load-num">load ${escapeHtml(feature.properties.load ?? "?")}</span></span>`;
    focus.innerHTML = `<span class="dot" style="background:${routeColor(index)}"></span>` +
      `<span class="lbl">R${feature.properties.route_index}</span>` +
      `<span class="meta">${feature.properties.stops} stops</span>` + loadHtml;
    focus.addEventListener("click", () => {
      state.hiddenRoutes.delete(index);
      state.focusedRoute = state.focusedRoute === index ? null : index;
      redrawRoutes();
    });
    row.append(eye, focus);
    legend.append(row);
  });
  syncAppearanceControls();
}

el("routes-all").addEventListener("click", () => {
  state.hiddenRoutes.clear();
  redrawRoutes();
});
el("routes-none").addEventListener("click", () => {
  (state.lastRendered?.geojson.features || []).forEach((_, index) => state.hiddenRoutes.add(index));
  state.focusedRoute = null;
  redrawRoutes();
});

function selectInstance(index, preferredRunId = null) {
  return activateInstance(state.instances[index], preferredRunId);
}

/* Makes `instance` the selected one: a workspace instance or a benchmark's. */
function activateInstance(instance, preferredRunId = null) {
  state.selected = instance;
  state.selected.solved = null;
  state.selected.rendered = null;
  state.solutionRuns = [];
  state.solutionReferences = [];
  state.selectedRunId = null;
  state.previewGeojson = null;
  state.lastRendered = null;
  resetRouteViewDefaults();
  previewLayers.clearLayers();
  instanceNodeLayers.clearLayers();
  clearRouteLayers();
  // A half-filled import and a comparison belong to the instance they were opened for.
  el("import-panel").hidden = true;
  state.comparison = null;
  renderInstanceList();
  renderBenchmarkList();
  syncMetricOptions();
  renderSelected();
  loadSolvers().catch((error) => console.warn("Unable to list solvers", error));
  return loadSolutionRuns(preferredRunId).catch((error) => {
    status(`Cannot load saved solutions: ${error.message}`);
  });
}

async function loadInstanceMapData(instance = state.selected) {
  if (!instance) return null;
  if (!instance.mapData) {
    instance.mapData = await api(`/api/instances/${encodeURIComponent(instance.instance_id)}/map-data`);
  }
  return state.selected === instance ? instance.mapData : null;
}

async function displayInstanceOnly(fit = true) {
  const instance = state.selected;
  if (!instance) return;
  await loadInstanceMapData(instance);
  if (state.selected !== instance) return;
  state.selectedRunId = null;
  instance.solved = null;
  instance.rendered = null;
  state.lastRendered = null;
  resetRouteViewDefaults();
  state.previewGeojson = null;
  previewLayers.clearLayers();
  clearRouteLayers();
  drawInstanceNodes(null, fit);
  refreshSolutionControls();
  renderSelected();
  status(`Displayed ${instance.base_name}: depot and ${instance.n} customer positions.`);
}

/* ── Benchmarks: published instances as a second, read-only source ── */
/* The same panel serves them: a solution found elsewhere is imported, checked by
   the reference checker, stored in the workspace and drawn, with the published
   BKS as the reference to compare against. */
const placeName = (slug) => String(slug || "").split("_").filter(Boolean)
  .map((word) => word[0].toUpperCase() + word.slice(1)).join(" ");

function setInstanceSource(source) {
  state.instanceSource = source;
  const benchmarks = source === "benchmarks";
  el("workspaceSource").hidden = benchmarks;
  el("benchmarkSource").hidden = !benchmarks;
  for (const [id, active] of [["sourceWorkspace", !benchmarks], ["sourceBenchmarks", benchmarks]]) {
    el(id).classList.toggle("tab-active", active);
    el(id).setAttribute("aria-pressed", String(active));
  }
  if (benchmarks && !state.benchmarks.loaded && !state.benchmarks.loading) {
    loadBenchmarks().catch((error) => {
      state.benchmarks.loading = false;
      el("benchList").innerHTML = `<div class="empty-note">Cannot list the benchmarks: ${escapeHtml(error.message)}</div>`;
    });
  }
}

async function loadBenchmarks(refresh = false) {
  state.benchmarks.loading = true;
  el("benchList").innerHTML = '<div class="empty-note">Reading the benchmarks folder…</div>';
  const data = await api(`/api/workbench/benchmarks${refresh ? "?refresh=true" : ""}`);
  Object.assign(state.benchmarks, { loaded: true, loading: false, instances: data.instances || [], hint: data.hint, root: data.root });
  const scopes = benchmarkScopes();
  if (!scopes.some((scope) => scope.key === state.benchmarks.scope)) state.benchmarks.scope = scopes[0]?.key || null;
  renderBenchmarkFilters();
  renderBenchmarkList();
}

/* A scope is a family under one problem class: what a solution is checked against. */
function benchmarkScopes() {
  const scopes = new Map();
  for (const entry of state.benchmarks.instances) {
    const key = `${entry.family}|${entry.problem_type}`;
    if (!scopes.has(key)) scopes.set(key, { key, family: entry.family, problem: entry.problem_type, count: 0 });
    scopes.get(key).count += 1;
  }
  return [...scopes.values()];
}

function scopedBenchmarks() {
  return state.benchmarks.instances.filter((entry) => `${entry.family}|${entry.problem_type}` === state.benchmarks.scope);
}

function renderBenchmarkFilters() {
  const scopes = benchmarkScopes();
  el("bench-scope").hidden = scopes.length < 2;
  el("bench-scope").innerHTML = scopes
    .map((scope) => `<button type="button" class="tab-btn${scope.key === state.benchmarks.scope ? " tab-active" : ""}" data-bench-scope="${escapeHtml(scope.key)}" aria-pressed="${scope.key === state.benchmarks.scope}" title="${scope.count} instances">${escapeHtml(scope.family)} ${escapeHtml(scope.problem)}</button>`)
    .join("");
  const scoped = scopedBenchmarks();
  const fill = (id, all, values, label) => {
    const select = el(id);
    const previous = select.value;
    select.innerHTML = "";
    select.append(new Option(all, ""));
    values.forEach((value) => select.append(new Option(label(value), String(value))));
    select.value = values.map(String).includes(previous) ? previous : "";
  };
  fill("bench-city", "All cities", [...new Set(scoped.map((entry) => entry.city))].sort(), placeName);
  const sizes = [...new Set(scoped.map((entry) => entry.n_customers))].sort((a, b) => a - b);
  // A family with one size per city has as many sizes as instances: the search does that job better.
  el("bench-size-field").hidden = sizes.length > 12;
  fill("bench-size", "All sizes", sizes.length > 12 ? [] : sizes, (value) => `${value} customers`);
}

function filteredBenchmarks() {
  const needle = el("bench-search").value.trim().toLowerCase();
  const city = el("bench-city").value;
  const size = el("bench-size").value;
  const mine = el("bench-solved").checked;
  return scopedBenchmarks().filter((entry) => {
    if (city && entry.city !== city) return false;
    if (size && String(entry.n_customers) !== size) return false;
    if (mine && !entry.solution_count) return false;
    return !needle || `${entry.base_name} ${entry.city} ${placeName(entry.city)}`.toLowerCase().includes(needle);
  });
}

function renderBenchmarkList() {
  const list = el("benchList");
  if (!state.benchmarks.loaded) return;
  const scoped = scopedBenchmarks();
  if (!state.benchmarks.instances.length) {
    el("bench-count").textContent = "";
    list.innerHTML = `<div class="empty-note">${escapeHtml(state.benchmarks.hint || "No published instance was found.")}</div>`;
    return;
  }
  const entries = filteredBenchmarks();
  el("bench-count").textContent = entries.length === scoped.length
    ? `${entries.length} instance${entries.length === 1 ? "" : "s"}`
    : `${entries.length} of ${scoped.length}`;
  if (!entries.length) {
    list.innerHTML = '<div class="empty-note">No instance matches these filters.</div>';
    return;
  }
  list.innerHTML = entries.map((entry) => {
    const selected = state.selected?.instance_id === entry.instance_id;
    const windows = entry.tw_set ? ` <span class="chip chip-acc" title="Time-window set">${escapeHtml(entry.tw_set)}</span>` : "";
    const mine = entry.solution_count
      ? ` <span class="chip chip-gr">${entry.solution_count} of mine</span>`
      : "";
    return `<button type="button" class="inst-row${selected ? " selected" : ""}" data-bench-id="${escapeHtml(entry.instance_id)}">`
      + `<span class="inst-head"><span class="inst-name">${escapeHtml(entry.base_name)}</span>${windows}${mine}</span>`
      + `<span class="inst-meta">${escapeHtml(placeName(entry.city))} · ${escapeHtml(entry.n_customers)} customers · ${escapeHtml(entry.metrics.join(", "))}</span></button>`;
  }).join("");
}

/* The panel's instance object for a benchmark entry, kept so its map data is fetched once. */
async function selectBenchmark(instanceId, preferredRunId = null) {
  let instance = state.benchmarkObjects.get(instanceId);
  if (!instance) {
    const data = await api(`/api/workbench/benchmarks/${encodeURIComponent(instanceId)}`);
    const entry = data.instance;
    instance = {
      instance_id: entry.instance_id,
      source: "benchmark",
      family: entry.family,
      problem: entry.problem_type,
      tw_set: entry.tw_set,
      base_name: entry.base_name,
      folder: entry.folder,
      files: entry.files,
      summary: entry.summary || {},
      vrptw: null,
      city: placeName(entry.city),
      n: entry.n_customers ?? "?",
      seed: null,
      solution_count: entry.solution_count || 0,
      solved: null,
      rendered: null,
      mapData: null,
    };
    state.benchmarkObjects.set(instanceId, instance);
  }
  return activateInstance(instance, preferredRunId);
}

el("sourceWorkspace").addEventListener("click", () => setInstanceSource("workspace"));
el("sourceBenchmarks").addEventListener("click", () => setInstanceSource("benchmarks"));
el("bench-scope").addEventListener("click", (event) => {
  const button = event.target instanceof Element ? event.target.closest("[data-bench-scope]") : null;
  if (!button) return;
  // City and size carry over between the problem classes of one family, not between
  // families: a city both have would silently hide the rest of the other one.
  if (button.dataset.benchScope.split("|")[0] !== String(state.benchmarks.scope).split("|")[0]) {
    el("bench-city").value = "";
    el("bench-size").value = "";
  }
  state.benchmarks.scope = button.dataset.benchScope;
  renderBenchmarkFilters();
  renderBenchmarkList();
});
["bench-search", "bench-city", "bench-size", "bench-solved"].forEach((id) => el(id).addEventListener("input", renderBenchmarkList));
el("benchList").addEventListener("click", (event) => {
  const row = event.target instanceof Element ? event.target.closest("[data-bench-id]") : null;
  if (!row) return;
  selectBenchmark(row.dataset.benchId).catch((error) => status(`Cannot open the benchmark instance: ${error.message}`));
});

/* Not every instance has the three metric variants (a VRPTW benchmark is fastest only):
   the metric selects offer what the selected instance has. */
function syncMetricOptions() {
  const available = Object.keys(state.selected?.files?.vrp_json || {});
  for (const id of ["solve-metric", "import-metric", "download-metric"]) {
    const select = el(id);
    for (const option of select.options) option.disabled = available.length > 0 && !available.includes(option.value);
    if (select.selectedOptions[0]?.disabled) select.value = available[0];
  }
  syncVrpExportControls();
}

/* ── Persistent solutions ── */
/* Runs of an earlier version of the instance (it was regenerated or replaced)
   are "stale": the server flags them and refuses to render or compare them, so
   they stay out of both selectors and are only counted. */
function allSolutionRecords() {
  return [...state.solutionRuns.filter((run) => !run.stale), ...state.solutionReferences];
}

function staleRunCount() {
  return state.solutionRuns.filter((run) => run.stale).length;
}

/* The metric a run was produced on. Every instance ships shortest/fastest/
   euclidean variants, so a cost is meaningless without it. */
function runMetric(run) {
  return run?.metadata?.metric || "unknown metric";
}

/* Where a run comes from, and when: "pyvrp · pyvrp-ils-v1", "Imported · LKH-3", "BKS". */
function runOrigin(run) {
  if (run.source === "bks") return "BKS";
  if (run.source === "imported") return `Imported${run.solver && run.solver !== "external" ? ` · ${run.solver}` : ""}`;
  const solver = run.solver || "pyvrp";
  return run.method && run.method !== solver ? `${solver} · ${run.method}` : solver;
}

function runDate(run) {
  const date = run.created_at ? new Date(run.created_at) : null;
  return date && !Number.isNaN(date.getTime())
    ? date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
    : "";
}

function runWhere(run) {
  return [runOrigin(run), runDate(run)].filter(Boolean).join(" · ");
}

function runProblem(run) {
  if (state.selected?.source === "benchmark") return state.selected.problem;
  return run.instance_path && /cvrptw/i.test(String(run.instance_path)) ? "VRPTW" : "CVRP";
}

/* Costs only compare between runs of the same variant under the same objective,
   so that is what a group of the run list is. */
function runGroupLabel(run) {
  return `${runMetric(run)} · ${run.objective_function || "MonoCost"}${runProblem(run) === "VRPTW" ? " · VRPTW" : ""}`;
}

function runIsHierarchical(run) {
  return /hierarchical/i.test(String(run.objective_function || ""));
}

function runIsRanked(run) {
  return Boolean(run.validation?.valid) && run.cost != null && Number.isFinite(Number(run.cost));
}

/* Best first: valid runs before invalid ones, fewer routes first under a
   hierarchical objective, then the lower cost. */
function compareRunQuality(a, b) {
  const rankedA = runIsRanked(a);
  const rankedB = runIsRanked(b);
  if (rankedA !== rankedB) return rankedA ? -1 : 1;
  if (!rankedA) return 0;
  if (runIsHierarchical(a) && a.num_routes !== b.num_routes) return a.num_routes - b.num_routes;
  return Number(a.cost) - Number(b.cost);
}

function runGroups() {
  const groups = new Map();
  for (const run of allSolutionRecords()) {
    const label = runGroupLabel(run);
    if (!groups.has(label)) groups.set(label, { label, runs: [] });
    groups.get(label).runs.push(run);
  }
  for (const group of groups.values()) {
    group.runs.sort(compareRunQuality);
    group.best = runIsRanked(group.runs[0]) ? group.runs[0] : null;
  }
  return [...groups.values()];
}

/* How far a run is from the best of its group: a relative gap, or a number of
   routes when the objective puts the fleet first. */
function runGapLabel(run, group) {
  const best = group.best;
  if (!best || !runIsRanked(run) || group.runs.length < 2) return "";
  if (run === best) return "best";
  if (runIsHierarchical(run) && run.num_routes !== best.num_routes) {
    const extra = run.num_routes - best.num_routes;
    return `+${extra} route${extra === 1 ? "" : "s"}`;
  }
  const reference = Number(best.cost);
  if (!(reference > 0)) return "";
  const gap = (100 * (Number(run.cost) - reference)) / reference;
  return gap === 0 ? "= best" : `+${gap.toFixed(gap < 10 ? 2 : 1)}%`;
}

function displayedRun() {
  return state.selectedRunId ? state.selected?.solved || null : null;
}

/* The metric the user is working on: the displayed run's, else the Solve tab's. */
function preferredMetric() {
  const metric = displayedRun()?.metadata?.metric;
  return ["fastest", "shortest", "euclidean"].includes(metric) ? metric : el("solve-metric").value;
}

function renderRunCard() {
  const instance = state.selected;
  if (!instance) return;
  const records = allSolutionRecords();
  const run = displayedRun();
  el("run-switch").hidden = records.length === 0;
  const line = el("solution-validation");
  line.hidden = true;
  const body = el("runCardBody");
  if (!run) {
    body.innerHTML = '<div class="run-card-none">Instance only</div>'
      + `<div class="run-card-line">${records.length
        ? `Depot and customers. ${records.length} saved solution${records.length === 1 ? "" : "s"} can be drawn.`
        : "Depot and customers. No saved solution yet."}</div>`
      + (instance.source === "benchmark"
        ? '<div class="row run-card-actions"><button type="button" class="btn btn-secondary" data-panel-action="import">Check my solution</button></div>'
        : "");
    return;
  }
  const validation = run.validation || {};
  const metric = runMetric(run);
  const chip = validation.recorded
    ? `<span class="chip chip-gr" title="Published BKS: validated when it entered the collection${validation.validated_by ? ` (${escapeHtml(validation.validated_by)})` : ""}.">published</span>`
    : validation.valid
      ? `<span class="chip chip-gr" title="Validated on the ${escapeHtml(metric)} variant: canonical cost ${escapeHtml(validation.routing_cost)}, ${escapeHtml(validation.num_routes)} route(s).">validated</span>`
      : '<span class="chip chip-cor">invalid</span>';
  body.innerHTML = `<div class="sel-cost-row"><span class="sel-cost" id="selCost">${escapeHtml(run.cost ?? "n/a")}</span>`
    + `<span class="run-card-routes">${escapeHtml(run.num_routes)} route${run.num_routes === 1 ? "" : "s"}</span>${chip}</div>`
    + `<div class="run-card-line">${escapeHtml(runGroupLabel(run))}</div>`
    + `<div class="run-card-line">${escapeHtml(runWhere(run))}</div>`
    + bksStandingHtml(run);
  if (!validation.valid) {
    line.hidden = false;
    line.className = "validation-line invalid";
    line.textContent = `Invalid: ${validation.error_message || validation.status || "checker rejected this solution"}`;
  }
}

/* Where a displayed solution stands against the published BKS of its variant and objective. */
function bksStandingHtml(run) {
  if (run.source === "bks" || !runIsRanked(run)) return "";
  const label = runGroupLabel(run);
  const bks = allSolutionRecords().find((other) => other.source === "bks" && runIsRanked(other) && runGroupLabel(other) === label);
  if (!bks) return "";
  const order = compareRunQuality(run, bks);
  const share = Number(bks.cost) > 0 ? Math.abs((100 * (Number(run.cost) - Number(bks.cost))) / Number(bks.cost)) : 0;
  const amount = runIsHierarchical(run) && run.num_routes !== bks.num_routes
    ? `${Math.abs(run.num_routes - bks.num_routes)} route${Math.abs(run.num_routes - bks.num_routes) === 1 ? "" : "s"}`
    : `${share.toFixed(share < 10 ? 2 : 1)}%`;
  if (order < 0) {
    return `<div class="run-card-line run-card-better">Better than the published BKS by ${escapeHtml(amount)}. It can be submitted with <code>mamut-routing bks submit</code>.</div>`;
  }
  return `<div class="run-card-line">${order === 0 ? "Equals the published BKS." : `${escapeHtml(amount)} above the published BKS.`}</div>`;
}

function runRowHtml(run, group, { compare }) {
  const current = displayedRun();
  const displayed = run.run_id === state.selectedRunId;
  const gap = runGapLabel(run, group);
  const canCompare = compare && !displayed && current && runIsRanked(run) && runIsRanked(current) && group.runs.includes(current);
  return `<div class="run-row${displayed ? " selected" : ""}">`
    + `<button type="button" class="run-pick" data-run-id="${escapeHtml(run.run_id)}" aria-pressed="${displayed}" title="${displayed ? "Displayed on the map" : "Display this solution on the map"}">`
    + `<span class="run-cost">${escapeHtml(run.cost ?? "n/a")}</span>`
    + `<span class="run-routes">${escapeHtml(run.num_routes)} route${run.num_routes === 1 ? "" : "s"}</span>`
    + `<span class="run-gap">${run.validation?.valid ? escapeHtml(gap) : '<span class="chip chip-cor">invalid</span>'}</span>`
    + `<span class="run-origin">${escapeHtml(runWhere(run))}</span></button>`
    + (canCompare ? `<button type="button" class="mini-btn run-compare" data-compare-id="${escapeHtml(run.run_id)}" title="Compare the displayed solution against this one">Compare</button>` : "")
    + "</div>";
}

function runListHtml(options) {
  const groups = runGroups();
  if (!groups.length) return "";
  const none = `<div class="run-row${state.selectedRunId ? "" : " selected"}">`
    + `<button type="button" class="run-pick run-pick-none" data-run-id="" aria-pressed="${!state.selectedRunId}">`
    + '<span class="run-cost">Instance only</span><span class="run-origin">Depot and customers, no routes</span></button></div>';
  return none + groups
    .map((group) => `<div class="run-group">${escapeHtml(group.label)}</div>${group.runs.map((run) => runRowHtml(run, group, options)).join("")}`)
    .join("");
}

function renderRunLists() {
  const records = allSolutionRecords();
  el("runsTitle").textContent = records.length ? `Saved solutions · ${records.length}` : "Saved solutions";
  el("runList").innerHTML = records.length
    ? runListHtml({ compare: true })
    : '<div class="empty-note">No saved solution yet. Queue one in the Solve tab, or import a solution found elsewhere.</div>';
  el("runPopoverList").innerHTML = runListHtml({ compare: false });
  const stale = staleRunCount();
  const note = el("runsNote");
  note.hidden = stale === 0;
  note.textContent = stale
    ? `${stale} stale run${stale === 1 ? "" : "s"} of an earlier version of this instance (regenerated since) hidden.`
    : "";
  // A comparison belongs to the run it was made for.
  const comparison = state.comparison;
  const output = el("comparison-output");
  if (comparison && comparison.instanceId === state.selected?.instance_id && comparison.candidateId === state.selectedRunId) {
    output.hidden = false;
    output.innerHTML = comparison.html;
  } else {
    state.comparison = null;
    output.hidden = true;
    output.innerHTML = "";
  }
}

function refreshSolutionControls() {
  if (state.selectedRunId && !allSolutionRecords().some((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = null;
  }
  renderRunCard();
  renderRunLists();
}

async function displaySolutionRun(runId, fit = true) {
  const instance = state.selected;
  if (!instance) return;
  if (!runId) {
    await displayInstanceOnly(fit);
    return;
  }
  const run = allSolutionRecords().find((value) => value.run_id === runId);
  if (!run) return;
  await loadInstanceMapData(instance);
  if (state.selected !== instance) return;
  state.selectedRunId = runId;
  instance.solved = run;
  const metric = run.metadata?.metric || el("solve-metric").value || "fastest";
  if (el("solve-metric").value !== metric) {
    el("solve-metric").value = metric;
    // Setting a select's value fires no change event: do what its listeners would.
    syncVrpExportControls();
    loadSolvers().catch(() => {});
  }
  const rendered = await api(`/api/instances/${encodeURIComponent(instance.instance_id)}/solutions/${encodeURIComponent(runId)}/render?metric=${encodeURIComponent(metric)}`, {});
  if (state.selected !== instance) return;
  instance.rendered = rendered;
  state.lastRendered = rendered;
  state.previewGeojson = null;
  resetRouteViewDefaults(rendered);
  previewLayers.clearLayers();
  drawRoutes(rendered, fit);
  refreshSolutionControls();
  renderSelected();
  status(`Displayed ${run.source === "bks" ? "BKS" : "saved run"}: cost ${run.cost}, ${run.num_routes} routes; ${rendered.summary.render_mode}.`);
}

async function loadSolutionRuns(preferredRunId = null) {
  const instance = state.selected;
  if (!instance) return;
  const [data] = await Promise.all([
    api(`/api/instances/${encodeURIComponent(instance.instance_id)}/solutions`),
    loadInstanceMapData(instance),
  ]);
  if (state.selected !== instance) return;
  state.solutionRuns = data.runs || [];
  state.solutionReferences = data.references || [];
  instance.solution_count = state.solutionRuns.filter((run) => !run.stale).length;
  state.selectedRunId = preferredRunId;
  renderInstanceList();
  refreshSolutionControls();
  if (state.selectedRunId) {
    await displaySolutionRun(state.selectedRunId, true);
  } else {
    await displayInstanceOnly(true);
  }
}

/* ── Run list, run switcher and comparison ── */
function comparisonHtml(value, reference) {
  const gap = value.relative_gap_percent == null ? "n/a" : `${value.relative_gap_percent > 0 ? "+" : ""}${value.relative_gap_percent.toFixed(3)}%`;
  // A difference of two binary64 costs carries rounding noise (74.36700000000002): twelve digits say all there is.
  const signed = (number) => {
    const clean = Number.isInteger(number) ? number : Number(number.toPrecision(12));
    return clean > 0 ? `+${clean}` : String(clean);
  };
  const extent = (loads) => (loads.length ? `${Math.min(...loads)}–${Math.max(...loads)}` : "n/a");
  const verdict = { better: "better than", worse: "worse than", equal: "equal to" }[value.ordering] || value.ordering;
  const difference = value.route_difference;
  return '<div class="comparison-head"><span class="kicker">Comparison</span>'
    + '<button type="button" class="popover-close" data-comparison-close aria-label="Close the comparison"><svg class="icon" aria-hidden="true"><use href="#i-close"/></svg></button></div>'
    + `<div class="comparison-verdict">The displayed solution is <strong>${escapeHtml(verdict)}</strong> ${escapeHtml(runWhere(reference))} <span class="comparison-profile">(${escapeHtml(value.profile)})</span></div>`
    + '<table class="comparison-table"><thead><tr><th></th><th>Displayed</th><th>Reference</th><th>Difference</th></tr></thead><tbody>'
    // Each cell names its column: in a narrow panel the table stacks and the head row is gone (workbench.css).
    + `<tr><th>Cost</th><td data-label="Displayed">${escapeHtml(value.candidate.cost)}</td><td data-label="Reference">${escapeHtml(value.reference.cost)}</td><td data-label="Difference"><span class="comparison-value">${value.cost_delta == null ? "outside binary64" : escapeHtml(signed(value.cost_delta))} <span class="comparison-profile">${escapeHtml(gap)}</span></span></td></tr>`
    + `<tr><th>Routes</th><td data-label="Displayed">${value.candidate.num_routes}</td><td data-label="Reference">${value.reference.num_routes}</td><td data-label="Difference">${escapeHtml(signed(value.route_count_delta))}</td></tr>`
    + `<tr><th>Loads</th><td data-label="Displayed">${escapeHtml(extent(value.candidate.route_loads))}</td><td data-label="Reference">${escapeHtml(extent(value.reference.route_loads))}</td><td></td></tr>`
    + "</tbody></table>"
    + '<dl class="sel-facts comparison-facts">'
    + `<dt>Directed edges</dt><dd>+${difference.directed_edges_added} / −${difference.directed_edges_removed}</dd>`
    + `<dt>Customer pairs on one route</dt><dd>+${difference.co_routed_customer_pairs_added} / −${difference.co_routed_customer_pairs_removed}</dd>`
    + "</dl>";
}

async function compareWith(referenceId) {
  const instance = state.selected;
  const candidateId = state.selectedRunId;
  if (!instance || !candidateId || !referenceId) return;
  const output = el("comparison-output");
  const keep = (html) => {
    if (state.selected !== instance || state.selectedRunId !== candidateId) return;
    state.comparison = { instanceId: instance.instance_id, candidateId, html };
    output.hidden = false;
    output.innerHTML = html;
    output.scrollIntoView({ block: "nearest" });
  };
  try {
    const data = await api(`/api/instances/${encodeURIComponent(instance.instance_id)}/solutions/compare`, {
      candidate_run_id: candidateId,
      reference_run_id: referenceId,
    });
    const reference = allSolutionRecords().find((run) => run.run_id === referenceId);
    keep(comparisonHtml(data.comparison, reference || {}));
  } catch (error) {
    keep(`<div class="validation-line invalid">Comparison failed: ${escapeHtml(error.message)}</div>`);
  }
}

function onRunListClick(event) {
  const target = event.target instanceof Element ? event.target : null;
  const compare = target?.closest("[data-compare-id]");
  if (compare) {
    compareWith(compare.dataset.compareId);
    return;
  }
  const pick = target?.closest("[data-run-id]");
  if (!pick) return;
  closePopover();
  displaySolutionRun(pick.dataset.runId, true).catch((error) => status(`Cannot display solution: ${error.message}`));
}
el("runList").addEventListener("click", onRunListClick);
el("runPopoverList").addEventListener("click", onRunListClick);

el("comparison-output").addEventListener("click", (event) => {
  if (!(event.target instanceof Element) || !event.target.closest("[data-comparison-close]")) return;
  state.comparison = null;
  el("comparison-output").hidden = true;
});

/* Popovers of the selected-instance panel open beside the panel, over the map,
   so they never cover what they act on. */
function togglePanelPopover(popoverId, button, beforeOpen) {
  const popover = el(popoverId);
  if (!popover.hidden) {
    closePopover();
    return;
  }
  beforeOpen?.();
  showPopover(popover, button, () => {
    popover.hidden = true;
    button.setAttribute("aria-expanded", "false");
  }, document.querySelector(".panel-right"));
  button.setAttribute("aria-expanded", "true");
}
el("run-switch").addEventListener("click", (event) => togglePanelPopover("run-popover", event.currentTarget));
el("appearance-open").addEventListener("click", (event) => togglePanelPopover("route-appearance", event.currentTarget));
el("download-open").addEventListener("click", (event) => togglePanelPopover("download-popover", event.currentTarget, () => {
  el("download-metric").value = preferredMetric();
  syncVrpExportControls();
}));
for (const id of ["run-popover-close", "appearance-close", "download-close"]) {
  el(id).addEventListener("click", closePopover);
}

/* The shortcuts of an empty state: "Solve it", "Import a solution", "Choose a saved solution". */
document.querySelector(".panel-right").addEventListener("click", (event) => {
  const action = event.target instanceof Element ? event.target.closest("[data-panel-action]")?.dataset.panelAction : null;
  if (!action) return;
  if (action === "import") {
    activateInspectorTab("runs");
    openImportPanel();
  } else {
    activateInspectorTab(action);
  }
});

/* ── Persistent jobs ── */
const ACTIVE_JOB_STATUSES = ["queued", "running"];

/* Buttons that own a job kind: while one is in flight the button says so and refuses a
   second click. Before this, clicking Queue solve five times queued five solves with no
   sign that any of them had started. */
const BUSY_BUTTONS = [
  { id: "solve", kind: "solve", verb: "Solving" },
  { id: "generate", kind: "generate", verb: "Generating" },
  { id: "fetch", kind: "fetch-osm", verb: "Fetching" },
  { id: "osm-audit-refresh", kind: "refresh-pois", verb: "Updating" },
];
const busyButtonState = new Map();

function formatClock(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/* Seconds since the job started — or since it was created, while it still waits for a
   worker. The 1 Hz poll re-renders anyway, so the clock ticks without a timer of its own. */
function jobElapsed(job) {
  const started = Date.parse(job.started_at || job.created_at || "");
  if (!Number.isFinite(started)) return null;
  const finished = job.finished_at ? Date.parse(job.finished_at) : NaN;
  const end = Number.isFinite(finished) ? finished : Date.now();
  return Math.max(0, (end - started) / 1000);
}

/* A fraction in [0,1] when the job knows its own extent, otherwise null — the caller draws
   an indeterminate bar rather than inventing a number. */
function jobFraction(job) {
  const total = job.progress?.total;
  if (!total || !(total > 0)) return null;
  return Math.min(1, Math.max(0, (job.progress.current || 0) / total));
}

/* "0:18 / 0:30" for a time budget, "3/12" for a count of things. */
function jobCounter(job) {
  const progress = job.progress || {};
  if (!progress.total) return "";
  if (progress.unit === "s") return `${formatClock(progress.current || 0)} / ${formatClock(progress.total)}`;
  return `${progress.current || 0}/${progress.total}`;
}

function makeBar(className) {
  const bar = document.createElement("div");
  bar.className = className;
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.append(document.createElement("i"));
  return bar;
}

function setBar(bar, job) {
  const active = ACTIVE_JOB_STATUSES.includes(job.status);
  bar.hidden = !active;
  if (!active) return;
  const fraction = jobFraction(job);
  bar.classList.toggle("indeterminate", fraction === null);
  bar.firstChild.style.width = fraction === null ? "" : `${(fraction * 100).toFixed(1)}%`;
  if (fraction === null) bar.removeAttribute("aria-valuenow");
  else bar.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
}

/* The collapsed strip has to say enough that opening it is usually unnecessary. */
function renderActivitySummary() {
  const active = state.jobs.filter((job) => ACTIVE_JOB_STATUSES.includes(job.status));
  const failed = state.jobs.filter((job) => job.status === "failed");
  const dot = el("activity-dot");
  // Red means "the last thing you ran failed", not "this workspace has ever failed" —
  // the job history is permanent, so counting all failures pins the light red forever.
  const lastFailed = state.jobs.length > 0 && state.jobs[0].status === "failed";
  dot.classList.toggle("busy", active.length > 0);
  dot.classList.toggle("bad", active.length === 0 && lastFailed);

  const parts = [];
  const first = active[0];
  if (first) {
    const elapsed = jobElapsed(first);
    parts.push([
      active.length > 1 ? `${active.length} running · ${first.kind}` : first.kind,
      first.progress?.message,
      jobCounter(first) || (elapsed === null ? "" : formatClock(elapsed)),
    ].filter(Boolean).join(" · "));
  }
  if (failed.length) parts.push(`${failed.length} failed`);
  el("activity-count").textContent = parts.join(" · ");

  const bar = el("activity-bar");
  if (first) setBar(bar, first);
  else bar.hidden = true;

  // The rail survives collapsing the left panel, so the running count has to live there too.
  const rail = document.querySelector('[data-rail-target="jobs"]');
  if (rail) {
    if (active.length) {
      rail.dataset.jobCount = String(active.length);
      rail.title = `Jobs · ${active.length} running`;
    } else {
      delete rail.dataset.jobCount;
      rail.title = "Jobs";
    }
  }
}

function renderBusyButtons() {
  BUSY_BUTTONS.forEach(({ id, kind, verb }) => {
    const button = el(id);
    if (!button) return;
    const job = state.jobs.find((entry) => entry.kind === kind && ACTIVE_JOB_STATUSES.includes(entry.status));
    const saved = busyButtonState.get(id);
    if (job) {
      /* Capture the resting label and disabled state once. Some of these buttons are
         disabled for reasons of their own, so restoring a blanket `false` would wrongly
         re-enable them. */
      if (!saved) busyButtonState.set(id, { label: button.textContent, disabled: button.disabled });
      const elapsed = jobElapsed(job);
      button.textContent = job.status === "queued"
        ? "Queued…"
        : `${verb}…${elapsed === null ? "" : ` ${formatClock(elapsed)}`}`;
      button.disabled = true;
    } else if (saved) {
      button.textContent = saved.label;
      button.disabled = saved.disabled;
      busyButtonState.delete(id);
    }
  });
}

/* The log pane is pinned below the job list and shows one job at a time. While that
   job is still running the poll refreshes it in place, so it doubles as a live tail. */
async function loadJobLog(jobId) {
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/log`);
  if (state.openJobLogId !== jobId) return;
  const log = el("jobLog");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = data.log || "No log output yet.";
  if (atBottom) log.scrollTop = log.scrollHeight;
}

async function openJobLog(job) {
  state.openJobLogId = job.job_id;
  state.openJobLogStatus = job.status;
  el("jobLogTitle").textContent = `Log · ${job.kind}`;
  el("jobLog").textContent = "Loading…";
  el("jobLogPane").hidden = false;
  setActivityOpen(true);
  renderJobs();
  try { await loadJobLog(job.job_id); }
  catch (error) {
    if (state.openJobLogId === job.job_id) el("jobLog").textContent = `Cannot load job log: ${error.message}`;
  }
}

function closeJobLog() {
  state.openJobLogId = null;
  state.openJobLogStatus = null;
  el("jobLogPane").hidden = true;
  el("jobLog").textContent = "";
  renderJobs();
}

el("jobLogClose").addEventListener("click", closeJobLog);

function jobMatchesFilter(job) {
  const filter = el("jobs-filter").value;
  if (filter === "active") return ACTIVE_JOB_STATUSES.includes(job.status);
  if (filter === "failed") return ["failed", "interrupted", "cancelled"].includes(job.status);
  if (filter === "completed") return job.status === "completed";
  return true;
}

const jobRowParts = new WeakMap();

/* Build a row once. Its buttons resolve the job from `state.jobs` at click time rather than
   closing over it, because the row now outlives the record it was first built from. */
function createJobRow(jobId) {
  const row = document.createElement("div");
  row.className = "job-row";
  row.dataset.jobId = jobId;

  const dot = document.createElement("span");
  dot.className = "job-state";

  const main = document.createElement("div");
  main.className = "job-main";
  const name = document.createElement("div");
  name.className = "job-name";
  const title = document.createElement("span");
  const clock = document.createElement("span");
  clock.className = "job-clock";
  name.append(title, clock);
  const progress = document.createElement("div");
  progress.className = "job-progress";
  const bar = makeBar("job-bar");
  main.append(name, progress, bar);

  const actions = document.createElement("div");
  actions.className = "job-actions";
  const logButton = document.createElement("button");
  logButton.type = "button";
  logButton.className = "btn btn-ghost";
  logButton.textContent = "Log";
  logButton.setAttribute("aria-controls", "jobLogPane");
  logButton.addEventListener("click", () => {
    if (state.openJobLogId === jobId) return closeJobLog();
    const job = state.jobs.find((entry) => entry.job_id === jobId);
    if (job) openJobLog(job);
  });
  const cancelButton = document.createElement("button");
  cancelButton.type = "button";
  cancelButton.className = "btn btn-ghost";
  cancelButton.textContent = "Cancel";
  cancelButton.addEventListener("click", async () => {
    cancelButton.disabled = true;
    try { await apiDelete(`/api/jobs/${encodeURIComponent(jobId)}`); await refreshJobs(); }
    catch (error) { status(`Cancellation failed: ${error.message}`); cancelButton.disabled = false; }
  });
  actions.append(logButton, cancelButton);

  row.append(dot, main, actions);
  jobRowParts.set(row, { dot, title, clock, progress, bar, logButton, cancelButton });
  return row;
}

function updateJobRow(row, job) {
  const { dot, title, clock, progress, bar, logButton, cancelButton } = jobRowParts.get(row);
  dot.className = `job-state ${job.status}`;
  title.textContent = `${job.kind} · ${job.status}`;
  const elapsed = jobElapsed(job);
  clock.textContent = elapsed === null ? "" : formatClock(elapsed);
  progress.textContent = [job.error || job.progress?.message || "", jobCounter(job)]
    .filter(Boolean)
    .join(" · ");
  progress.title = progress.textContent;
  setBar(bar, job);
  logButton.setAttribute("aria-expanded", String(state.openJobLogId === job.job_id));
  const active = ACTIVE_JOB_STATUSES.includes(job.status);
  cancelButton.hidden = !active;
  if (active && !job.cancel_requested) cancelButton.disabled = false;
}

/* Reconcile rather than rebuild: this runs once a second, and `innerHTML = ""` would restart
   every pulse and discard every bar transition before it could play. */
function renderJobs() {
  const list = el("jobList");
  renderActivitySummary();
  renderBusyButtons();

  const shown = state.jobs.filter(jobMatchesFilter).slice(0, 40);
  if (!shown.length) {
    list.innerHTML = `<div class="empty-note">${
      state.jobs.length ? "No job matches this filter." : "No workbench jobs yet."
    }</div>`;
    return;
  }

  const existing = new Map();
  list.querySelectorAll(".job-row").forEach((row) => existing.set(row.dataset.jobId, row));
  list.querySelector(".empty-note")?.remove();

  shown.forEach((job, index) => {
    let row = existing.get(job.job_id);
    if (row) existing.delete(job.job_id);
    else row = createJobRow(job.job_id);
    updateJobRow(row, job);
    if (list.children[index] !== row) list.insertBefore(row, list.children[index] || null);
  });
  existing.forEach((row) => row.remove());
}

async function handleFinishedJob(job) {
  if (state.handledJobs.has(job.job_id)) return;
  state.handledJobs.add(job.job_id);
  if (job.status !== "completed") {
    status(`${job.kind} ${job.status}: ${job.error || job.progress?.message || "no result"}`);
    return;
  }
  const result = job.result || {};
  if (job.kind === "fetch-osm") {
    await loadCities();
    // An extract fetched without its amenities still generates, just parametrically:
    // the fetcher says so and the user has to hear it now, not at generation time.
    const mode = result.dataset_mode && result.dataset_mode !== "roads_and_amenities"
      ? ` Dataset: ${result.dataset_mode.replaceAll("_", " ")}.`
      : "";
    status(`Fetched ${result.city || job.request?.city || "city"}.`
      + mode + (result.warning ? ` ${result.warning}` : ""));
  } else if (job.kind === "refresh-pois") {
    // The extracts changed on disk, so anything cached from them is stale.
    clearPoiPool();
    try {
      const report = await api("/api/workbench/osmdata/audit");
      state.osmAudit = report;
      renderOsmAudit(report);
    } catch { /* the status line below still reports the run itself */ }
    const failed = (result.results || []).filter((entry) => !entry.ok);
    status(`Updated ${result.refreshed || 0} extract(s), ${result.gained || 0} POI(s) gained.`
      + (failed.length
        ? ` ${failed.length} failed: ${failed.map((entry) => `${entry.city} (${entry.error || "incomplete"})`).join(", ")}.`
        : ""));
  } else if (job.kind === "generate") {
    state.lastRendered = null;
    resetRouteViewDefaults();
    clearRouteLayers();
    await loadInstances(result.base_name);
    activateTab("visualize");
    const composition = result.summary?.composition;
    const split = composition
      ? ` — ${composition.poi_customers} POI + ${composition.parametric_customers} parametric`
      : "";
    const outcome = {
      unchanged: `Already generated with identical content: ${result.base_name} (nothing written)`,
      renamed: `Generated as ${result.base_name}: another configuration already used this name and was kept`,
      replaced: `Replaced ${result.base_name}: its twins and BKS were deleted and its saved runs are now stale`,
    }[result.action] || `Generated ${result.base_name}`;
    status(`${outcome}: capacity ${result.summary?.capacity}, ~${result.summary?.route_count} routes${split}.`
      + (result.notice ? ` ${result.notice}` : "")
      + (result.vrptw_error ? ` VRPTW twin failed: ${result.vrptw_error}` : ""));
  } else if (job.kind === "bulk-generate") {
    const results = Array.isArray(result.results) ? result.results : [];
    state.bulkBases = results.map((entry) => entry.base_name).filter(Boolean);
    el("bulk-download").disabled = state.bulkBases.length === 0;
    await loadInstances(state.bulkBases[0]);
    activateTab("visualize");
    // Rows the pool could not serve are dropped by the driver, and a twin can
    // fail on its own: both would otherwise leave the count silently short.
    const problems = (result.city_reports || [])
      .filter((report) => report.status && report.status !== "ok")
      .map((report) => report.error
        ? `${report.city}: ${report.error}`
        : `${report.city}: pool holds ${report.pool_total ?? 0} (${report.poi_available ?? 0} POI`
          + ` + ${report.parametric_filled ?? 0} parametric), so n = `
          + `${(report.skipped_sizes || []).join(", ")} was skipped`);
    // row_reports is index-aligned with the instances payload, which is built
    // from state.bulkRows in order, so each row learns its own outcome.
    // Only when the table still holds the rows that were submitted: editing it
    // mid-run would otherwise pin outcomes onto the wrong rows.
    const rowReports = Array.isArray(result.row_reports) ? result.row_reports : [];
    if (rowReports.length && rowReports.length === state.bulkRows.length) {
      state.bulkRows.forEach((row, index) => { row.outcome = rowReports[index]; });
      renderBulkTable();
    }
    const twinFailures = results.filter((entry) => entry.vrptw_error).length;
    const requested = (job.request?.instances || []).length;
    status(`Bulk generated ${result.generated || 0}`
      + (requested ? ` of ${requested} requested` : "")
      + ` instance(s).`
      + (problems.length ? ` ${problems.join(" · ")}.` : "")
      + (twinFailures ? ` ${twinFailures} VRPTW twin(s) could not be derived.` : ""));
  } else if (job.kind === "solve") {
    const preferredRunId = result.solution?.run_id || null;
    let selectedByHandler = false;
    if (!state.selected || state.selected.instance_id !== result.instance_id) {
      await loadInstances();
      const index = state.instances.findIndex((instance) => instance.instance_id === result.instance_id);
      if (index >= 0) {
        await selectInstance(index, preferredRunId);
        selectedByHandler = true;
      }
    }
    if (!selectedByHandler) await loadSolutionRuns(preferredRunId);
    activateTab("visualize");
    status(`Solved ${result.solution?.instance_name || "instance"} with ${result.solver || "pyvrp"}: cost ${result.cost}; checker ${result.validation?.status}.`);
  }
}

let refreshingJobs = false;
async function refreshJobs() {
  if (refreshingJobs) return;
  refreshingJobs = true;
  try {
    const data = await api("/api/jobs?limit=50");
    state.jobs = data.jobs || [];
    renderJobs();
    if (state.openJobLogId) {
      const open = state.jobs.find((job) => job.job_id === state.openJobLogId);
      // Tail the log while the job is live; a finished job's log will not change again.
      // The status check also catches the transition to terminal, so the tail picks up
      // the lines written after the last active poll.
      if (!open) closeJobLog();
      else if (ACTIVE_JOB_STATUSES.includes(open.status) || open.status !== state.openJobLogStatus) {
        state.openJobLogStatus = open.status;
        await loadJobLog(open.job_id).catch(() => {});
      }
    }
    for (const job of state.jobs) {
      if (state.submittedJobs.has(job.job_id) && ["completed", "failed", "cancelled", "interrupted"].includes(job.status)) {
        await handleFinishedJob(job);
      }
    }
  } finally {
    refreshingJobs = false;
  }
}

/* ── Actions ── */

/* A modal question the caller can await; resolves false when dismissed. */
let confirmResolver = null;
function askConfirm(title, text, okLabel) {
  el("confirm-modal-title").textContent = title;
  el("confirm-text").textContent = text;
  el("confirm-ok").textContent = okLabel || "Generate anyway";
  el("confirm-modal").hidden = false;
  el("confirm-ok").focus();
  return new Promise((resolve) => { confirmResolver = resolve; });
}

function closeConfirm(answer) {
  el("confirm-modal").hidden = true;
  const resolve = confirmResolver;
  confirmResolver = null;
  if (resolve) resolve(answer);
}

el("confirm-ok").addEventListener("click", () => closeConfirm(true));
el("confirm-cancel").addEventListener("click", () => closeConfirm(false));
el("confirm-close").addEventListener("click", () => closeConfirm(false));
el("confirm-modal").addEventListener("click", (event) => {
  if (event.target === el("confirm-modal")) closeConfirm(false);
});

/* The composition line under the form: what the instance will actually hold,
   in green when the request is served as asked and amber when POIs ran out. */
function showGenNotice(notice, summary) {
  const box = el("gen-notice");
  const composition = summary && summary.composition;
  if (notice) {
    box.className = "preflight-line warn";
    box.textContent = notice;
    box.hidden = false;
  } else if (composition) {
    box.className = "preflight-line good";
    box.textContent = `${composition.delivered} customers = ${composition.poi_customers} POI`
      + ` + ${composition.parametric_customers} parametric.`;
    box.hidden = false;
  } else {
    box.hidden = true;
    box.textContent = "";
  }
}

el("preview").addEventListener("click", async () => {
  try {
    status("Previewing selection…");
    const data = await api("/api/workbench/generation/preview", requestBody());
    drawPreview(data.geojson);
    showGenNotice(data.notice, data.summary);
    status(`Preview: ${data.summary.customers} customers (${data.summary.poi_customers} POI · ${data.summary.parametric_customers} parametric).`
      + (data.notice ? ` ${data.notice}` : ""));
  } catch (error) { status(`Preview failed: ${error.message}`); }
});

el("generate").addEventListener("click", async () => {
  let body;
  try { body = requestBody(); }
  catch (error) { status(error.message); return; }
  try {
    // The same selection the job will build, so the split shown here is the one
    // that lands on disk; the road graph is cached, so the job does not repay it.
    status("Checking how many customers can sit on a real POI…");
    const report = await api("/api/workbench/generation/preflight", body);
    showGenNotice(report.notice, report.summary);
    if (report.notice) {
      const composition = report.summary?.composition || {};
      let title = "Not enough POIs for this request";
      if (composition.method === "manual") title = "Some picks cannot be used";
      else if (composition.poi_pool_matching === 0) title = "No POI in the selected categories";
      else if (composition.delivered < composition.requested) {
        title = "This city cannot serve that many customers";
      }
      const proceed = await askConfirm(title, report.notice);
      if (!proceed) {
        status("Generation cancelled — lower the customer count or select more POI categories.");
        return;
      }
    }
    const job = await submitJob("generate", body);
    status(`Generation queued as job ${job.job_id.slice(0, 8)}.`);
  } catch (error) { status(`Cannot queue generation: ${error.message}`); }
});

function syncVrpExportControls() {
  const euclidean = el("download-metric").value === "euclidean";
  el("download-vrp-euc2d-field").hidden = !euclidean;
  if (!euclidean) el("download-vrp-euc2d").checked = false;
}

el("download-metric").addEventListener("change", syncVrpExportControls);
el("solve-metric").addEventListener("change", () => loadSolvers().catch(() => {}));
el("solve-objective").addEventListener("change", () => loadSolvers().catch(() => {}));
el("solve-solver").addEventListener("change", syncSolverNote);
syncVrpExportControls();

el("download-vrp").addEventListener("click", async () => {
  const instance = state.selected;
  if (!instance) return;
  const metric = el("download-metric").value;
  const edgeWeightType = el("download-vrp-euc2d").checked ? "EUC_2D" : "EXPLICIT";
  const filename = `${instance.base_name}_${metric}.vrp`;
  try {
    status(`Exporting ${filename}…`);
    const blob = await apiBlob(`/api/workbench/instances/${encodeURIComponent(instance.instance_id)}/export-vrp`, {
      metric,
      edge_weight_type: edgeWeightType,
    });
    saveBlob(blob, filename);
    status(`Downloaded ${filename} (${edgeWeightType})`);
  } catch (error) { status(`Export failed: ${error.message}`); }
});

el("solve").addEventListener("click", async () => {
  const instance = state.selected;
  if (!instance) return;
  try {
    const budget = Math.max(1, Number(el("solve-budget").value) || 30);
    const solver = el("solve-solver").value || "pyvrp";
    const job = await submitJob("solve", {
      instance_id: instance.instance_id,
      metric: el("solve-metric").value,
      objective_function: el("solve-objective").value,
      seed: Number(el("solve-seed").value) || 0,
      time_limit: budget,
      solver,
    });
    status(`Solve queued as job ${job.job_id.slice(0, 8)} (${solver}, ${budget} s budget).`);
  } catch (error) { status(`Cannot queue solve: ${error.message}`); }
});

el("download").addEventListener("click", async () => {
  const instance = state.selected;
  if (!instance) return;
  try {
    status(`Zipping ${instance.base_name}…`);
    const blob = await apiBlob("/api/workbench/generation/single-download", { folder: instance.folder, base_name: instance.base_name });
    saveBlob(blob, `${instance.base_name}.zip`);
    status(`Downloaded ${instance.base_name}.zip`);
  } catch (error) { status(`Download failed: ${error.message}`); }
});

/* ── Stored extracts · POI coverage ── */
const OSM_STATUS_LABELS = {
  complete: ["ok", "up to date"],
  nodes_only: ["old", "points only"],
  no_amenities: ["old", "no POIs"],
  unreadable: ["bad", "unreadable"],
};

function renderOsmAudit(report) {
  const list = el("osm-audit-list");
  list.innerHTML = "";
  const extracts = report.extracts || [];
  if (!extracts.length) {
    list.innerHTML = '<div class="empty-note">No .osm extract in the workspace yet.</div>';
    el("osm-audit-count").textContent = "";
    el("osm-audit-refresh").disabled = true;
    return;
  }
  extracts.forEach((entry) => {
    const [chipClass, label] = OSM_STATUS_LABELS[entry.status] || ["bad", entry.status || "?"];
    const row = document.createElement("div");
    row.className = "osm-row";
    const name = document.createElement("span");
    name.className = "nm";
    name.textContent = entry.city;
    const count = document.createElement("span");
    count.className = "cnt";
    // Ways and relations are the half that older extracts are missing entirely.
    count.textContent = entry.error
      ? entry.error
      : `${entry.poi_total} POI (${entry.poi_nodes}n / ${entry.poi_ways}w / ${entry.poi_relations}r)`;
    const chip = document.createElement("span");
    chip.className = `chip ${chipClass}`;
    chip.textContent = label;
    row.append(name, count, chip);
    list.append(row);
  });
  const outdated = report.outdated || 0;
  el("osm-audit-count").textContent = outdated
    ? `${outdated} of ${extracts.length} outdated`
    : `${extracts.length} up to date`;
  el("osm-audit-refresh").disabled = outdated === 0;
}

el("osm-audit-check").addEventListener("click", async () => {
  try {
    status("Checking the POI coverage of the stored extracts…");
    const report = await api("/api/workbench/osmdata/audit");
    state.osmAudit = report;
    renderOsmAudit(report);
    status(report.outdated
      ? `${report.outdated} extract(s) hold only point-mapped POIs; updating adds the ones mapped as building outlines.`
      : `All ${(report.extracts || []).length} extract(s) already hold way-mapped POIs.`);
  } catch (error) { status(`Cannot check the extracts: ${error.message}`); }
});

el("osm-audit-refresh").addEventListener("click", async () => {
  const outdated = (state.osmAudit?.extracts || [])
    .filter((entry) => entry.can_refresh && entry.status !== "complete")
    .map((entry) => entry.city);
  if (!outdated.length) return;
  try {
    const job = await submitJob("refresh-pois", { cities: outdated });
    status(`Updating ${outdated.length} extract(s) as job ${job.job_id.slice(0, 8)}: ${outdated.join(", ")}.`);
  } catch (error) { status(`Cannot queue the update: ${error.message}`); }
});

el("fetch").addEventListener("click", async () => {
  const name = el("fetch-name").value.trim();
  if (!name) return;
  try {
    const job = await submitJob("fetch-osm", {
      city: name,
      poiCategories: [...POI_CATEGORIES],
    });
    status(`OSM fetch for ${name} queued with all ${POI_CATEGORIES.length} POI categories as job ${job.job_id.slice(0, 8)}.`);
  } catch (error) { status(`Cannot queue fetch: ${error.message}`); }
});

/* ── Bulk configuration modal ── */
function bulkBaseRow() {
  // Everything a row inherits from the Generate form when it is created.
  const categories = selectedPoiCategories();
  return {
    problemType: el("bulk-problem")?.value || "cvrp",
    city: el("city").value,
    nCustomers: Number(el("n").value) || 10,
    demandType: Number(el("demand").value) || 7,
    avgRouteSize: Number(el("ars").value) || 4,
    method: el("method").value === "manual" ? "poi_categories" : el("method").value,
    // null = let the generator derive a seed per (city, n, demand type, band),
    // which is what makes instances of one batch differ. Copying the Generate
    // form's seed into every row would pin them all to the same value.
    seed: null,
    depotMode: el("depot-mode").value,
    customerMode: el("customer-mode").value,
    twMethod: el("bulk-tw")?.value || "route_centered",
    onlyIntersections: true,
    clusterSeeds: Number(el("cluster-seeds").value) || 4,
    clusterDecayMeters: Number(el("cluster-decay").value) || 800,
    hybridPoiShare: Number(el("hybrid-poi-share").value) / 100,
    categories: categories.join(" "),
  };
}

function bulkRowKey(row) {
  return [row.problemType, row.city, row.nCustomers, row.demandType, row.avgRouteSize,
    row.method, row.seed, row.categories].join("|");
}

function refreshBulkCounts() {
  const label = `${state.bulkRows.length} instance${state.bulkRows.length === 1 ? "" : "s"}`;
  el("bulk-count-badge").textContent = label;
  el("bulk-modal-count").textContent = label;
}

function bulkCell(row, field, kind, options) {
  const cell = document.createElement("td");
  let input;
  if (kind === "select") {
    input = document.createElement("select");
    options.forEach(([value, text]) => input.append(new Option(text, value)));
    input.value = String(row[field]);
  } else if (kind === "checkbox") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(row[field]);
  } else {
    input = document.createElement("input");
    input.type = kind;
    input.value = String(row[field] ?? "");
    if (options?.placeholder) input.placeholder = options.placeholder;
    if (kind === "number") { input.min = options?.min ?? 0; if (options?.step) input.step = options.step; }
  }
  input.addEventListener("change", () => {
    if (kind === "checkbox") row[field] = input.checked;
    // An emptied optional number means "unset", not 0: the seed column relies
    // on null to hand the choice back to the generator.
    else if (options?.nullable && input.value.trim() === "") row[field] = null;
    else if (BULK_INT_FIELDS.has(field)) row[field] = Number.parseInt(input.value, 10) || 0;
    else if (BULK_FLOAT_FIELDS.has(field)) row[field] = Number.parseFloat(input.value) || 0;
    else row[field] = input.value;
  });
  cell.append(input);
  return cell;
}

/* What the last bulk run did with this row: rows the city pool could not serve
   are dropped by the driver, so without this the table looks like it all ran. */
function bulkOutcomeCell(row) {
  const cell = document.createElement("td");
  const outcome = row.outcome;
  if (!outcome) {
    cell.className = "preflight-line";
    cell.textContent = "—";
    return cell;
  }
  if (outcome.status === "skipped") {
    cell.className = "preflight-line bad";
    cell.textContent = "skipped";
    cell.title = outcome.reason || "Skipped by the last run";
    return cell;
  }
  const parametric = Number(outcome.parametric_customers) || 0;
  cell.className = parametric || outcome.vrptw_error ? "preflight-line warn" : "preflight-line good";
  cell.textContent = outcome.vrptw_error
    ? "no twin"
    : (parametric ? `${outcome.poi_customers} POI + ${parametric} param` : "generated");
  cell.title = [outcome.base_name, outcome.notice, outcome.vrptw_error]
    .filter(Boolean).join(" · ") || "Generated";
  return cell;
}

function renderBulkTable() {
  const body = el("bulk-table-body");
  body.innerHTML = "";
  if (!state.bulkRows.length) {
    body.innerHTML = '<tr><td colspan="18" class="bulk-empty empty-note">'
      + "No rows yet. Build combinations on the left, add rows by hand, or import a CSV.</td></tr>";
    refreshBulkCounts();
    return;
  }
  const cityOptions = [...el("city").options].map((option) => [option.value, option.textContent]);
  state.bulkRows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.dataset.id = String(row.id);

    const check = document.createElement("td");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.className = "bulk-row-check";
    box.setAttribute("aria-label", "Select this row");
    check.append(box);
    tr.append(check);
    // Kept at the front: a skipped row must be visible without scrolling the
    // table sideways past a dozen parameter columns.
    tr.append(bulkOutcomeCell(row));

    tr.append(bulkCell(row, "problemType", "select", [["cvrp", "CVRP"], ["vrptw", "VRPTW"]]));
    tr.append(bulkCell(row, "city", "select", cityOptions.length ? cityOptions : [[row.city, row.city]]));
    tr.append(bulkCell(row, "nCustomers", "number", { min: 2 }));
    tr.append(bulkCell(row, "demandType", "select", DEMAND_TYPES.map((e) => [String(e.value), `${e.value} · ${e.label}`])));
    tr.append(bulkCell(row, "avgRouteSize", "select", ROUTE_SIZE_BANDS.map((e) => [String(e.value), `${e.value} · ${e.label}`])));
    tr.append(bulkCell(row, "method", "select", SAMPLING_METHODS.filter((m) => m !== "manual").map((m) => [m, m])));
    tr.append(bulkCell(row, "seed", "number", { min: 0, nullable: true, placeholder: "auto" }));
    tr.append(bulkCell(row, "depotMode", "select", DEPOT_MODES.map((m) => [m, m])));
    tr.append(bulkCell(row, "customerMode", "select", CUSTOMER_MODES.map((m) => [m, m])));
    tr.append(bulkCell(row, "twMethod", "select", TW_METHODS.map((m) => [m, m])));
    tr.append(bulkCell(row, "onlyIntersections", "checkbox"));
    tr.append(bulkCell(row, "clusterSeeds", "number", { min: 1 }));
    tr.append(bulkCell(row, "clusterDecayMeters", "number", { min: 1, step: 50 }));
    tr.append(bulkCell(row, "hybridPoiShare", "number", { min: 0, step: 0.05 }));
    const categoryCell = bulkCell(row, "categories", "text");
    categoryCell.className = "cat-cell";
    tr.append(categoryCell);

    const remove = document.createElement("td");
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "bulk-delete-btn";
    removeButton.textContent = "×";
    removeButton.title = "Remove this row";
    removeButton.addEventListener("click", () => {
      state.bulkRows = state.bulkRows.filter((entry) => entry.id !== row.id);
      renderBulkTable();
    });
    remove.append(removeButton);
    tr.append(remove);
    body.append(tr);
  });
  el("bulk-select-all").checked = false;
  refreshBulkCounts();
}

function addBulkRow(overrides = {}) {
  state.bulkRows.push({ id: state.bulkNextId, ...bulkBaseRow(), ...overrides });
  state.bulkNextId += 1;
}

function checkedValues(containerId) {
  return [...el(containerId).querySelectorAll("input[type='checkbox']:checked")]
    .map((box) => Number.parseInt(box.value, 10))
    .filter(Number.isFinite);
}

function expandBulkCombinations() {
  const cities = [...el("bulk-cities").querySelectorAll("input[type='checkbox']:checked")]
    .map((box) => box.value);
  const sizes = el("bulk-sizes").value.split(",")
    .map((part) => Number.parseInt(part.trim(), 10))
    .filter((value) => Number.isFinite(value) && value >= 2);
  const demands = checkedValues("bulk-demand-checks");
  const bands = checkedValues("bulk-band-checks");
  if (!cities.length || !sizes.length || !demands.length || !bands.length) {
    status("Select at least one city, size, demand type and route size band.");
    return;
  }
  const existing = new Set(state.bulkRows.map(bulkRowKey));
  let added = 0;
  let duplicates = 0;
  cities.forEach((city) => sizes.forEach((nCustomers) => demands.forEach((demandType) => bands.forEach((avgRouteSize) => {
    const candidate = { ...bulkBaseRow(), city, nCustomers, demandType, avgRouteSize };
    const key = bulkRowKey(candidate);
    if (existing.has(key)) { duplicates += 1; return; }
    existing.add(key);
    addBulkRow({ city, nCustomers, demandType, avgRouteSize });
    added += 1;
  }))));
  renderBulkTable();
  status(`Added ${added} combination(s)${duplicates ? `, skipped ${duplicates} duplicate(s)` : ""}.`);
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (quoted) {
      if (char === '"' && line[index + 1] === '"') { current += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else current += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { values.push(current.trim()); current = ""; }
    else current += char;
  }
  values.push(current.trim());
  return values;
}

function exportBulkCsv() {
  if (!state.bulkRows.length) { status("No rows to export."); return; }
  const lines = [BULK_CSV_COLUMNS.join(",")];
  state.bulkRows.forEach((row) => {
    lines.push(BULK_CSV_COLUMNS.map((column) => {
      const value = column === "onlyIntersections" ? (row[column] ? "true" : "false") : row[column];
      return csvEscape(value);
    }).join(","));
  });
  saveBlob(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }), "bulk_instances.csv");
  status(`Exported ${state.bulkRows.length} row(s) to bulk_instances.csv`);
}

function importBulkCsv(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const lines = String(reader.result || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (lines.length < 2) { status("The CSV needs a header row and at least one data row."); return; }
    const header = parseCsvLine(lines[0]).map((value) => value.trim());
    const lowered = header.map((value) => value.toLowerCase());
    const required = ["city", "ncustomers", "demandtype", "avgroutesize"];
    const missing = required.filter((column) => !lowered.includes(column));
    if (missing.length) { status(`The CSV is missing required column(s): ${missing.join(", ")}`); return; }
    let imported = 0;
    lines.slice(1).forEach((line) => {
      const values = parseCsvLine(line);
      const overrides = {};
      header.forEach((column, index) => {
        const canonical = BULK_CSV_COLUMNS.find((name) => name.toLowerCase() === column.toLowerCase());
        if (!canonical) return;
        const raw = values[index];
        if (raw === undefined || raw === "") return;
        if (canonical === "onlyIntersections") overrides[canonical] = /^(true|1|yes)$/i.test(raw);
        else if (BULK_INT_FIELDS.has(canonical)) overrides[canonical] = Number.parseInt(raw, 10) || 0;
        else if (BULK_FLOAT_FIELDS.has(canonical)) overrides[canonical] = Number.parseFloat(raw) || 0;
        else overrides[canonical] = raw;
      });
      addBulkRow(overrides);
      imported += 1;
    });
    renderBulkTable();
    status(`Imported ${imported} row(s) from ${file.name}.`);
  };
  reader.readAsText(file);
}

function bulkPayload() {
  if (!state.bulkRows.length) throw new Error("Add at least one instance row.");
  return {
    seed: Number(el("seed").value) || 0,
    // Shared by every row: the server merges top-level fields into each one, so
    // this needs no column of its own in an already wide table.
    poiAttachMode: el("poi-attach-mode").value,
    poiAttachRadiusM: Number(el("poi-attach-radius").value) || 50,
    overwrite: el("overwrite").checked,
    instances: state.bulkRows.map((row) => ({
      problemType: row.problemType,
      city: row.city,
      nCustomers: row.nCustomers,
      demandType: row.demandType,
      avgRouteSize: row.avgRouteSize,
      method: row.method,
      seed: row.seed,
      depotMode: row.depotMode,
      customerMode: row.customerMode,
      twMethod: row.twMethod,
      onlyIntersections: row.onlyIntersections,
      clusterSeeds: row.clusterSeeds,
      clusterDecayMeters: row.clusterDecayMeters,
      hybridPoiShare: row.hybridPoiShare,
      categories: String(row.categories || "").split(/[\s,]+/).filter(Boolean),
    })),
  };
}

function renderBulkCities() {
  const container = el("bulk-cities");
  container.innerHTML = "";
  [...el("city").options].forEach((option) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = option.value;
    box.checked = option.value === el("city").value;
    box.addEventListener("change", updateBulkCityCount);
    const text = document.createElement("span");
    text.textContent = option.textContent;
    label.append(box, text);
    container.append(label);
  });
  updateBulkCityCount();
}

function updateBulkCityCount() {
  const selected = el("bulk-cities").querySelectorAll("input:checked").length;
  el("bulk-city-count").textContent = `${selected} selected`;
}

function renderBulkChecks() {
  const fill = (containerId, entries, defaults) => {
    const container = el(containerId);
    container.innerHTML = "";
    entries.forEach((entry) => {
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = String(entry.value);
      box.checked = defaults.includes(entry.value);
      const text = document.createElement("span");
      text.textContent = `${entry.value} · ${entry.label}`;
      const band = document.createElement("span");
      band.className = "band";
      band.textContent = entry.detail;
      label.append(box, text, band);
      container.append(label);
    });
  };
  fill("bulk-demand-checks", DEMAND_TYPES, [7]);
  fill("bulk-band-checks", ROUTE_SIZE_BANDS, [4]);
}

function openBulkModal() {
  renderBulkCities();
  renderBulkTable();
  el("bulk-modal").hidden = false;
}

function closeBulkModal() {
  el("bulk-modal").hidden = true;
}

el("bulk-open").addEventListener("click", openBulkModal);
el("bulk-close").addEventListener("click", closeBulkModal);
el("bulk-close-2").addEventListener("click", closeBulkModal);
el("bulk-modal").addEventListener("click", (event) => {
  if (event.target === el("bulk-modal")) closeBulkModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  // The confirm dialog sits on top of the bulk modal, so it closes first.
  if (!el("confirm-modal").hidden) closeConfirm(false);
  else if (!el("bulk-modal").hidden) closeBulkModal();
});
el("bulk-city-all").addEventListener("click", () => {
  el("bulk-cities").querySelectorAll("input").forEach((box) => { box.checked = true; });
  updateBulkCityCount();
});
el("bulk-city-none").addEventListener("click", () => {
  el("bulk-cities").querySelectorAll("input").forEach((box) => { box.checked = false; });
  updateBulkCityCount();
});
el("bulk-expand").addEventListener("click", expandBulkCombinations);
el("bulk-add-row").addEventListener("click", () => { addBulkRow(); renderBulkTable(); });
el("bulk-clear").addEventListener("click", () => { state.bulkRows = []; renderBulkTable(); });
el("bulk-delete-selected").addEventListener("click", () => {
  const selected = new Set();
  el("bulk-table-body").querySelectorAll(".bulk-row-check:checked").forEach((box) => {
    const id = Number.parseInt(box.closest("tr")?.dataset.id || "", 10);
    if (Number.isFinite(id)) selected.add(id);
  });
  if (!selected.size) { status("No rows selected."); return; }
  state.bulkRows = state.bulkRows.filter((row) => !selected.has(row.id));
  renderBulkTable();
});
el("bulk-select-all").addEventListener("change", (event) => {
  el("bulk-table-body").querySelectorAll(".bulk-row-check").forEach((box) => {
    box.checked = event.target.checked;
  });
});
el("bulk-export-csv").addEventListener("click", exportBulkCsv);
el("bulk-import-csv").addEventListener("click", () => el("bulk-csv-file").click());
el("bulk-csv-file").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) importBulkCsv(file);
  event.target.value = "";
});

el("bulk-check").addEventListener("click", async () => {
  const line = el("bulk-preflight");
  try {
    line.className = "preflight-line";
    line.textContent = "Checking how many customers each city pool can offer…";
    const report = await api("/api/workbench/generation/bulk-preflight", bulkPayload());
    const bad = report.groups.filter((group) => group.skipped_sizes?.length || group.status === "skipped");
    // A pool that fits can still be short of POIs: rows bigger than the POI
    // count take parametric road points, which is worth saying up front.
    const thin = report.groups.filter((group) => !bad.includes(group)
      && group.method !== "parametric_attach" && group.method !== "manual"
      && (group.sizes_needing_parametric || []).length);
    if (!bad.length && !thin.length) {
      line.className = "preflight-line good";
      line.textContent = `All ${report.instances} instance(s) fit their city pool.`;
      return;
    }
    line.className = "preflight-line warn";
    line.textContent = [
      ...bad.map((group) => `${group.city}: pool holds ${group.pool_total ?? 0}, `
        + `so n = ${(group.skipped_sizes || []).join(", ")} would be skipped`),
      ...thin.map((group) => `${group.city}: only ${group.poi_available ?? 0} POI(s) attach to the `
        + `road graph, so n = ${(group.sizes_needing_parametric || []).join(", ")} will be `
        + "completed with parametric points"),
    ].join(" · ");
  } catch (error) {
    line.className = "preflight-line bad";
    line.textContent = `Feasibility check failed: ${error.message}`;
  }
});

el("bulk-run").addEventListener("click", async () => {
  try {
    const payload = bulkPayload();
    state.bulkRows.forEach((row) => { row.outcome = null; });
    renderBulkTable();
    const job = await submitJob("bulk-generate", payload);
    status(`Bulk generation of ${state.bulkRows.length} instance(s) queued as job ${job.job_id.slice(0, 8)}.`);
    closeBulkModal();
  } catch (error) { status(`Cannot queue bulk generation: ${error.message}`); }
});

el("bulk-download").addEventListener("click", async () => {
  if (!state.bulkBases?.length) return;
  try {
    status("Zipping the bulk batch…");
    const blob = await apiBlob("/api/workbench/generation/bulk-download", { base_names: state.bulkBases });
    saveBlob(blob, "mamut-generated-instances.zip");
    status("Downloaded mamut-generated-instances.zip");
  } catch (error) { status(`Download failed: ${error.message}`); }
});

for (const button of document.querySelectorAll("[data-animate-routes]")) {
  button.addEventListener("click", () => {
    void toggleRouteAnimation();
  });
}

el("clear").addEventListener("click", () => {
  previewLayers.clearLayers();
  poiPickLayers.clearLayers();
  solutionData = { key: null, routeLines: [], nodes: [], depot: null, routeMode: "straight_line", hasRoutes: false };
  pushSolution();
  state.previewGeojson = null;
  state.lastRendered = null;
  if (state.selected) state.selected.rendered = null;
  renderLegend();
  syncAnimateButton();  // nothing is drawn any more: no Animate (pushSolution ran before lastRendered was cleared)
  status("Map cleared.");
});

/* ── Import an externally found solution ── */
function resetImportPanel() {
  el("import-objective").value = "";
  el("import-text").value = "";
  el("import-file").value = "";
  el("import-label").value = "";
  el("import-validation").className = "validation-line";
  el("import-validation").textContent =
    "Imported solutions are checked against the chosen metric before they are stored.";
}

function openImportPanel() {
  resetImportPanel();
  el("import-metric").value = preferredMetric();
  el("import-panel").hidden = false;
  el("import-panel").scrollIntoView({ block: "nearest" });
}

el("import-open").addEventListener("click", () => {
  if (el("import-panel").hidden) openImportPanel();
  else el("import-panel").hidden = true;
});
el("import-cancel").addEventListener("click", () => { el("import-panel").hidden = true; });
el("import-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  el("import-text").value = await file.text();
  el("import-validation").className = "validation-line";
  el("import-validation").textContent = `Loaded ${file.name}; press "Check and import" to validate it.`;
});

el("import-run").addEventListener("click", async () => {
  const instance = state.selected;
  const line = el("import-validation");
  if (!instance) { status("Select an instance first."); return; }
  const metric = el("import-metric").value;
  try {
    line.className = "validation-line";
    line.textContent = `Checking the routes against the ${metric} variant…`;
    const result = await api(
      `/api/instances/${encodeURIComponent(instance.instance_id)}/solutions/import`,
      {
        metric,
        objective_function: el("import-objective").value || undefined,
        text: el("import-text").value,
        filename: el("import-file").files?.[0]?.name || "",
        label: el("import-label").value,
      },
    );
    line.className = "validation-line valid";
    line.textContent = result.warning
      ? `Imported and validated. ${result.warning}`
      : `Imported and validated: cost ${result.solution.cost}, ${result.solution.num_routes} route(s).`;
    el("import-panel").hidden = true;
    await loadSolutionRuns(result.solution.run_id);
    status(`Imported an external solution for ${instance.base_name} (${metric}).`);
  } catch (error) {
    // Nothing was stored: the server refuses before writing.
    line.className = "validation-line invalid";
    line.textContent = `Rejected: ${error.message}`;
  }
});

/* ── Narrow viewports: the right panel becomes a collapsible sheet ── */
el("sheet-toggle").addEventListener("click", () => {
  const panel = document.querySelector(".panel-right");
  const collapsed = panel.classList.toggle("sheet-collapsed");
  el("sheet-toggle").textContent = collapsed ? "Show" : "Hide";
  el("sheet-toggle").setAttribute("aria-expanded", collapsed ? "false" : "true");
});

fillBandSelects();
renderBulkChecks();
renderBulkTable();
renderPoiCategories();
updateHybridShare();
updateGenerationControls();
loadCities().catch((error) => status(`Cannot reach the local server: ${error.message}`));
loadSolvers().catch((error) => console.warn("Unable to list solvers", error));
loadInstances().catch((error) => console.warn("Unable to list workspace instances", error));
refreshJobs().catch((error) => console.warn("Unable to list workbench jobs", error));
setInterval(() => refreshJobs().catch(() => {}), 1000);
