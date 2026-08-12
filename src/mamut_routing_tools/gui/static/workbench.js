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
const map = L.map("map", { zoomControl: true }).setView([46.5, 2.5], 6);
const positron = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
  maxZoom: 20, subdomains: "abcd", attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
});
const darkMatter = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  maxZoom: 20, subdomains: "abcd", attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
});
const openStreetMap = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
});
let baseLayer = isDark() ? darkMatter : positron;
baseLayer.addTo(map);

L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);
L.control.layers(
  { "Dark Matter": darkMatter, "Positron": positron, "OpenStreetMap": openStreetMap },
  {},
  { position: "topright" },
).addTo(map);

/* The basemap normally follows the theme. Once the user picks one explicitly that
   link is cut, otherwise flipping the theme would silently undo their choice. */
let basemapPinned = false;
map.on("baselayerchange", (event) => {
  baseLayer = event.layer;
  basemapPinned = true;
});

const previewLayers = L.layerGroup().addTo(map);
const instanceNodeLayers = L.layerGroup().addTo(map);
const poiPickLayers = L.layerGroup().addTo(map);
let routeLayerGroups = [];

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
  const nextBase = isDark() ? darkMatter : positron;
  if (!basemapPinned && nextBase !== baseLayer) {
    map.removeLayer(baseLayer);
    nextBase.addTo(map);
    baseLayer = nextBase;
  }
  redrawRoutes();
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
  hiddenRoutes: new Set(),
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
  /* Leaflet caches the container size; without this the map keeps the old width and
     the tiles tear or grey out after a drag. */
  onResize: () => map.invalidateSize({ animate: false }),
  onPersist: (state) => savePreferences({ layout: state }),
});

/* The markup ships with Generate active; run it once so the rail highlight agrees. */
activateTab(el("tabGenerate").classList.contains("tab-active") ? "generate" : "visualize");

/* ── Inspector tabs ── */
const INSPECTOR_TABS = { instance: "instancePanel", solve: "solvePanel", runs: "runsPanel" };
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
activateInspectorTab("instance");

/* ── Instance filters ── */
["inst-search", "inst-city", "inst-sort", "inst-solved"].forEach((id) => {
  el(id).addEventListener("input", renderInstanceList);
});

/* Copy the full workspace path out of the facts list. */
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
function placePopover(popover, anchor) {
  popover.hidden = false;
  const box = anchor.getBoundingClientRect();
  const size = popover.getBoundingClientRect();
  const margin = 10;
  let left = box.right + margin;
  if (left + size.width > window.innerWidth - margin) {
    left = Math.max(margin, box.left - size.width - margin);
  }
  let top = box.top;
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
function showPopover(popover, anchor, onClose) {
  closePopover();
  placePopover(popover, anchor);
  openPopover = { popover, onClose: onClose || (() => { popover.hidden = true; }) };
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closePopover();
});
document.addEventListener("pointerdown", (event) => {
  if (!openPopover) return;
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (openPopover.popover.contains(target) || target.closest(".help-btn, #poi-edit")) return;
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
  } else if (state.selected) {
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

/* ── Map drawing ── */
function clearRouteLayers() {
  routeLayerGroups.forEach((group) => map.removeLayer(group));
  routeLayerGroups = [];
}

function drawPreview(geojson) {
  state.previewGeojson = geojson;
  clearRouteLayers();
  instanceNodeLayers.clearLayers();
  previewLayers.clearLayers();
  const colors = markerColors();
  const bounds = [];
  geojson.features.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates;
    bounds.push([lat, lon]);
    const tag = String(feature.properties.source_tag || "");
    const role = feature.properties.role === "depot"
      ? "depot"
      : (tag.startsWith("poi") ? "poi" : "parametric");
    L.circleMarker([lat, lon], {
      radius: role === "depot" ? 7 : 4,
      color: colors[role],
      fillColor: colors[role],
      fillOpacity: 0.85,
      weight: role === "depot" ? 2 : 1,
    }).bindTooltip(nodeTooltip(feature.properties)).addTo(previewLayers);
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [60, 60] });
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

function renderPoiMarkers(fit = false) {
  poiPickLayers.clearLayers();
  const colors = markerColors();
  const bounds = [];
  state.poiPool.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates;
    const key = poiKey(feature.properties);
    const picked = state.manualPicks.has(key);
    const isDepot = state.manualDepotKey === key;
    bounds.push([lat, lon]);
    const marker = L.circleMarker([lat, lon], {
      radius: isDepot ? 7 : picked ? 5.5 : 3.5,
      color: isDepot ? colors.depot : picked ? colors.poi : colors.parametric,
      fillColor: isDepot ? colors.depot : picked ? colors.poi : colors.parametric,
      fillOpacity: picked || isDepot ? 0.9 : 0.35,
      weight: picked || isDepot ? 2 : 1,
    });
    const suffix = isDepot ? " · DEPOT" : picked ? " · picked" : "";
    marker.bindTooltip(`${poiLabel(feature.properties)}${suffix}`);
    marker.on("click", (event) => {
      // Shift-click promotes a POI to the depot instead of toggling it.
      if (event.originalEvent && event.originalEvent.shiftKey) setManualDepot(key, feature);
      else toggleManualPick(key, feature);
    });
    marker.addTo(poiPickLayers);
  });
  if (fit && bounds.length) map.fitBounds(bounds, { padding: [50, 50] });
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

function drawInstanceNodes(routes = null, fit = false) {
  instanceNodeLayers.clearLayers();
  const mapData = state.selected?.mapData;
  if (!mapData?.geojson?.features) return [];
  const assignments = new Map();
  (routes || []).forEach((route, routeIndex) => {
    route.forEach((modelNodeId) => assignments.set(Number(modelNodeId), routeIndex));
  });
  const colors = markerColors();
  const bounds = [];
  mapData.geojson.features.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates;
    const properties = feature.properties || {};
    const isDepot = properties.role === "depot";
    const routeIndex = assignments.get(Number(properties.model_node_id));
    const color = isDepot
      ? colors.depot
      : (routeIndex == null
        ? (String(properties.source_tag || "").startsWith("poi") ? colors.poi : colors.parametric)
        : routeColor(routeIndex));
    const marker = L.circleMarker([lat, lon], {
      radius: isDepot ? 7 : 5,
      color,
      fillColor: color,
      fillOpacity: 0.9,
      weight: isDepot ? 3 : 2,
    });
    const label = isDepot ? "Depot" : `Customer ${properties.model_node_id}`;
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
    marker.bindTooltip(
      `${label}${named} · demand ${properties.demand} · ${properties.source_tag}`
      + `${snapped}${alsoHere}`);
    marker.addTo(instanceNodeLayers);
    bounds.push([lat, lon]);
  });
  if (fit && bounds.length) map.fitBounds(bounds, { padding: [60, 60] });
  return bounds;
}

function drawRoutes(rendered, fit) {
  clearRouteLayers();
  previewLayers.clearLayers();
  const bounds = [];
  rendered.geojson.features.forEach((feature, index) => {
    const group = L.layerGroup();
    const line = feature.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    bounds.push(...line);
    L.polyline(line, { color: routeColor(index), weight: 3, opacity: 0.9 })
      .bindTooltip(`R${feature.properties.route_index}: ${feature.properties.stops} stops, load ${feature.properties.load} (${feature.properties.render_mode})`)
      .addTo(group);
    routeLayerGroups.push(group);
    if (!state.hiddenRoutes.has(index)) group.addTo(map);
  });
  bounds.push(...drawInstanceNodes(state.selected?.solved?.routes || [], false));
  if (fit && bounds.length) map.fitBounds(bounds, { padding: [60, 60] });
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
      `<span class="inst-meta">${instance.city} · n=${instance.n} · seed ${instance.seed} · capacity ${instance.summary.capacity}</span>`;
    row.addEventListener("click", () => selectInstance(index));
    list.append(row);
  });
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

function renderSelected() {
  const instance = state.selected;
  el("selEmpty").hidden = Boolean(instance);
  el("selBody").hidden = !instance;
  if (!instance) return;
  el("selName").textContent = instance.base_name;
  el("selMeta").textContent = `CVRP · ${instance.city} · n=${instance.n}`;
  const facts = [
    ["Capacity", instance.summary.capacity],
    ["Expected routes", instance.summary.route_count != null ? `~${instance.summary.route_count}` : "n/a"],
    ["Seed", instance.seed],
    ["Folder", folderFactHtml(instance.folder)],
    ["Saved solutions", instance.solution_count],
  ];
  if (instance.vrptw) facts.push(["VRPTW twin", "derived"]);
  if (instance.solved) {
    // Which of the three metric variants this cost actually belongs to.
    facts.push(["Solved on", solvedVariantFile(instance)]);
  }
  el("selFacts").innerHTML = facts
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
  if (instance.solved) {
    const metric = runMetric(instance.solved);
    el("selObjective").hidden = false;
    el("selObjLabel").textContent =
      `${metric} · ${instance.solved.objective_function || "MonoCost"} · ${runSourceLabel(instance.solved)}`;
    el("selCost").textContent = String(instance.solved.cost);
    el("selMethod").textContent = instance.solved.method || "pyvrp";
  } else {
    el("selObjective").hidden = true;
  }
  renderLegend();
}

function renderLegend() {
  const legend = el("legend");
  const instance = state.selected;
  if (!instance || !instance.solved || !state.lastRendered) {
    legend.innerHTML = '<div class="empty-note">Customer positions are displayed on the map. Select or create a saved solution to draw routes.</div>';
    return;
  }
  legend.innerHTML = "";
  state.lastRendered.geojson.features.forEach((feature, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "legend-row" + (state.hiddenRoutes.has(index) ? " off" : "");
    row.innerHTML = `<span class="dot" style="background:${routeColor(index)}"></span>` +
      `<span class="lbl">R${feature.properties.route_index}</span>` +
      `<span class="meta">${feature.properties.stops} stops · load ${feature.properties.load}</span>`;
    row.addEventListener("click", () => {
      if (state.hiddenRoutes.has(index)) {
        state.hiddenRoutes.delete(index);
        routeLayerGroups[index]?.addTo(map);
      } else {
        state.hiddenRoutes.add(index);
        if (routeLayerGroups[index]) map.removeLayer(routeLayerGroups[index]);
      }
      row.classList.toggle("off", state.hiddenRoutes.has(index));
    });
    legend.append(row);
  });
}

function selectInstance(index, preferredRunId = null) {
  state.selected = state.instances[index];
  state.selected.solved = null;
  state.selected.rendered = null;
  state.solutionRuns = [];
  state.solutionReferences = [];
  state.selectedRunId = null;
  state.previewGeojson = null;
  state.hiddenRoutes = new Set();
  state.lastRendered = null;
  previewLayers.clearLayers();
  instanceNodeLayers.clearLayers();
  clearRouteLayers();
  // A half-filled import belongs to the instance it was opened for.
  el("import-panel").hidden = true;
  renderInstanceList();
  renderSelected();
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
  state.hiddenRoutes = new Set();
  state.previewGeojson = null;
  previewLayers.clearLayers();
  clearRouteLayers();
  drawInstanceNodes(null, fit);
  refreshSolutionControls();
  renderSelected();
  status(`Displayed ${instance.base_name}: depot and ${instance.n} customer positions.`);
}

/* ── Persistent solutions ── */
function allSolutionRecords() {
  return [...state.solutionRuns, ...state.solutionReferences];
}

/* The metric a run was produced on. Every instance ships shortest/fastest/
   euclidean variants, so a cost is meaningless without it. */
function runMetric(run) {
  return run?.metadata?.metric || "unknown metric";
}

function runSourceLabel(run) {
  if (run.source === "bks") return "BKS";
  if (run.source === "imported") return `imported${run.solver && run.solver !== "external" ? ` · ${run.solver}` : ""}`;
  return run.created_at ? new Date(run.created_at).toLocaleString() : "saved run";
}

function solvedVariantFile(instance) {
  const metric = runMetric(instance.solved);
  const file = instance.files?.[metric];
  return file || `${instance.base_name}_${metric}.vrp`;
}

function solutionLabel(run) {
  const validity = run.validation?.valid ? "valid" : "invalid";
  const problem = run.instance_path && /cvrptw/i.test(String(run.instance_path)) ? "VRPTW" : "CVRP";
  return `${runMetric(run)} · ${problem} · ${runSourceLabel(run)} · ${run.objective_function}`
    + ` · cost ${run.cost ?? "?"} · ${run.num_routes} routes · ${validity}`;
}

function refreshSolutionControls() {
  const runs = allSolutionRecords();
  const runSelect = el("solution-run");
  const comparisonSelect = el("comparison-run");
  runSelect.innerHTML = "";
  runSelect.append(new Option("Instance only · customer locations", ""));
  if (!runs.length) {
    comparisonSelect.innerHTML = '<option value="">Select a reference run</option>';
    el("compare").disabled = true;
    el("solution-validation").className = "validation-line";
    el("solution-validation").textContent = "No saved solutions yet; instance data is stored locally.";
    return;
  }
  runs.forEach((run) => runSelect.append(new Option(solutionLabel(run), run.run_id)));
  if (state.selectedRunId && !runs.some((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = null;
  }
  runSelect.value = state.selectedRunId || "";
  const selected = runs.find((run) => run.run_id === state.selectedRunId);
  comparisonSelect.innerHTML = "";
  comparisonSelect.append(new Option("Select a reference run", ""));
  runs
    .filter((run) => run.run_id !== state.selectedRunId
      && run.objective_function === selected?.objective_function
      && run.metadata?.metric === selected?.metadata?.metric)
    .forEach((run) => comparisonSelect.append(new Option(solutionLabel(run), run.run_id)));
  el("compare").disabled = !selected || comparisonSelect.options.length <= 1;
  if (!selected) {
    el("solution-validation").className = "validation-line";
    el("solution-validation").textContent = `${state.solutionRuns.length} saved solution run(s) available locally.`;
    return;
  }
  const validation = selected?.validation || {};
  const metric = runMetric(selected);
  const line = el("solution-validation");
  line.className = `validation-line ${validation.valid ? "valid" : "invalid"}`;
  line.textContent = validation.valid
    ? `Validated on the ${metric} variant: canonical cost ${validation.routing_cost}, ${validation.num_routes} route(s).`
    : `Invalid: ${validation.error_message || validation.status || "checker rejected this solution"}`;
  // Only same-metric runs can be compared, so say so rather than leaving an
  // empty reference list looking broken.
  if (validation.valid && el("solve-metric").value !== metric) {
    line.className = "validation-line";
    line.textContent += ` The Metric selector is on ${el("solve-metric").value}; only ${metric} runs compare with this one.`;
  }
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
  el("solve-metric").value = metric;
  const rendered = await api(`/api/instances/${encodeURIComponent(instance.instance_id)}/solutions/${encodeURIComponent(runId)}/render?metric=${encodeURIComponent(metric)}`, {});
  if (state.selected !== instance) return;
  instance.rendered = rendered;
  state.lastRendered = rendered;
  state.previewGeojson = null;
  state.hiddenRoutes = new Set();
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
  instance.solution_count = state.solutionRuns.length;
  state.selectedRunId = preferredRunId;
  renderInstanceList();
  refreshSolutionControls();
  if (state.selectedRunId) {
    await displaySolutionRun(state.selectedRunId, true);
  } else {
    await displayInstanceOnly(true);
  }
}

el("solution-run").addEventListener("change", (event) => {
  displaySolutionRun(event.target.value, true).catch((error) => status(`Cannot display solution: ${error.message}`));
});

el("comparison-run").addEventListener("change", (event) => {
  el("compare").disabled = !event.target.value;
});

el("compare").addEventListener("click", async () => {
  if (!state.selected || !state.selectedRunId || !el("comparison-run").value) return;
  try {
    const data = await api(`/api/instances/${encodeURIComponent(state.selected.instance_id)}/solutions/compare`, {
      candidate_run_id: state.selectedRunId,
      reference_run_id: el("comparison-run").value,
    });
    const value = data.comparison;
    const gap = value.relative_gap_percent == null ? "n/a" : `${value.relative_gap_percent.toFixed(3)}%`;
    el("comparison-output").hidden = false;
    el("comparison-output").textContent = [
      `Result: ${value.ordering}`,
      `Cost delta: ${value.cost_delta} (gap ${gap})`,
      `Route-count delta: ${value.route_count_delta}`,
      `Candidate loads: ${value.candidate.route_loads.join(", ")}`,
      `Reference loads: ${value.reference.route_loads.join(", ")}`,
      `Directed edges +${value.route_difference.directed_edges_added} / -${value.route_difference.directed_edges_removed}`,
      `Co-routed pairs +${value.route_difference.co_routed_customer_pairs_added} / -${value.route_difference.co_routed_customer_pairs_removed}`,
    ].join("\n");
  } catch (error) {
    el("comparison-output").hidden = false;
    el("comparison-output").textContent = `Comparison failed: ${error.message}`;
  }
});

/* ── Persistent jobs ── */
const ACTIVE_JOB_STATUSES = ["queued", "running"];

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
  if (active.length) {
    const first = active[0];
    const counts = first.progress?.total ? ` ${first.progress.current || 0}/${first.progress.total}` : "";
    parts.push(`${active.length} running · ${first.kind}${counts}`);
  }
  if (failed.length) parts.push(`${failed.length} failed`);
  el("activity-count").textContent = parts.join(" · ");
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

function renderJobs() {
  const list = el("jobList");
  list.innerHTML = "";
  renderActivitySummary();
  if (!state.jobs.length) {
    list.innerHTML = '<div class="empty-note">No workbench jobs yet.</div>';
    return;
  }
  const shown = state.jobs.filter(jobMatchesFilter).slice(0, 40);
  if (!shown.length) {
    list.innerHTML = '<div class="empty-note">No job matches this filter.</div>';
    return;
  }
  shown.forEach((job) => {
    const row = document.createElement("div");
    row.className = "job-row";
    const dot = document.createElement("span");
    dot.className = `job-state ${job.status}`;
    const main = document.createElement("div");
    main.className = "job-main";
    const name = document.createElement("div");
    name.className = "job-name";
    name.textContent = `${job.kind} · ${job.status}`;
    const progress = document.createElement("div");
    progress.className = "job-progress";
    const counts = job.progress?.total ? ` ${job.progress.current || 0}/${job.progress.total}` : "";
    progress.textContent = `${job.error || job.progress?.message || ""}${counts}`;
    main.append(name, progress);
    const actions = document.createElement("div");
    actions.className = "job-actions";
    const logButton = document.createElement("button");
    logButton.type = "button";
    logButton.className = "btn btn-ghost";
    logButton.textContent = "Log";
    logButton.setAttribute("aria-expanded", String(state.openJobLogId === job.job_id));
    logButton.setAttribute("aria-controls", "jobLogPane");
    logButton.addEventListener("click", () => {
      if (state.openJobLogId === job.job_id) closeJobLog();
      else openJobLog(job);
    });
    actions.append(logButton);
    if (["queued", "running"].includes(job.status)) {
      const cancelButton = document.createElement("button");
      cancelButton.type = "button";
      cancelButton.className = "btn btn-ghost";
      cancelButton.textContent = "Cancel";
      cancelButton.addEventListener("click", async () => {
        try { await apiDelete(`/api/jobs/${encodeURIComponent(job.job_id)}`); await refreshJobs(); }
        catch (error) { status(`Cancellation failed: ${error.message}`); }
      });
      actions.append(cancelButton);
    }
    row.append(dot, main, actions);
    list.append(row);
  });
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
    state.hiddenRoutes = new Set();
    state.lastRendered = null;
    clearRouteLayers();
    await loadInstances(result.base_name);
    activateTab("visualize");
    const composition = result.summary?.composition;
    const split = composition
      ? ` — ${composition.poi_customers} POI + ${composition.parametric_customers} parametric`
      : "";
    status(`Generated ${result.base_name}: capacity ${result.summary?.capacity}, ~${result.summary?.route_count} routes${split}.`
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
    status(`Solved ${result.solution?.instance_name || "instance"}: cost ${result.cost}; checker ${result.validation?.status}.`);
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

el("solve").addEventListener("click", async () => {
  const instance = state.selected;
  if (!instance) return;
  try {
    const budget = Math.max(1, Number(el("solve-budget").value) || 30);
    const job = await submitJob("solve", {
      instance_id: instance.instance_id,
      metric: el("solve-metric").value,
      objective_function: el("solve-objective").value,
      seed: Number(el("solve-seed").value) || 0,
      time_limit: budget,
    });
    status(`Solve queued as job ${job.job_id.slice(0, 8)} (${budget} s budget).`);
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

el("clear").addEventListener("click", () => {
  previewLayers.clearLayers();
  instanceNodeLayers.clearLayers();
  poiPickLayers.clearLayers();
  clearRouteLayers();
  state.previewGeojson = null;
  state.lastRendered = null;
  if (state.selected) state.selected.rendered = null;
  renderLegend();
  status("Map cleared.");
});

/* ── Import an externally found solution ── */
function resetImportPanel() {
  el("import-text").value = "";
  el("import-file").value = "";
  el("import-label").value = "";
  el("import-validation").className = "validation-line";
  el("import-validation").textContent =
    "Imported solutions are checked against the metric selected above before they are stored.";
}

el("import-open").addEventListener("click", () => {
  const panel = el("import-panel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) resetImportPanel();
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
  const metric = el("solve-metric").value;
  try {
    line.className = "validation-line";
    line.textContent = `Checking the routes against the ${metric} variant…`;
    const result = await api(
      `/api/instances/${encodeURIComponent(instance.instance_id)}/solutions/import`,
      {
        metric,
        objective_function: el("solve-objective").value,
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
loadInstances().catch((error) => console.warn("Unable to list workspace instances", error));
refreshJobs().catch((error) => console.warn("Unable to list workbench jobs", error));
setInterval(() => refreshJobs().catch(() => {}), 1000);
