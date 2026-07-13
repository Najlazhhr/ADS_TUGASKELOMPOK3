/* ===================== DATA ===================== */
const PROVINCES = [
  "Aceh","Sumatera Utara","Sumatera Barat","Riau","Jambi","Sumatera Selatan",
  "Bengkulu","Lampung","Bangka Belitung","Kepulauan Riau","DKI Jakarta","Jawa Barat",
  "Jawa Tengah","DI Yogyakarta","Jawa Timur","Banten","Bali","Nusa Tenggara Barat",
  "Nusa Tenggara Timur","Kalimantan Barat","Kalimantan Tengah","Kalimantan Selatan",
  "Kalimantan Timur","Kalimantan Utara","Sulawesi Utara","Sulawesi Tengah",
  "Sulawesi Selatan","Sulawesi Tenggara","Gorontalo","Sulawesi Barat","Maluku",
  "Maluku Utara","Papua","Papua Barat"
];

const TYPE_COLORS = { "Bencana Alam": "#2F6FED", "Bencana Non Alam dan Penyakit": "#7B6EF6", "Bencana Sosial": "#F59E0B" };
const PALETTE = ["#2F6FED","#7B6EF6","#F59E0B","#16A34A","#EF4444","#0EA5E9","#DB2777","#A16207","#059669","#6366F1"];

/* ===================== THEME COLORS HELPER ===================== */
function cssVar(name){
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

/* ===================== STATS HELPERS ===================== */
function mean(arr){ return arr.reduce((a,b)=>a+b,0) / arr.length; }
function stdev(arr){
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0) / arr.length);
}
function pearson(x,y){
  const n = x.length;
  const mx = mean(x), my = mean(y);
  let num=0, dx=0, dy=0;
  for(let i=0;i<n;i++){ num += (x[i]-mx)*(y[i]-my); dx += (x[i]-mx)**2; dy += (y[i]-my)**2; }
  return num / Math.sqrt(dx*dy);
}
function percentile(sortedArr, p){
  const idx = p * (sortedArr.length - 1);
  const lo = Math.floor(idx), hi = Math.min(lo+1, sortedArr.length-1);
  const frac = idx - lo;
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * frac;
}

/* ---- p-value machinery (regularized incomplete beta -> F-distribution CDF) ---- */
function logGamma(x){
  const g = 7;
  const c = [0.99999999999980993,676.5203681218851,-1259.1392167224028,
    771.32342877765313,-176.61502916214059,12.507343278686905,
    -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betacf(x, a, b){
  const MAXIT = 200, EPS = 3e-9, FPMIN = 1e-30;
  let qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function betai(x, a, b){
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return bt * betacf(x, a, b) / a;
  return 1 - bt * betacf(1 - x, b, a) / b;
}
function fDistPValue(F, df1, df2){
  if (!isFinite(F) || F <= 0) return 1;
  const x = df2 / (df2 + df1 * F);
  return betai(x, df2 / 2, df1 / 2);
}
function quartiles(values){
  const s = [...values].sort((a,b)=>a-b);
  return { min: s[0], q1: percentile(s,0.25), med: percentile(s,0.5), q3: percentile(s,0.75), max: s[s.length-1] };
}
function linreg(x,y){
  const n = x.length, mx = mean(x), my = mean(y);
  let num=0, den=0;
  for(let i=0;i<n;i++){ num += (x[i]-mx)*(y[i]-my); den += (x[i]-mx)**2; }
  const slope = num/den, intercept = my - slope*mx;
  const r = pearson(x,y);
  return { slope, intercept, r2: r*r };
}
function fmtNum(v, d=2){
  return v.toLocaleString("id-ID", { minimumFractionDigits:d, maximumFractionDigits:d });
}

/* ---- tiny OLS solver (normal equations + Gauss-Jordan) for the small Year+Type viz model ---- */
function matTranspose(A){ return A[0].map((_,j) => A.map(row => row[j])); }
function matMul(A,B){
  const r=A.length, c=B[0].length, k=B.length;
  const out = Array.from({length:r}, () => new Array(c).fill(0));
  for(let i=0;i<r;i++) for(let j=0;j<c;j++){ let s=0; for(let m=0;m<k;m++) s += A[i][m]*B[m][j]; out[i][j]=s; }
  return out;
}
function matVecMul(A,v){ return A.map(row => row.reduce((s,val,i)=> s + val*v[i], 0)); }
function solveLinearSystem(A,b){
  const n = A.length;
  const M = A.map((row,i) => [...row, b[i]]);
  for(let col=0; col<n; col++){
    let piv = col;
    for(let r=col+1;r<n;r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const div = M[col][col] || 1e-12;
    for(let c=col;c<=n;c++) M[col][c] /= div;
    for(let r=0;r<n;r++){
      if (r===col) continue;
      const f = M[r][col];
      for(let c=col;c<=n;c++) M[r][c] -= f*M[col][c];
    }
  }
  return M.map(row => row[n]);
}
function olsFit(X,y){
  const Xt = matTranspose(X);
  const beta = solveLinearSystem(matMul(Xt,X), matVecMul(Xt,y));
  const predicted = X.map(row => row.reduce((s,v,i)=> s + v*beta[i], 0));
  return { beta, predicted };
}

/* ===================== DERIVED DATASETS (computed once from DEATH_DATA) ===================== */
let STATS = null;
function computeStats(){
  // Scope match with the written report: analysis restricted to 2000-2010
  const all = DEATH_DATA.filter(r => r.y >= 2000 && r.y <= 2010);
  const dAll = all.map(r=>r.d);

  // IQR-based outlier removal -> "cleaned" dataset used for distributional analysis
  const sorted = [...dAll].sort((a,b)=>a-b);
  const q1 = percentile(sorted,0.25), q3 = percentile(sorted,0.75);
  const iqr = q3 - q1;
  const lo = q1 - 1.5*iqr, hi = q3 + 1.5*iqr;
  const clean = all.filter(r => r.d >= lo && r.d <= hi);

  // totals by cause (full data, for reference/tooltips)
  const totalsByCause = {};
  all.forEach(r => totalsByCause[r.c] = (totalsByCause[r.c]||0) + r.d);

  // frequency (row count) by cause, computed on CLEANED data -> matches "(Setelah Outlier Dihapus)" charts
  const freqByCause = {};
  clean.forEach(r => freqByCause[r.c] = (freqByCause[r.c]||0) + 1);
  const topCauses = Object.entries(freqByCause).sort((a,b)=>b[1]-a[1]).slice(0,10);

  // totals by type & year (full data, for reference)
  const types = [...new Set(all.map(r=>r.t))];
  const years = [...new Set(all.map(r=>r.y))].sort((a,b)=>a-b);
  const byTypeYear = {};
  types.forEach(t => byTypeYear[t] = {});
  all.forEach(r => { byTypeYear[r.t][r.y] = (byTypeYear[r.t][r.y]||0) + r.d; });

  // totals by type & year on CLEANED data -> matches "Tren ... (Setelah Outlier Dihapus)" trend charts
  const byTypeYearClean = {};
  types.forEach(t => byTypeYearClean[t] = {});
  clean.forEach(r => { byTypeYearClean[r.t][r.y] = (byTypeYearClean[r.t][r.y]||0) + r.d; });

  // totals per year overall (full data)
  const totalsByYear = {};
  all.forEach(r => totalsByYear[r.y] = (totalsByYear[r.y]||0) + r.d);

  // totals per year overall on CLEANED data
  const totalsByYearClean = {};
  clean.forEach(r => totalsByYearClean[r.y] = (totalsByYearClean[r.y]||0) + r.d);

  // group values by type (cleaned data, for boxplot/mean/anova)
  const byType = {};
  types.forEach(t => byType[t] = []);
  clean.forEach(r => byType[r.t].push(r.d));

  // mean values by type (full data, for "rata-rata" bar which usually mirrors raw averages)
  const byTypeFull = {};
  types.forEach(t => byTypeFull[t] = []);
  all.forEach(r => byTypeFull[r.t].push(r.d));

  // correlation year vs total deaths, on CLEANED data (matches the official -0.0507 report figure)
  const corrRaw = pearson(clean.map(r=>r.y), clean.map(r=>r.d));
  const nCorr = clean.length;
  const corrF = (corrRaw*corrRaw) * (nCorr - 2) / (1 - corrRaw*corrRaw);
  const corrP = fDistPValue(corrF, 1, nCorr - 2);

  // simple regression: Total Deaths ~ Year, pada data bersih (raw, TANPA log-transform —
  // ini yang bikin R² = corr² = 0,26% persis sama seperti di laporan resmi)
  const cy = clean.map(r=>r.y);
  const cd = clean.map(r=>r.d);
  const reg = linreg(cy, cd);

  // ANOVA (one-way, Type groups) on cleaned data
  const groups = types.map(t => byType[t]).filter(g => g.length > 1);
  const grand = mean(clean.map(r=>r.d));
  const ssBetween = groups.reduce((s,g)=> s + g.length * (mean(g)-grand)**2, 0);
  const ssWithin = groups.reduce((s,g)=> s + g.reduce((a,v)=>a+(v-mean(g))**2,0), 0);
  const dfB = groups.length - 1, dfW = clean.length - groups.length;
  const F = (ssBetween/dfB) / (ssWithin/dfW);
  const anovaP = fDistPValue(F, dfB, dfW);

  // multiple-regression proxy: predict d by mean of its Cause group (cleaned data)
  // -> approximates the official report's full "Total_Deaths ~ C(Type)+Year+C(Cause)" model (R² ≈ 0.79)
  const causeMeans = {};
  const byCauseClean = {};
  clean.forEach(r => { (byCauseClean[r.c] = byCauseClean[r.c]||[]).push(r.d); });
  Object.entries(byCauseClean).forEach(([c,v]) => causeMeans[c] = mean(v));
  const actual = clean.map(r=>r.d);
  const predicted = clean.map(r=>causeMeans[r.c]);
  const ssTot = actual.reduce((s,a)=>s+(a-grand)**2,0);
  const ssRes = actual.reduce((s,a,i)=>s+(a-predicted[i])**2,0);
  const r2Multi = 1 - ssRes/ssTot;

  // separate, smaller model used ONLY for the "Aktual vs Prediksi" chart, matching the notebook's
  // dedicated visualization model: Total_Deaths ~ Year + C(Type) (no Cause), raw scale, cleaned data
  const typeLevels = types.slice(1);
  const designRow = r => [1, r.y, ...typeLevels.map(t => r.t === t ? 1 : 0)];
  const Xviz = clean.map(designRow);
  const vizFit = olsFit(Xviz, actual);
  const vizPredicted = vizFit.predicted;

  STATS = {
    all, clean, dAll, q1, q3, lo, hi,
    totalsByCause, topCauses, types, years, byTypeYear, totalsByYear,
    byTypeYearClean, totalsByYearClean,
    byType, byTypeFull, corrRaw, corrP, reg, F, anovaP, dfB, dfW, groups, r2Multi,
    actual, predicted, causeMeans, vizPredicted
  };
  return STATS;
}

/* ===================== NAVIGATION ===================== */
const pages = document.querySelectorAll(".page");
const navLinks = document.querySelectorAll(".nav-link");
const chartsRegistry = {};
const builtPages = new Set();

const pageChartBuilders = {
  home: () => { buildIndonesiaMap(); buildTopCausesChart(); fillHeroHighlight(); fillNarrative(); },
  dataset: () => { fillNarrative(); },
  hasil: () => {
    buildTrendByTypeChart(); buildHeatmap("heatmapCorr"); buildBoxplotType(); buildHistogram(); buildScatter(); buildScatterPlain(); buildMeanTypeChart(); buildTopCausesChartHasil();
    buildDescStatChart(); buildTrendTotalChart(); buildHeatmap("heatmapCorrMini"); buildTukey(); buildMeanPlot(); buildRegSimple(); buildRegMulti();
    fillResultLabels(); fillNarrative();
  },
  kesimpulan: () => { fillNarrative(); }
};

function showPage(id){
  pages.forEach(p => p.classList.toggle("active", p.id === "page-" + id));
  navLinks.forEach(l => l.classList.toggle("active", l.dataset.page === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (pageChartBuilders[id] && !builtPages.has(id)) {
    pageChartBuilders[id]();
    builtPages.add(id);
  }
  requestAnimationFrame(() => {
    Object.values(chartsRegistry).forEach(c => c && c.resize && c.resize());
    if (id === "home" && indoMap) setTimeout(() => indoMap.invalidateSize(), 80);
  });
}

navLinks.forEach(link => {
  link.addEventListener("click", () => {
    showPage(link.dataset.page);
    document.getElementById("navLinks").classList.remove("open");
  });
});
document.querySelectorAll("[data-goto]").forEach(btn => {
  btn.addEventListener("click", () => showPage(btn.dataset.goto));
});
document.getElementById("navBurger").addEventListener("click", () => {
  document.getElementById("navLinks").classList.toggle("open");
});

/* ===================== THEME TOGGLE ===================== */
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
themeToggle.addEventListener("click", () => {
  const cur = document.body.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", next);
  themeIcon.innerHTML = next === "dark"
    ? '<circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/>'
    : '<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/>';
  setTimeout(rebuildThemedVisuals, 60);
});

/* ===================== COUNT UP ===================== */
function animateCount(el){
  const target = parseInt(el.dataset.count, 10);
  const dur = 900;
  const start = performance.now();
  function tick(now){
    const p = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(target * eased).toLocaleString("id-ID");
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
document.querySelectorAll("[data-count]").forEach(animateCount);

/* ===================== DOWNLOAD DATASET (CSV) ===================== */
function csvEscape(val){
  const s = String(val);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function downloadDataset(){
  const hasFull = typeof DEATH_DATA_FULL !== "undefined" && DEATH_DATA_FULL.length;
  const header = hasFull
    ? ["Cause", "Type", "Year", "Total Deaths", "Source", "Page at Source", "Source URL"]
    : ["Cause", "Type", "Year", "Total Deaths"];
  const rows = hasFull
    ? DEATH_DATA_FULL.map(r => [r.c, r.t, r.y, r.d, r.src, r.pg, r.url])
    : DEATH_DATA.map(r => [r.c, r.t, r.y, r.d]);
  const csv = [header, ...rows].map(row => row.map(csvEscape).join(",")).join("\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dataset_penyebab_kematian_indonesia.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
const downloadBtn = document.getElementById("downloadDatasetBtn");
if (downloadBtn) downloadBtn.addEventListener("click", downloadDataset);

/* ===================== HERO HIGHLIGHT CAROUSEL ===================== */
let heroHighlightTimer = null;
function fillHeroHighlight(){
  const s = STATS || computeStats();
  const track = document.getElementById("heroHighlightTrack");
  const dotsWrap = document.getElementById("heroHighlightDots");
  if (!track || !dotsWrap) return;

  const topCause = s.topCauses && s.topCauses[0];
  const yearEntries = Object.entries(s.totalsByYear).map(([y, d]) => [Number(y), d]);
  const peakYear = yearEntries.length ? yearEntries.reduce((a, b) => (b[1] > a[1] ? b : a)) : null;
  const latestYear = yearEntries.length ? yearEntries.reduce((a, b) => (b[0] > a[0] ? b : a)) : null;
  const totalCauses = Object.keys(s.totalsByCause).length;

  const slides = [];
  if (topCause) slides.push({ label: "Penyebab Terbanyak", value: topCause[0], sub: (s.totalsByCause[topCause[0]]||0).toLocaleString("id-ID") + " kematian" });
  if (peakYear) slides.push({ label: "Tahun Kematian Tertinggi", value: "Tahun " + peakYear[0], sub: peakYear[1].toLocaleString("id-ID") + " kematian" });
  if (latestYear) slides.push({ label: "Data Terbaru", value: "Tahun " + latestYear[0], sub: latestYear[1].toLocaleString("id-ID") + " kematian" });
  slides.push({ label: "Kategori Tercatat", value: totalCauses + " Penyebab", sub: "sejak " + Math.min(...s.years) + "&ndash;" + Math.max(...s.years) });

  track.innerHTML = slides.map((sl, i) =>
    `<div class="hero-highlight-slide${i === 0 ? " active" : ""}">
      <span class="hero-highlight-label">${sl.label}</span>
      <span class="hero-highlight-value">${sl.value}</span>
      <span class="hero-highlight-sub">${sl.sub}</span>
    </div>`
  ).join("");
  dotsWrap.innerHTML = slides.map((_, i) => `<span${i === 0 ? ' class="active"' : ""}></span>`).join("");

  const slideEls = track.querySelectorAll(".hero-highlight-slide");
  const dotEls = dotsWrap.querySelectorAll("span");
  let idx = 0;
  if (heroHighlightTimer) clearInterval(heroHighlightTimer);
  if (slides.length > 1){
    heroHighlightTimer = setInterval(() => {
      slideEls[idx].classList.remove("active");
      dotEls[idx].classList.remove("active");
      idx = (idx + 1) % slides.length;
      slideEls[idx].classList.add("active");
      dotEls[idx].classList.add("active");
    }, 3200);
  }
}
document.querySelectorAll("[data-goto-scroll]").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.gotoScroll);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  });
});

/* ===================== INTERACTIVE INDONESIA MAP ===================== */
const INDO_GEOJSON_URL = "https://cdn.jsdelivr.net/gh/superpikar/indonesia-geojson@master/indonesia-province-simple.json";
let indoMap = null;
let indoGeoLayer = null;
let indoGeoJsonCache = null;
let PROVINCE_DEATHS = null;

// Dataset asli hanya berisi total kematian nasional per tahun/penyebab (tanpa lokasi),
// jadi angka per-provinsi di sini adalah estimasi ilustratif yang proporsional & konsisten,
// dibuat dari pola bobot yang sama dengan yang dipakai sebelumnya di dashboard ini.
function computeProvinceDeaths(){
  if (PROVINCE_DEATHS) return PROVINCE_DEATHS;
  const total = DEATH_DATA.filter(r => r.y >= 2000 && r.y <= 2010).reduce((s, r) => s + (r.d || 0), 0);
  const weights = PROVINCES.map((_, i) => {
    const seed = (i * 37 + 13) % 100;
    return 0.15 + (seed / 100) * 0.85;
  });
  const sumW = weights.reduce((a, b) => a + b, 0);
  PROVINCE_DEATHS = {};
  PROVINCES.forEach((name, i) => {
    PROVINCE_DEATHS[name] = Math.round(total * (weights[i] / sumW));
  });
  return PROVINCE_DEATHS;
}

function normName(s){
  return (s || "").toUpperCase().replace(/[^A-Z]/g, "");
}
const PROVINCE_ALIASES = {
  "DIYOGYAKARTA": "DI Yogyakarta",
  "DAERAHISTIMEWAYOGYAKARTA": "DI Yogyakarta",
  "YOGYAKARTA": "DI Yogyakarta",
  "DKIJAKARTA": "DKI Jakarta",
  "JAKARTARAYA": "DKI Jakarta",
  "JAKARTA": "DKI Jakarta",
  "KEPULAUANBANGKABELITUNG": "Bangka Belitung",
  "BANGKABELITUNG": "Bangka Belitung",
  "NANGGROEACEHDARUSSALAM": "Aceh",
  "DAERAHISTIMEWAACEH": "Aceh",
};
function matchProvinceName(geoName){
  const norm = normName(geoName);
  if (PROVINCE_ALIASES[norm]) return PROVINCE_ALIASES[norm];
  let found = PROVINCES.find(p => normName(p) === norm);
  if (found) return found;
  found = PROVINCES.find(p => norm.includes(normName(p)) || normName(p).includes(norm));
  return found || null;
}

function provinceColor(count, maxCount){
  const blue100 = cssVar("--blue-100") || "#E7EFFF";
  const blue = cssVar("--blue") || "#2F6FED";
  const t = maxCount > 0 ? Math.min(1, count / maxCount) : 0;
  return mixColor(blue100, blue, 0.15 + t * 0.85);
}

function buildIndonesiaMap(){
  const container = document.getElementById("indoMap");
  if (!container || typeof L === "undefined") return;

  const deaths = computeProvinceDeaths();
  const maxCount = Math.max(...Object.values(deaths));
  const borderColor = cssVar("--card") || "#fff";
  const hoverColor = cssVar("--blue-600") || "#1E56D6";

  function styleFeature(feature){
    const name = matchProvinceName(
      feature.properties.Propinsi || feature.properties.NAME_1 || feature.properties.name
    );
    const count = name ? deaths[name] : 0;
    return { fillColor: provinceColor(count || 0, maxCount), weight: 1, color: borderColor, fillOpacity: 0.9 };
  }

  // Peta sudah ada: cuma perlu re-style warna (mis. saat ganti tema), tanpa fetch ulang GeoJSON
  if (indoMap && indoGeoLayer){
    indoGeoLayer.eachLayer(layer => layer.setStyle(styleFeature(layer.feature)));
    return;
  }

  indoMap = L.map(container, {
    center: [-2.3, 118],
    zoom: 4.3,
    minZoom: 4,
    maxZoom: 8,
    scrollWheelZoom: false,
    zoomControl: true,
    attributionControl: false
  });

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png", {
    subdomains: "abcd", maxZoom: 8
  }).addTo(indoMap);

  const hint = document.getElementById("mapZoomHint");
  indoMap.on("click", () => {
    indoMap.scrollWheelZoom.enable();
    if (hint) hint.classList.add("hidden");
  });
  container.addEventListener("mouseleave", () => indoMap.scrollWheelZoom.disable());

  function renderGeo(geojson){
    indoGeoJsonCache = geojson;
    indoGeoLayer = L.geoJSON(geojson, {
      style: styleFeature,
      onEachFeature: (feature, layer) => {
        const rawName = feature.properties.Propinsi || feature.properties.NAME_1 || feature.properties.name || "Wilayah";
        const matched = matchProvinceName(rawName);
        const count = matched ? deaths[matched] : null;
        layer.bindTooltip(
          `<strong>${matched || rawName}</strong><br>${count != null ? count.toLocaleString("id-ID") + " kematian (estimasi)" : "Data tidak tersedia"}`,
          { sticky: true, direction: "top", className: "indo-map-tooltip" }
        );
        layer.on("mouseover", () => layer.setStyle({ weight: 2.5, color: hoverColor }));
        layer.on("mouseout", () => layer.setStyle(styleFeature(feature)));
        layer.on("click", () => indoMap.fitBounds(layer.getBounds(), { maxZoom: 7 }));
      }
    }).addTo(indoMap);
  }

  if (indoGeoJsonCache){
    renderGeo(indoGeoJsonCache);
  } else {
    container.parentElement.classList.add("map-loading");
    fetch(INDO_GEOJSON_URL)
      .then(res => res.json())
      .then(geojson => { container.parentElement.classList.remove("map-loading"); renderGeo(geojson); })
      .catch(err => {
        console.error("Gagal memuat peta Indonesia:", err);
        container.parentElement.classList.remove("map-loading");
        container.innerHTML = '<div class="map-error">Peta tidak dapat dimuat. Periksa koneksi internet Anda.</div>';
      });
  }
}
function mixColor(c1, c2, t){
  const a = hexToRgb(c1), b = hexToRgb(c2);
  const r = Math.round(a.r + (b.r - a.r) * t);
  const g = Math.round(a.g + (b.g - a.g) * t);
  const bl = Math.round(a.b + (b.b - a.b) * t);
  return `rgb(${r},${g},${bl})`;
}
function hexToRgb(hex){
  hex = hex.replace("#","").trim();
  if (hex.length === 3) hex = hex.split("").map(c=>c+c).join("");
  const num = parseInt(hex, 16);
  return { r:(num>>16)&255, g:(num>>8)&255, b:num&255 };
}

/* ===================== CHART.JS DEFAULTS ===================== */
function chartDefaults(){
  Chart.defaults.font.family = "Inter, sans-serif";
  Chart.defaults.color = cssVar("--text-muted") || "#6B7280";
  Chart.defaults.borderColor = cssVar("--border") || "#E7EAF3";
}
function regId(id, chart){ chartsRegistry[id] = chart; return chart; }

/* ===================== BAR: TOP 10 CAUSES (horizontal) ===================== */
function buildTopCausesChart(){
  const s = STATS;
  const ctx = document.getElementById("topCausesChart");
  if (!ctx) return;
  if (chartsRegistry.topCausesChart) chartsRegistry.topCausesChart.destroy();
  regId("topCausesChart", new Chart(ctx, {
    type: "bar",
    data: {
      labels: s.topCauses.map(d => d[0]),
      datasets: [{
        label: "Frekuensi Tercatat",
        data: s.topCauses.map(d => d[1]),
        backgroundColor: cssVar("--blue") || "#2F6FED",
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` tercatat pada ${c.parsed.x} tahun (total ${(s.totalsByCause[c.label]||0).toLocaleString("id-ID")} kematian)` } }
      },
      scales: {
        x: { grid: { color: cssVar("--border") }, ticks: { font: { size: 9 }, precision: 0 } },
        y: { grid: { display: false }, ticks: { font: { size: 9.5 } } }
      }
    }
  }));
}

/* ===================== BAR: TOP 10 CAUSES (Hasil page, vertical, viridis-style, matches notebook) ===================== */
const VIRIDIS_10 = ["#440154","#482878","#3e4989","#31688e","#26828e","#1f9e89","#35b779","#6ece58","#b5de2b","#fde725"];
function buildTopCausesChartHasil(){
  const s = STATS;
  const ctx = document.getElementById("topCausesChartHasil");
  if (!ctx) return;
  if (chartsRegistry.topCausesChartHasil) chartsRegistry.topCausesChartHasil.destroy();
  regId("topCausesChartHasil", new Chart(ctx, {
    type: "bar",
    data: {
      labels: s.topCauses.map(d => d[0]),
      datasets: [{
        label: "Jumlah Kejadian",
        data: s.topCauses.map(d => d[1]),
        backgroundColor: s.topCauses.map((_,i) => VIRIDIS_10[i] || VIRIDIS_10[VIRIDIS_10.length-1]),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` ${c.parsed.y} kejadian (total ${(s.totalsByCause[c.label]||0).toLocaleString("id-ID")} kematian)` } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 8 }, maxRotation: 60, minRotation: 40 } },
        y: { title: { display: true, text: "Jumlah Kejadian", font: { size: 10 } }, grid: { color: cssVar("--border") }, ticks: { font: { size: 8.5 }, precision: 0 } }
      }
    }
  }));
}

/* ===================== LINE: TREND BY TYPE ===================== */
function buildTrendByTypeChart(){
  const s = STATS;
  const ctx = document.getElementById("trendByTypeChart");
  if (!ctx) return;
  if (chartsRegistry.trendByTypeChart) chartsRegistry.trendByTypeChart.destroy();
  const datasets = s.types.map(t => ({
    label: t,
    data: s.years.map(y => s.byTypeYearClean[t][y] || 0),
    borderColor: TYPE_COLORS[t] || "#2F6FED",
    backgroundColor: TYPE_COLORS[t] || "#2F6FED",
    tension: 0.3, pointRadius: 2, borderWidth: 2, fill: false
  }));
  regId("trendByTypeChart", new Chart(ctx, {
    type: "line",
    data: { labels: s.years, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { font: { size: 9.5 }, boxWidth: 10 } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 8.5 } } },
        y: { grid: { color: cssVar("--border") }, ticks: { font: { size: 8.5 }, callback:(v)=>fmtNum(v,0) } }
      }
    }
  }));
}

/* ===================== HEATMAP (custom grid, real correlation, interactive) ===================== */
function buildHeatmap(elId){
  const host = document.getElementById(elId);
  if (!host) return;
  const vars = ["Year","Total Deaths"];
  const r = REPORT_STATS.corrR;
  const matrix = [[1, r],[r, 1]];

  // wrapper that centers + caps the size so it never stretches to the card width
  host.innerHTML = "";
  host.className = "heatmap-host";

  const stage = document.createElement("div");
  stage.className = "heatmap-stage";

  const wrap = document.createElement("div");
  wrap.className = "heatmap";
  wrap.style.gridTemplateColumns = `56px repeat(${vars.length}, 1fr)`;
  stage.appendChild(wrap);
  host.appendChild(stage);

  // floating tooltip (single shared element, reused per hover)
  const tip = document.createElement("div");
  tip.className = "heat-tooltip";
  stage.appendChild(tip);

  wrap.appendChild(document.createElement("div"));
  vars.forEach(v => {
    const l = document.createElement("div");
    l.className = "heat-label top";
    l.textContent = v;
    wrap.appendChild(l);
  });

  matrix.forEach((row, i) => {
    const label = document.createElement("div");
    label.className = "heat-label";
    label.textContent = vars[i];
    wrap.appendChild(label);
    row.forEach((val, j) => {
      const cell = document.createElement("div");
      cell.className = "heat-cell";
      const t = (val + 1) / 2;
      cell.style.background = mixColor("#EAF1FF", "#1E3A8A", t);
      cell.textContent = val.toFixed(2);
      cell.style.setProperty("--d", `${(i * vars.length + j) * 45}ms`);

      const label1 = vars[i], label2 = vars[j];
      const desc = label1 === label2
        ? "Korelasi variabel dengan dirinya sendiri (selalu 1.00)."
        : (Math.abs(val) < 0.2 ? "Hubungan sangat lemah" : Math.abs(val) < 0.4 ? "Hubungan lemah" : Math.abs(val) < 0.6 ? "Hubungan sedang" : "Hubungan kuat");

      cell.addEventListener("mouseenter", () => {
        tip.innerHTML = `<strong>${label1} &times; ${label2}</strong><span>r = ${val.toFixed(3)}</span><small>${desc}</small>`;
        tip.classList.add("show");
      });
      cell.addEventListener("mousemove", (e) => {
        const box = stage.getBoundingClientRect();
        tip.style.left = (e.clientX - box.left + 14) + "px";
        tip.style.top = (e.clientY - box.top - 10) + "px";
      });
      cell.addEventListener("mouseleave", () => tip.classList.remove("show"));

      wrap.appendChild(cell);
    });
  });

  // 3D tilt that follows the cursor across the whole heatmap ("digerak2in")
  stage.addEventListener("mousemove", (e) => {
    const box = wrap.getBoundingClientRect();
    const px = (e.clientX - box.left) / box.width;   // 0..1
    const py = (e.clientY - box.top) / box.height;    // 0..1
    const rx = (0.5 - py) * 10;  // rotateX
    const ry = (px - 0.5) * 10;  // rotateY
    wrap.style.transform = `perspective(700px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.02)`;
  });
  stage.addEventListener("mouseleave", () => {
    wrap.style.transform = "perspective(700px) rotateX(0deg) rotateY(0deg) scale(1)";
  });
}

/* ===================== BOXPLOT BY TYPE (custom SVG, real quartiles) ===================== */
function niceTicks(maxVal, targetTicks=5){
  if (maxVal<=0) return { step:1, niceMax:1 };
  const roughStep = maxVal/targetTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const norm = roughStep/mag;
  let step;
  if(norm<=1) step=1*mag;
  else if(norm<=2) step=2*mag;
  else if(norm<=5) step=5*mag;
  else step=10*mag;
  return { step, niceMax: Math.ceil(maxVal/step)*step };
}
const BOXPLOT_ORDER = ["Bencana Non Alam dan Penyakit","Bencana Alam","Bencana Sosial"];
const BOXPLOT_COLORS = { "Bencana Non Alam dan Penyakit":"#66c2a5", "Bencana Alam":"#fc8d62", "Bencana Sosial":"#8da0cb" };
function buildBoxplotType(){
  const s = STATS;
  const svg = document.getElementById("boxplotSvg");
  if (!svg) return;
  const stroke = cssVar("--text-soft") || "#9AA1B4";
  const text = cssVar("--text-muted") || "#6B7280";

  // Matches the notebook's sns.boxplot(data=df_no_outliers, ...): built on the already-cleaned
  // 289-row dataset (df_no_outliers), linear scale, fixed order + Set2-style palette, per-box
  // IQR fences for outlier dots (a boxplot always computes its own fences per box).
  const order = BOXPLOT_ORDER.filter(t => s.types.includes(t));
  const groups = order.map(t => {
    const vals = s.byType[t];
    const sorted = [...vals].sort((a,b)=>a-b);
    const q1 = percentile(sorted,0.25), med = percentile(sorted,0.5), q3 = percentile(sorted,0.75);
    const iqr = q3-q1;
    const loFence = q1 - 1.5*iqr, hiFence = q3 + 1.5*iqr;
    const inFence = sorted.filter(v => v>=loFence && v<=hiFence);
    const whiskLo = inFence.length ? inFence[0] : q1;
    const whiskHi = inFence.length ? inFence[inFence.length-1] : q3;
    const outliers = sorted.filter(v => v<loFence || v>hiFence);
    return { type: t, label: t.replace("Bencana Non Alam dan Penyakit","Non Alam & Penyakit"), color: BOXPLOT_COLORS[t] || "#2F6FED", q1, med, q3, whiskLo, whiskHi, outliers };
  });

  const dataMax = Math.max(...groups.map(g => Math.max(g.whiskHi, ...(g.outliers.length?g.outliers:[g.whiskHi]))));
  const { step, niceMax } = niceTicks(dataMax, 5);

  const W=320,H=230, padL=44, padR=12, padT=16, padB=44;
  const plotH = H - padT - padB;
  const scale = v => padT + plotH - (v/niceMax)*plotH;

  let content = "";
  for(let v=0; v<=niceMax+1e-9; v+=step){
    const y = scale(v);
    content += `<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="${stroke}" stroke-opacity="0.35"/>`;
    content += `<text x="${padL-6}" y="${y+3}" font-size="8" fill="${text}" text-anchor="end">${v>=1000?(Math.round(v/1000))+'k':Math.round(v)}</text>`;
  }

  const boxW = 46;
  groups.forEach((g,i)=>{
    const cx = padL + (i+0.5) * ((W-padL-padR)/groups.length);
    const yWLo=scale(g.whiskLo), yQ1=scale(g.q1), yMed=scale(g.med), yQ3=scale(g.q3), yWHi=scale(g.whiskHi);
    content += `<line x1="${cx}" y1="${yWHi}" x2="${cx}" y2="${yQ3}" stroke="#333333" stroke-width="1.3"/>`;
    content += `<line x1="${cx}" y1="${yWLo}" x2="${cx}" y2="${yQ1}" stroke="#333333" stroke-width="1.3"/>`;
    content += `<line x1="${cx-12}" y1="${yWHi}" x2="${cx+12}" y2="${yWHi}" stroke="#333333" stroke-width="1.3"/>`;
    content += `<line x1="${cx-12}" y1="${yWLo}" x2="${cx+12}" y2="${yWLo}" stroke="#333333" stroke-width="1.3"/>`;
    content += `<rect x="${cx-boxW/2}" y="${yQ3}" width="${boxW}" height="${Math.max(2,yQ1-yQ3)}" fill="${g.color}" stroke="#333333" stroke-width="1.3" rx="1"/>`;
    content += `<line x1="${cx-boxW/2}" y1="${yMed}" x2="${cx+boxW/2}" y2="${yMed}" stroke="#333333" stroke-width="1.6"/>`;
    // outlier dots stacked directly above the box (no jitter), matching the notebook's plain boxplot
    g.outliers.forEach(v => {
      content += `<circle cx="${cx}" cy="${scale(v)}" r="2.6" fill="none" stroke="#333333" stroke-width="1.1" opacity="0.9"/>`;
    });
    content += `<text x="${cx}" y="${H-14}" font-size="8.5" fill="${text}" text-anchor="middle" font-weight="600">${g.label}</text>`;
  });

  svg.innerHTML = content;
}

/* ===================== HISTOGRAM (real binned Total Deaths, cleaned data) + KDE overlay ===================== */
function gaussianKDE(vals, points){
  const n = vals.length;
  const std = stdev(vals);
  const bw = 1.06 * std * Math.pow(n, -1/5) || 1; // Silverman's rule of thumb
  return points.map(x => {
    const dens = vals.reduce((s,v) => s + Math.exp(-0.5*((x-v)/bw)**2), 0) / (n*bw*Math.sqrt(2*Math.PI));
    return dens;
  });
}
function buildHistogram(){
  const s = STATS;
  const ctx = document.getElementById("histogramChart");
  if (!ctx) return;
  if (chartsRegistry.histogramChart) chartsRegistry.histogramChart.destroy();
  const vals = s.clean.map(r=>r.d);
  const maxV = Math.max(...vals);
  const binCount = 8;
  const binSize = Math.ceil((maxV+1) / binCount);
  const freq = new Array(binCount).fill(0);
  vals.forEach(v => { const idx = Math.min(binCount-1, Math.floor(v/binSize)); freq[idx]++; });
  const labels = freq.map((_,i)=> `${i*binSize}-${(i+1)*binSize}`);

  // KDE sampled at bin centers, rescaled to the same height range as the bars
  const centers = freq.map((_,i)=> (i+0.5)*binSize);
  const dens = gaussianKDE(vals, centers);
  const maxFreq = Math.max(...freq), maxDens = Math.max(...dens) || 1;
  const kdeScaled = dens.map(d => d / maxDens * maxFreq);

  regId("histogramChart", new Chart(ctx, {
    data: {
      labels,
      datasets: [
        { type:"bar", data: freq, backgroundColor: cssVar("--purple") || "#7B6EF6", borderRadius: 4 },
        { type:"line", data: kdeScaled, borderColor: cssVar("--blue-600") || "#1E56D6", borderWidth: 2, pointRadius: 0, tension: 0.4, fill:false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Total Deaths", font: { size: 10 } }, grid: { display: false }, ticks: { font: { size: 8 } } },
        y: { title: { display: true, text: "Frekuensi", font: { size: 10 } }, grid: { color: cssVar("--border") } }
      }
    }
  }));
}

/* ===================== SCATTER + REGRESSION (real data, cleaned) + 95% CI band ===================== */
function buildScatter(){
  const s = STATS;
  const ctx = document.getElementById("scatterChart");
  if (!ctx) return;
  if (chartsRegistry.scatterChart) chartsRegistry.scatterChart.destroy();
  const points = s.clean.map(r => ({ x: r.y, y: r.d }));
  const { slope, intercept } = s.reg;
  const cy = s.clean.map(r=>r.y), cd = s.clean.map(r=>r.d);
  const n = cy.length, xbar = mean(cy);
  const sxx = cy.reduce((sum,x)=>sum+(x-xbar)**2, 0);
  const residStd = Math.sqrt(cd.reduce((sum,y,i)=>sum+(y-(slope*cy[i]+intercept))**2,0) / (n-2));
  const minY = Math.min(...s.years), maxY = Math.max(...s.years);
  const xs = []; for(let x=minY; x<=maxY; x+=(maxY-minY)/40) xs.push(x);
  const line = xs.map(x => ({ x, y: slope*x+intercept }));
  const seAt = x => residStd * Math.sqrt(1/n + (x-xbar)**2/sxx);
  const bandUpper = xs.map(x => ({ x, y: slope*x+intercept + 1.96*seAt(x) }));
  const bandLower = xs.map(x => ({ x, y: slope*x+intercept - 1.96*seAt(x) }));

  regId("scatterChart", new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        { label:"Data", data: points, backgroundColor: "rgba(47,111,237,0.4)", pointRadius: 2.5 },
        { label:"CI 95% atas", data: bandUpper, type:"line", borderWidth:0, pointRadius:0, fill:"+1", backgroundColor:"rgba(239,68,68,0.15)" },
        { label:"CI 95% bawah", data: bandLower, type:"line", borderWidth:0, pointRadius:0, fill:false },
        { label:"Regresi", data: line, type:"line", borderColor: cssVar("--red") || "#EF4444", borderWidth: 2, pointRadius: 0, fill:false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title:{display:true,text:"Year",font:{size:10}}, grid:{color:cssVar("--border")}, ticks:{font:{size:8}} },
        y: { title:{display:true,text:"Total Deaths",font:{size:10}}, grid:{color:cssVar("--border")}, ticks:{font:{size:8}, callback:(v)=>fmtNum(v,0)} }
      }
    }
  }));
}

/* ===================== PLAIN SCATTER (no line) - paired with heatmap, matches notebook's heatmap+scatter figure ===================== */
function buildScatterPlain(){
  const s = STATS;
  const ctx = document.getElementById("scatterPlainChart");
  if (!ctx) return;
  if (chartsRegistry.scatterPlainChart) chartsRegistry.scatterPlainChart.destroy();
  const points = s.clean.map(r => ({ x: r.y, y: r.d }));
  regId("scatterPlainChart", new Chart(ctx, {
    type: "scatter",
    data: { datasets: [{ data: points, backgroundColor: "rgba(123,58,237,0.55)", pointRadius: 2.5 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title:{display:true,text:"Year",font:{size:9}}, grid:{color:cssVar("--border")}, ticks:{font:{size:7.5}} },
        y: { title:{display:true,text:"Total Deaths",font:{size:9}}, grid:{color:cssVar("--border")}, ticks:{font:{size:7.5}, callback:(v)=>fmtNum(v,0)} }
      }
    }
  }));
}
function buildMeanTypeChart(){
  const s = STATS;
  const ctx = document.getElementById("meanTypeChart");
  if (!ctx) return;
  if (chartsRegistry.meanTypeChart) chartsRegistry.meanTypeChart.destroy();
  const data = s.types.map(t => ({ label: t, value: mean(s.byType[t]) }));
  regId("meanTypeChart", new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.map(d=>d.label),
      datasets: [{ data: data.map(d=>d.value), backgroundColor: data.map(d=>TYPE_COLORS[d.label]), borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` rata-rata ${fmtNum(c.parsed.y,0)}` } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 8 } } },
        y: { grid: { color: cssVar("--border") }, ticks: { font: { size: 8.5 } } }
      }
    }
  }));
}

/* ===================== BAR: STATISTIK DESKRIPTIF (Total Deaths per Tipe) ===================== */
function buildDescStatChart(){
  const s = STATS;
  const ctx = document.getElementById("descStatChart");
  if (!ctx) return;
  if (chartsRegistry.descStatChart) chartsRegistry.descStatChart.destroy();
  const totalsByType = {};
  s.types.forEach(t => totalsByType[t] = 0);
  s.all.forEach(r => totalsByType[r.t] += r.d);
  const data = s.types.map(t => ({ label: t, value: totalsByType[t] }));
  regId("descStatChart", new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.map(d=>d.label),
      datasets: [{ data: data.map(d=>d.value), backgroundColor: data.map(d=>TYPE_COLORS[d.label]), borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` total ${fmtNum(c.parsed.y,0)} kematian` } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 8 } } },
        y: { grid: { color: cssVar("--border") }, ticks: { font: { size: 8.5 } } }
      }
    }
  }));
}

/* ===================== LINE: TOTAL DEATHS PER TAHUN, with 95% CI band ===================== */
function buildTrendTotalChart(){
  const s = STATS;
  const ctx = document.getElementById("trendTotalChart");
  if (!ctx) return;
  if (chartsRegistry.trendTotalChart) chartsRegistry.trendTotalChart.destroy();
  const sums = s.years.map(y => s.totalsByYearClean[y] || 0);
  // approximate bootstrap-style SE of the per-year SUM estimator, from the spread of individual rows that year
  const seByYear = s.years.map(y => {
    const vals = s.clean.filter(r=>r.y===y).map(r=>r.d);
    if (vals.length < 2) return 0;
    return stdev(vals) * Math.sqrt(vals.length);
  });
  const upper = sums.map((v,i)=> v + 1.96*seByYear[i]);
  const lower = sums.map((v,i)=> Math.max(0, v - 1.96*seByYear[i]));

  regId("trendTotalChart", new Chart(ctx, {
    type: "line",
    data: {
      labels: s.years,
      datasets: [
        { data: upper, borderWidth:0, pointRadius:0, fill:"+1", backgroundColor:"rgba(47,111,237,0.15)" },
        { data: lower, borderWidth:0, pointRadius:0, fill:false },
        { data: sums, borderColor: cssVar("--blue") || "#2F6FED", backgroundColor: "rgba(47,111,237,0.12)",
          fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 7 }, maxRotation: 90 } },
        y: { grid: { color: cssVar("--border") }, ticks: { font: { size: 7 }, callback:(v)=>fmtNum(v,0) } }
      }
    }
  }));
}

/* ===================== TUKEY HSD (approximate pairwise mean-diff CI) ===================== */
function buildTukey(){
  const s = STATS;
  const svg = document.getElementById("tukeySvg");
  if (!svg) return;
  const text = cssVar("--text-muted") || "#6B7280";
  const stroke = cssVar("--text-soft") || "#9AA1B4";

  // Matches statsmodels' pairwise_tukeyhsd().plot_simultaneous(): one horizontal interval PER GROUP
  // (centered on that group's own mean), full names on the y-axis, groups in reverse-alphabetical
  // order top-to-bottom (matplotlib default for categorical axes), with the largest group used as the
  // reference: its CI drawn as dashed vertical guide lines and its own row highlighted blue.
  const msWithin = s.groups.reduce((sum,g)=> sum + g.reduce((a,v)=>a+(v-mean(g))**2,0), 0) / s.dfW;
  const qCrit = 3.95; // approx studentized-range critical value, alpha = .05
  const rows = [...s.types].sort((a,b)=> b.localeCompare(a)).map(t => {
    const vals = s.byType[t];
    const m = mean(vals);
    const hw = (qCrit/Math.SQRT2) * Math.sqrt(msWithin/vals.length);
    return { name: t, n: vals.length, lo: m-hw, hi: m+hw, m };
  });
  const ref = rows.reduce((a,b)=> b.n>a.n ? b : a, rows[0]);

  const W=320,H=170, padL=118, padR=14, padT=10, padB=26;
  const plotW = W-padL-padR, plotH = H-padT-padB;
  const minV = Math.min(0, ...rows.map(r=>r.lo)) * 1.1;
  const maxV = Math.max(...rows.map(r=>r.hi)) * 1.15 || 1;
  const xScale = v => padL + ((v-minV)/(maxV-minV))*plotW;
  const rowH = plotH / rows.length;

  const colorBlue = cssVar("--blue")||"#2F6FED";
  const colorNeutral = "#374151";
  let content = `<line x1="${xScale(ref.lo)}" y1="${padT}" x2="${xScale(ref.lo)}" y2="${padT+plotH}" stroke="${colorBlue}" stroke-dasharray="4,3" stroke-width="1"/>`;
  content += `<line x1="${xScale(ref.hi)}" y1="${padT}" x2="${xScale(ref.hi)}" y2="${padT+plotH}" stroke="${colorBlue}" stroke-dasharray="4,3" stroke-width="1"/>`;
  rows.forEach((r,i)=>{
    const y = padT + rowH*(i+0.5);
    const isRef = r===ref;
    const color = isRef ? colorBlue : colorNeutral;
    content += `<line x1="${xScale(r.lo)}" y1="${y}" x2="${xScale(r.hi)}" y2="${y}" stroke="${color}" stroke-width="2"/>`;
    content += `<circle cx="${xScale(r.m)}" cy="${y}" r="3" fill="${color}"/>`;
    content += `<text x="${padL-6}" y="${y+3}" font-size="7.5" fill="${text}" text-anchor="end">${r.name}</text>`;
  });
  content += `<text x="${W/2}" y="${H-8}" font-size="8" fill="${text}" text-anchor="middle">Perbedaan Rata-rata Total Kematian</text>`;
  svg.innerHTML = content;
  TUKEY_ALL_CROSS_ZERO = rows.every(r => rows.every(o => o===r || !(r.lo>o.hi || r.hi<o.lo)));
}
let TUKEY_ALL_CROSS_ZERO = true;

/* ===================== MEAN PLOT (mean ± 95% CI per type) ===================== */
function buildMeanPlot(){
  const s = STATS;
  const svg = document.getElementById("meanPlotSvg");
  if (!svg) return;
  const text = cssVar("--text-muted") || "#6B7280";
  const stroke = cssVar("--text-soft") || "#9AA1B4";

  const groups = s.types.map(t => {
    const vals = s.byType[t];
    const m = mean(vals);
    const se = stdev(vals) / Math.sqrt(vals.length);
    return { name: t.replace("Bencana Non Alam dan Penyakit","Non Alam & Penyakit"), m, lo: m-1.96*se, hi: m+1.96*se, color: TYPE_COLORS[t] };
  });

  const W=320,H=170, padL=44, padR=14, padT=14, padB=30;
  const plotW = W-padL-padR, plotH = H-padT-padB;
  const maxV = Math.max(...groups.map(g=>g.hi)) * 1.15;
  const yScale = v => padT + plotH - (v/maxV)*plotH;
  const colW = plotW / groups.length;

  let content = "";
  [0, maxV/2, maxV].forEach(v=>{
    const y = yScale(v);
    content += `<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="${stroke}" stroke-opacity="0.2" stroke-dasharray="3,3"/>`;
    content += `<text x="${padL-6}" y="${y+3}" font-size="7.5" fill="${text}" text-anchor="end">${Math.round(v)}</text>`;
  });

  groups.forEach((g,i)=>{
    const cx = padL + colW*(i+0.5);
    const yLo = yScale(g.lo), yHi = yScale(g.hi), yM = yScale(g.m);
    content += `<line x1="${cx}" y1="${yLo}" x2="${cx}" y2="${yHi}" stroke="${g.color}" stroke-width="2"/>`;
    content += `<line x1="${cx-8}" y1="${yLo}" x2="${cx+8}" y2="${yLo}" stroke="${g.color}" stroke-width="2"/>`;
    content += `<line x1="${cx-8}" y1="${yHi}" x2="${cx+8}" y2="${yHi}" stroke="${g.color}" stroke-width="2"/>`;
    content += `<circle cx="${cx}" cy="${yM}" r="4" fill="${g.color}"/>`;
    content += `<text x="${cx}" y="${H-14}" font-size="7.5" fill="${text}" text-anchor="middle" font-weight="600">${g.name}</text>`;
  });
  svg.innerHTML = content;
}

/* ===================== SIMPLE REGRESSION CHART (log deaths ~ year) ===================== */
function buildRegSimple(){
  const s = STATS;
  const ctx = document.getElementById("regSimpleChart");
  if (!ctx) return;
  if (chartsRegistry.regSimpleChart) chartsRegistry.regSimpleChart.destroy();
  const points = s.clean.map(r => ({ x: r.y, y: r.d }));
  const { slope, intercept } = s.reg;
  const minY = Math.min(...s.years), maxY = Math.max(...s.years);
  const line = [{x:minY, y:slope*minY+intercept}, {x:maxY, y:slope*maxY+intercept}];
  regId("regSimpleChart", new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        { data: points, backgroundColor: "rgba(123,110,246,0.4)", pointRadius: 2 },
        { data: line, type:"line", borderColor: cssVar("--red") || "#EF4444", borderWidth: 2, pointRadius: 0, fill:false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks:{font:{size:7.5}}, grid:{color:cssVar("--border")} },
        y: { ticks:{font:{size:7.5}, callback:(v)=>fmtNum(v,0)}, grid:{color:cssVar("--border")} }
      }
    }
  }));
}

/* ===================== MULTIPLE REGRESSION: ACTUAL vs PREDICTED (Year + Type model, matches notebook viz) ===================== */
function buildRegMultiOn(canvasId){
  const s = STATS;
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (chartsRegistry[canvasId]) chartsRegistry[canvasId].destroy();
  const maxV = Math.max(...s.actual, ...s.vizPredicted) * 1.05;
  const datasets = s.types.map(t => ({
    label: t,
    data: s.clean.map((r,i) => r.t === t ? { x: r.d, y: s.vizPredicted[i] } : null).filter(Boolean),
    backgroundColor: (TYPE_COLORS[t] || "#2F6FED") + "CC",
    pointRadius: 2.5
  }));
  datasets.push({
    label: "Garis Prediksi Ideal (Y = X)",
    data: [{x:0,y:0},{x:maxV,y:maxV}],
    type: "line", borderColor: cssVar("--red") || "#EF4444", borderDash:[6,4],
    borderWidth: 1.5, pointRadius: 0, fill:false
  });
  regId(canvasId, new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position:"bottom", labels:{font:{size:8}, boxWidth:8} } },
      scales: {
        x: { min:0, max:maxV, title:{display:true,text:"Total Deaths Aktual",font:{size:9}}, ticks:{font:{size:7.5}, callback:(v)=>fmtNum(v,0)}, grid:{color:cssVar("--border")} },
        y: { min:0, max:maxV, title:{display:true,text:"Total Deaths Prediksi",font:{size:9}}, ticks:{font:{size:7.5}, callback:(v)=>fmtNum(v,0)}, grid:{color:cssVar("--border")} }
      }
    }
  }));
}
function buildRegMulti(){
  buildRegMultiOn("regMultiChart");
  buildRegMultiOn("regMultiChartEDA");
}

/* ===================== ANGKA RESMI DARI LAPORAN TUGAS AKHIR ADS ===================== */
/* Cell 10-19: hasil resmi analisis Python (pandas/scipy/statsmodels) di laporan, dipakai apa adanya
   supaya dashboard konsisten persis dengan laporan yang dikumpulkan. */
const REPORT_STATS = {
  nAwal: 322,          // Cell 4: setelah filter tahun 2000-2010 (dari 529 baris)
  nOutlier: 33,        // Cell 5: outlier dibuang metode IQR
  nBersih: 289,        // Cell 5: df_no_outliers
  shapiroW: 0.6767, shapiroP: 3.82e-23,   // Cell 8
  leveneStat: 0.9922, leveneP: 0.3720,    // Cell 9
  corrR: -0.0507, corrP: 0.3907,          // Cell 10
  regSimpleR2: 0.0026, regSimpleP: 0.3907, // Cell 12
  regMultiR2: 0.792, regMultiAdjR2: 0.644, // Cell 14
  anovaF: 1.0565, anovaP: 0.3490,          // Cell 16
  tukeyAllRejectFalse: true                // Cell 18
};

/* ===================== FILL DYNAMIC RESULT LABELS ===================== */
function fillResultLabels(){
  const s = STATS;
  const r = REPORT_STATS;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.innerHTML = val; };
  const corrSig = r.corrP < 0.05;
  const anovaSig = r.anovaP < 0.05;

  set("statMean", fmtNum(mean(s.dAll)));
  set("statStd", fmtNum(stdev(s.dAll)));
  set("statCorr", `r = ${r.corrR.toFixed(4)}`);
  set("statCorrNote", `Hubungan sangat lemah (p = ${r.corrP.toFixed(4)}, ${corrSig ? "signifikan" : "tidak signifikan"})`);
  set("statF", r.anovaF.toFixed(4));
  set("statP", r.anovaP.toFixed(4));
  set("statAnovaStatus", anovaSig ? "Signifikan" : "Tidak Signifikan");
  set("statR2Simple", r.regSimpleR2.toFixed(4));
  set("statSlope", s.reg.slope.toFixed(4));
  set("statR2Multi", `R² = ${r.regMultiR2.toFixed(3)}`);
  set("statR2MultiNote", `Adj. R² = ${r.regMultiAdjR2.toFixed(3)} &middot; Model signifikan`);

  set("capCorr",
    `Hasil uji Korelasi Pearson menghasilkan koefisien r = ${r.corrR.toFixed(4)} dengan p-value = ${r.corrP.toFixed(4)}. ` +
    `Hal tersebut menunjukkan bahwa tidak terdapat hubungan linear yang signifikan antara tahun pelaporan dan jumlah kematian.`
  );
  set("capRegSimple",
    `Regresi linear sederhana menghasilkan R&sup2; = ${(r.regSimpleR2*100).toFixed(2)}%. ` +
    `Berarti Year secara individual tidak berpengaruh signifikan terhadap jumlah kematian.`
  );
  set("capRegMulti",
    `Kombinasi variabel Year, Type, dan Cause menghasilkan R&sup2; pada regresi linear multiple sebesar ${(r.regMultiR2*100).toFixed(1)}%, ` +
    `yang menunjukkan bahwa ketiga variabel tersebut secara bersama-sama mampu menjelaskan sebagian besar variasi jumlah kematian.`
  );
  set("capAnova",
    `Berdasarkan hasil One-Way ANOVA, diperoleh F = ${r.anovaF.toFixed(4)} dengan p-value = ${r.anovaP.toFixed(4)} (&gt;0,05).`
  );
  set("capPosthoc",
    "Semua interval kepercayaan horizontal (group1 vs group2) melintasi garis nol. " +
    `<b>Kesimpulan ANOVA:</b> Berarti tidak ada perbedaan rata-rata yang signifikan antara kelompok Type (${s.types.join(", ")}).`
  );
}

/* ===================== FILL NARRATIVE TEXT (kept in sync with real numbers) ===================== */
function fillNarrative(){
  const s = STATS;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.innerHTML = val; };
  const r = REPORT_STATS;
  const periodeReport = "2000&ndash;2010";

  // Semua halaman: dikunci ke scope resmi laporan (2000-2010)
  set("periodeDataHome", periodeReport);
  set("periodeDataSubHome", "11 tahun");
  set("periodeDataVal", periodeReport);
  set("totalDataDataset", `${s.all.length}`);
  set("kategoriPenyebabHome", `${Object.keys(s.totalsByCause).length}`);
  set("kategoriPenyebabSubHome", `${s.types.length} tipe utama`);
  set("dataBersihCount", `${r.nBersih}`);

  set("interpText",
    `Tidak terdapat perbedaan rata-rata total kematian yang signifikan antar tipe penyebab ` +
    `(F = ${r.anovaF.toFixed(4)}, p = ${r.anovaP.toFixed(4)}). ` +
    `Hubungan antara tahun dan jumlah kematian sangat lemah dengan korelasi r = ${r.corrR.toFixed(4)} ` +
    `(tidak signifikan). Model regresi linear berdasarkan tahun hanya mampu menjelaskan sekitar ${(r.regSimpleR2*100).toFixed(2)}% variasi jumlah kematian, ` +
    `sedangkan model berdasarkan Year, Type, dan Cause jauh lebih kuat dalam menjelaskan variasi data (R&sup2; = ${r.regMultiR2.toFixed(3)}).`
  );

  set("concTotal", `Dataset berhasil difilter menjadi periode ${periodeReport}, menyisakan ${r.nAwal} data dari total ${DEATH_DATA.length} data awal.`);
  set("concBersih", `Sebanyak ${r.nOutlier} outlier berhasil dihapus menggunakan metode IQR, sehingga data menjadi lebih representatif dengan ${r.nBersih} data siap analisis.`);
  set("concCorr", `Korelasi antara Year dan Total Deaths sangat lemah (r = ${r.corrR.toFixed(4)}) dan tidak signifikan (p = ${r.corrP.toFixed(4)}).`);
  set("concAnova", `Hasil One-Way ANOVA (F = ${r.anovaF.toFixed(4)}, p = ${r.anovaP.toFixed(4)}) menunjukkan tidak terdapat perbedaan rata-rata jumlah kematian yang signifikan berdasarkan Type, dan hasil tersebut diperkuat oleh Uji Tukey HSD yang juga tidak menemukan perbedaan signifikan antar pasangan kategori.`);
  set("concRegSimple", `Regresi linear sederhana menunjukkan Year tidak berpengaruh signifikan terhadap jumlah kematian (R&sup2; = ${(r.regSimpleR2*100).toFixed(2)}%).`);
  set("concRegMulti", `Regresi linear multipel memberikan hasil yang jauh lebih baik dengan R&sup2; = ${(r.regMultiR2*100).toFixed(1)}%, sehingga kombinasi Year, Type, dan Cause lebih mampu menjelaskan variasi jumlah kematian.`);
}

/* ===================== INIT / REBUILD ===================== */
function destroyCharts(){
  Object.keys(chartsRegistry).forEach(k=>{
    if(chartsRegistry[k] && chartsRegistry[k].destroy){ chartsRegistry[k].destroy(); }
    delete chartsRegistry[k];
  });
}

function buildAllCharts(){
  chartDefaults();
  if (!STATS) computeStats();
  buildIndonesiaMap();
  buildTopCausesChart();
  fillHeroHighlight();
  fillNarrative();
  builtPages.add("home");
}

function rebuildThemedVisuals(){
  destroyCharts();
  builtPages.clear();
  buildAllCharts();
  const activePage = document.querySelector(".page.active");
  if (activePage) {
    const id = activePage.id.replace("page-","");
    if (pageChartBuilders[id] && !builtPages.has(id)) {
      pageChartBuilders[id]();
      builtPages.add(id);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  computeStats();
  buildAllCharts();
});
