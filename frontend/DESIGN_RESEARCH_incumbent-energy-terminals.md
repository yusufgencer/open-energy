# OpenEnergy Design Convergence Report
### Aligning a light-theme shadcn/ui dashboard with incumbent energy forecasting & trading terminals

**Scope:** Montel, Meteologica (xTraders), enercast, plus adjacent terminals (Volue Insight, Energy Quantified, EPEX SPOT, Nord Pool, Bloomberg). Researched June 2026 from product pages, support docs, shipped front-end source, screenshots, and design-standard references. Every section cites URLs.

---

## 0. The single most important finding (read this first)

There are **two distinct UX archetypes** in this sector, and they look opposite:

| Archetype | Examples | Theme | Layout | Density |
|---|---|---|---|---|
| **Trading terminal** | Montel Online/Markets, EnAppSys, Bloomberg, EPEX/Nord Pool ISV desks | **Dark**, near-black | Multi-panel floating windows, topbar commodity tabs | Very high (18–20px rows) |
| **Forecasting workbench** | **Meteologica xTraders, enercast portal** | **Light** | Top bar + left sidebar + central chart, table-centric | High but not extreme |

**OpenEnergy is a forecasting app, not a trading terminal.** Your two closest incumbents — Meteologica and enercast — are *both light-themed*. Meteologica's shipped CSS is literally a light Material Design app (`#e5e5e5` canvas, `#1565c0` blue primary); enercast's portal is "almost certainly light" with a chart-first "Workbench." Only the trading terminals (Montel, Bloomberg) are dark.

**Conclusion:** Your light-theme instinct is *correct and on-brand for forecasting*. Do not chase Bloomberg darkness. Instead, borrow the terminals' **density, numeric discipline, and chart conventions** and apply them inside a clean light shell — which is exactly what Meteologica and enercast already do. The rest of this report is built on that thesis.

Sources: [Meteologica management-tool CSS](https://xtraders.meteologica.com/management-tool/assets/index-BTuxgB5e.css) · [enercast portal demo](https://www.enercast.de/enercast-portal-demo/) · [Montel Online support](https://support.montel.energy/what-is-montel-online) · [CRD: why dark matters](https://www.crd.com/insights/importance-of-themes-and-why-dark-matters)

---

## 1. Montel — the dark trading terminal

Montel's marketing site (montel.energy) is light SaaS, but the **product is dark-theme only** (~`#0d1117`–`#121212`). Confirmed from screenshots.

- **Navigation:** Horizontal **topbar with commodity tabs** — `News | Power | Gas | LNG | Coal | Oil | Green | Markets | FinEq | EnAppSys | EQ | My pages`. NOT a left sidebar. A `My pages` tab holds saved personal workspaces. Closer to Reuters Eikon than to Bloomberg's command line.
- **Layout:** Multi-panel **draggable/resizable floating windows** on a dark canvas. Each window header has Settings / Pop-out (2nd screen) / Collapse / Close. Layouts persist via "Save View." A "Personal Page" lets users choose 1/2/3-column frames of price tables or charts.
- **Default dashboard panels:** left = full-height news feed; center = large candlestick chart; right = stacked compact price tables (EEX, Nordic, ICE Crude, EUA).
- **Accent color:** **teal/cyan** for interactive elements (filter pills, Add buttons, alert bells); orange/amber for breaking-news strip and "recently updated" cell highlights.
- **Data density:** Bloomberg-class. EnAppSys intraday tables show 24 hourly rows × ~13 columns (Product, Bid, Size, Ask, Size, Open, High, Low, Last, Chg%, VL, VA, Time), ~18–20px row height, red for down values, amber row-flash on update.
- **Charts:** full candlestick (OHLC) with MA overlay and OHLCV readout top-right; intraday line charts with a **mini range-selector strip** at the bottom for zoom; multi-series + stacked-bar combos. Technical studies: MA, EMA, MACD, RSI, Stochastics, Bollinger, Elder Impulse. Chart toolbar: `1 Day | Contract | Help | Compare | Export | Chart | Studies`.
- **News is native**, not bolted on: full-height feed panel, date-grouped headlines with right-aligned timestamps, a "LATEST" breaking strip pinned to the top, source filter chips (Montel | E&M).
- **Modern touches over Bloomberg:** rounded cards, pill buttons, country-flag icons, sans-serif (not monospace green-on-black), drag-drop windows, **card-grid "Explore dashboards" picker** (3-col, commodity badge + flag + name + Preview/Add).
- **Bloomberg-likeness: ~7–8/10** on density, higher than Bloomberg on visual polish.

Sources: [Montel Online](https://montel.energy/platforms/montel-online) · [personalize views](https://support.montel.energy/how-to-personalize-my-views-in-montel-online) · [studies](https://support.montel.energy/how-to-add-and-remove-studies) · [EnAppSys](https://montel.energy/platforms/enappsys) · [insightcommodity: Montel Online](https://insightcommodity.com/montel-online)

---

## 2. Meteologica (xTraders) — the light Material forecasting app

Evidence is unusually solid here: the **Management tool** ships its Vite + Lit source publicly, so these are real design tokens, not guesses.

- **Theme:** **Light by default** — `#e5e5e5` body canvas, `#f5f5f5`/`#fff` surfaces. Full dark-theme token set exists (`#061c36` navy surface) but light is the shipped default.
- **Brand palette (Material Design):** primary blue **`#1565c0`** (+ light blue `#039be5`), accent magenta/pink **`#d81b60`**, dark-slate tooltips `#212f3c`.
- **Typography:** **Roboto** 400/500/700, with **`html { font-size: 10px }`** root — a deliberate compact-density trick (so `1.4rem` = 14px). No dedicated monospace; numbers render in Roboto.
- **Shell:** persistent **top bar** (`mt-header`: title, config, help, logout, **apps-launcher**) + persistent **left sidebar** (`mt-sidemenu`: assets, groups, external-access, users) + central routed pages. Table/list-centric, not a draggable-widget canvas.
- **Multi-app launcher:** Google-style grid popover top-right switches between three sibling apps — **xTraders**, **Data manager**, **Management tool**.
- **Asset model:** assets (wind/solar *parques*) → **clusters/groups** → portfolios. Surfaced as **sortable, searchable, multi-select tables** (columns: name, country `pais`, zone, max power `pot_max`, tags, used-by) with inline add/edit/delete/info row actions — not a literal tree widget. Both per-asset and per-portfolio are first-class.
- **Density:** enterprise data-grid — sort/search/multi-select, copy-to-clipboard, **PowerPoint export** icon.
- **Forecasts (marketing copy only — charts are login-gated):** "multi-model, ensemble and probabilistic," "14-day ensemble scenarios quantifying uncertainty," hourly resolution, updated 4×/day. Implies uncertainty-fan line charts, but exact styling is **not publicly observable**.

Sources: [shipped CSS tokens](https://xtraders.meteologica.com/management-tool/assets/index-BTuxgB5e.css) · [management tool](https://xtraders.meteologica.com/management-tool/) · [services](https://www.meteologica.com/services.html)

---

## 3. enercast — the light "Workbench" forecasting portal

Portal at `portal.enercast.de` (hash-routed SPA). Almost certainly light theme (no dark mode found; white logo only on the dark marketing footer).

- **Signature layout = the "Workbench"**, a confirmed three-zone structure (from a release screenshot caption):
  - **Left panel:** list of forecast **models** to toggle/compare.
  - **Center:** large **main chart** where models overlay for instant comparison ("instant backcasting").
  - **Bottom:** a **"quality widget"** — RMSE / MAE / Bias(nBIAS) as a **table**, recalculating in seconds when you switch models or the evaluation window.
- **Chart-first**, table-subordinate. Chart types: forecast-vs-actual time-series overlay, a **forecast-error curve** on the same axis (zoom into anomalies), daily-mean chart for systematic errors, **RMSE histogram**, and **percentile/uncertainty bands** (added 2025.02). Dual data streams (forecast + metered) shown together.
- **Asset nav:** per-site assets grouped into **clusters** for portfolios; regulations assignable per-asset or per-cluster. Exact switcher mechanism not publicly documented.
- **Horizons:** 15-min intraday → day-ahead → up to 31 days; wind refreshed every 15 min, solar 4×/day. Analyses break down "by forecast horizon (intraday vs day-ahead)."
- **Brand colors:** not publicly documented (portal gated). Chart series colors for wind/solar/actual/forecast are behind login — **unknown**, infer from sector norms (§5).

Sources: [portal demo](https://www.enercast.de/enercast-portal-demo/) · [2022.02 Workbench screenshot](https://www.enercast.de/magazine/enercast202202/) · [interactive quality eval](https://www.enercast.de/magazine/interactive-forecast-quality-evaluation-in-the-enercast-portal/) · [2025.02 percentiles](https://www.enercast.de/magazine/enercast-2025-02-optimized-feed-in-forecasts-for-solar-assets-with-self-consumption/)

---

## 4. Adjacent platforms (quick signal)

- **Volue Insight** (ex-Wattsight): light marketing brand (navy/blue), app gated. Three access modes (web app, **Excel plug-in**, REST/Python) — a near-universal sector pattern. Technical-analysis module offers TradingView-style candlestick + MA/Bollinger/MACD/RSI. Data model = TIME_SERIES + TAGGED/INSTANCE curves (forecast ensembles). [volue.com/data-and-forecasts](https://volue.com/data-and-forecasts)
- **Energy Quantified** (now Montel-owned): **dark-first** marketing brand. EQ 2.0 introduced **map-as-navigation** (click a European bidding zone to drill in) + Google-style search + preset filters. Role-coded icon colors (trader=purple, analyst=green, etc.). Web + Excel + Python + API. [EQ 2.0 launch](https://www.energyquantified.com/blog/energy-quantified-2-0-launched)
- **EPEX SPOT / Nord Pool:** exchanges, not SaaS. Public results pages use a **three-view toggle: data table | data graph | aggregated curves**, plus a **clickable European bidding-zone map** and a **prominent time-zone selector** in the nav. 83%+ of EPEX orders are automated; the real trader "UX" is dark ISV desktops. [Nord Pool portal](https://www.nordpoolgroup.com/en/services/power-market-data-services/dataportal-launch-update/) · [EPEX market results](https://www.epexspot.com/en/market-results)
- **Kpler power** (current best-practice reference): light/neutral panels, **interactive plant-level map**, bidding-zone toggle, supply-stack viz, side-by-side price/demand grids. [kpler.com/market/power-market](https://www.kpler.com/market/power-market)

---

## 5. Shared conventions sector users expect (the "muscle memory")

These cross every platform and are what makes an energy app feel native to the sector:

**(a) Information density** — terminals run **1.5–2× denser** than mainstream SaaS. Padding lives in the **4 / 8 / 12px** band (never 16–24px). Row-height tiers: HD trading ≤28px, MD analyst ~32px, LD portal ≥44px. [Designing for data density](https://paulwallas.medium.com/designing-for-data-density-what-most-ui-tutorials-wont-teach-you-091b3e9b51f4) · [Institutional finance design system](https://edwson.com/design-system-showcase.html)

**(b) Typography** — **tabular figures are mandatory.** `font-variant-numeric: tabular-nums slashed-zero` on every numeric column, right-aligned. UI text 14px, captions 12px. Full monospace is optional flavor; the clean hybrid is **proportional sans for labels + tabular/mono numerals** (e.g., JetBrains Mono just for numbers). [Tabular numbers in CSS](https://dev.to/alanwest/tabular-numbers-in-css-font-variant-numeric-vs-monospace-hacks-25cn)

**(c) Time-series conventions** — time on X always; **dashed vertical "Now" line** splitting solid actuals (left) from dashed forecast (right); **range-selector button row `1D 1W 1M 3M 6M 1Y All`** top-right of each chart; gridlines very light (`#D9D9D9` 1px, baseline `#B3B3B3` 1.5px); soft Y min/max so spikes overflow. [ONS chart elements](https://service-manual.ons.gov.uk/data-visualisation/build-specifications/chart-elements) · [Highcharts range selector](https://www.highcharts.com/docs/stock/range-selector)

**(d) Color semantics** — **green = above/gain, red = below/loss, gray = neutral**, ALWAYS paired with arrow/icon (8% red-green CVD; Bloomberg's own research). **Forecast confidence band = shaded fill, 20–30% opacity, funnel-widening with horizon.** Energy series de-facto standard: **solar = orange/amber `#ff9800`, wind = blue** (ERCOT, Home Assistant). [ERCOT wind+solar](https://www.ercot.com/gridmktinfo/dashboards/combinedwindandsolar) · [Bloomberg color accessibility](https://www.bloomberg.com/company/stories/designing-the-terminal-for-color-accessibility/) · [HA energy colours](https://community.home-assistant.io/t/energy-panel-colours/368175)

**(e) Dark vs light** — dark is the **trading-desk** norm (all-day, low-light control rooms); **light is the forecasting/analytics norm** (Meteologica, enercast, Volue marketing, Nord Pool). Your light default is correct; a dark toggle is a nice-to-have, not required.

**(f) Navigation** — forecasting apps use **top bar + persistent left sidebar with a portfolio→site→device hierarchy**; trading terminals use topbar commodity tabs + saved workspaces. Map-as-navigation is a rising pattern in EU power (bidding zones). Excel/CSV export and time-zone selector are table stakes.

**(g) Dense numerics** — `tabular-nums`, right-align, sticky header, both gridlines, optional zebra, `white-space: nowrap`, virtualize >1k rows, persisted density toggle.

---

## 6. Concrete recommendations for OpenEnergy (light shadcn/ui)

### 6.1 Verdict table — adopt / skip

| Convention | Verdict | Why |
|---|---|---|
| **Light theme as default** | ✅ Keep | Matches Meteologica + enercast (your real peers) |
| Dark theme as **option** | ➕ Optional later | `next-themes` toggle; not a priority |
| **enercast "Workbench" 3-zone layout** | ✅ **Adopt — your hero screen** | Left model/asset panel · center chart · bottom quality (RMSE/MAE/Bias) widget |
| **Top bar + left sidebar w/ asset tree** | ✅ Adopt | portfolio → site → turbine/inverter; both Meteologica & enercast do this |
| **Dense tables (28–32px rows, tabular nums)** | ✅ Adopt | Single biggest "feels like the sector" lever |
| **Range selector `1D/1W/1M/3M/1Y/All`** on charts | ✅ Adopt | Universal; muscle memory |
| **Dashed "Now" line + solid-actual/dashed-forecast + 20–30% band** | ✅ Adopt | The defining forecast-chart idiom |
| **solar = amber, wind = blue; green/red = above/below target w/ arrows** | ✅ Adopt | ERCOT/HA standard; instantly legible to sector users |
| **KPI strip** (capacity factor, MW now, day-ahead nMAE) | ✅ Adopt | Bloomberg "big number top, muted secondary below" |
| **CSV/Excel + PPT export, last-updated badge** | ✅ Adopt | Table stakes; Meteologica ships PPT export |
| **Map-as-navigation** (asset/bidding-zone map) | ➕ Phase 2 | EQ/Kpler pattern; valuable once you have many geo-distributed assets |
| Bloomberg **multi-panel floating draggable windows** | ❌ Skip | High build cost, low payoff for a focused forecasting tool; enercast/Meteologica use fixed layouts |
| **Dark near-black terminal aesthetic** | ❌ Skip as default | Wrong archetype for forecasting; would diverge from your peers |
| Candlestick/OHLC + RSI/MACD technical studies | ❌ Skip | That's price-trading, not power forecasting |
| Topbar **commodity tabs** (Montel) | ❌ Skip | You're single-domain (wind/solar); sidebar asset tree fits better |
| Full **monospace** UI | ⚠️ Numerals only | Mono whole UI hurts label readability; apply to numbers only |

### 6.2 Design tokens (light theme)

```css
/* Surfaces — slightly cooler/greyer than default shadcn white, à la Meteologica #f5f5f5 */
--background: 0 0% 100%;
--card:       0 0% 100%;
--muted:      210 20% 97%;      /* page canvas / table header fill */
--border:     214 15% 88%;      /* light hairline, ~ #D9D9D9 chart gridline family */

/* Brand accent — pick ONE primary. Two strong, sector-resonant options: */
/* Option A (Meteologica-aligned, trustworthy blue):  #1565c0  -> 211 80% 42% */
/* Option B (Montel-aligned, distinctive teal):        #0e7490  -> 192 82% 31% */
--primary: 211 80% 42%;         /* recommend Option A blue as base */

/* Domain series (use for chart-1..5) */
--solar:   33 100% 50%;   /* #ff9800 amber  */
--wind:    211 80% 50%;   /* blue           */
--actual:  215 16% 35%;   /* dark slate, solid line */
--forecast:33 100% 50%;   /* amber dashed + 25% band */
--pos:     142 64% 38%;   /* above target / green */
--neg:     0 72% 48%;     /* below target / red   */

/* shadcn chart vars */
--chart-1: var(--wind);
--chart-2: var(--solar);
--chart-3: 215 16% 35%;   /* actual   */
--chart-4: 161 60% 40%;   /* combined/teal */
--chart-5: 280 50% 55%;   /* hydro/other purple */
```

### 6.3 Typography

- **UI font:** Inter or IBM Plex Sans (you likely already have one). 14px base (`text-sm`), 12px captions (`text-xs`).
- **Numerals:** add **JetBrains Mono** or just `tabular-nums slashed-zero` on numeric cells. Canonical utility:
  ```css
  .num { font-variant-numeric: tabular-nums slashed-zero; font-feature-settings:"tnum","zero"; }
  ```
  Pair every numeric column/KPI with `text-right` (tables) or `tabular-nums` (KPI tiles).
- Optional density trick à la Meteologica: tighter type scale, but Tailwind `text-sm/text-xs` already gets you there without a 10px root.

### 6.4 Table density (the highest-impact change)

Dense row recipe (shadcn `Table` + TanStack Table):
```
TableRow:  h-8                       /* 32px MD; h-7 = 28px HD */
TableCell: py-1 px-3 text-sm tabular-nums   /* numeric: + text-right */
TableHead: sticky top-0 bg-muted h-8 text-xs font-medium
```
- Sticky header, both horizontal+vertical hairline gridlines (`border` on cells), `whitespace-nowrap`, optional zebra (`even:bg-muted/40`).
- Virtualize past ~1,000 rows (TanStack Virtual).
- Ship a **Compact / Comfortable** density toggle (`ButtonGroup size="sm"`), persisted to localStorage.

### 6.5 Chart config (shadcn `Chart` → Recharts `ComposedChart`)

```tsx
// Actual (solid) + Forecast (dashed) + confidence band (Area) + Now line
<ComposedChart data={series}>
  <CartesianGrid stroke="#D9D9D9" strokeWidth={1} vertical={false} />
  <XAxis dataKey="t" type="number" scale="time" tick={{ fontSize: 11 }} />
  <YAxis tick={{ fontSize: 11 }} width={44} />          {/* soft min/max, not forced 0 */}
  <Area dataKey="band" fill="var(--color-forecast)" fillOpacity={0.22} stroke="none" />
  <Line dataKey="actual"   stroke="var(--color-actual)"   dot={false} strokeWidth={1.75} />
  <Line dataKey="forecast" stroke="var(--color-forecast)" dot={false} strokeDasharray="4 3" />
  <ReferenceLine x={nowTs} stroke="#999" strokeDasharray="3 3" label="Now" />
</ComposedChart>
```
- **Range selector** above-right via shadcn `ToggleGroup` (single-select): `1D 1W 1M 3M 1Y All`.
- Confidence band **funnels wider with horizon**; for P10/P50/P90 stack two Areas at increasing opacity.
- If you later need heavy time-series, built-in `markLine`/`dataZoom`, or `type:'time'` auto-parsing → switch that chart to **Apache ECharts** (common in grid/SCADA). Recharts-via-shadcn is the fast path for now.

### 6.6 shadcn component map

| OpenEnergy element | Component |
|---|---|
| Left asset tree (portfolio→site→device) | `Sidebar` + `Collapsible`, active via `usePathname`, ARIA tree keys, state persisted |
| Top KPI strip (MW now, capacity factor, nMAE, next-update) | `Card` + `text-2xl tabular-nums` + arrow `Badge` |
| Forecast-vs-actual chart | `Chart` (Recharts `ComposedChart`) + `ToggleGroup` range selector |
| Quality widget (RMSE/MAE/Bias) — enercast pattern | small `Table`, sticky, recompute on model/window change |
| Forecast/asset grid | `Table` + TanStack Table (sort/filter/virtualize) + density `ButtonGroup` |
| Asset/model/horizon filters | `Select` / `Combobox` / status `Badge` pills |
| Drill-down detail | `Sheet` (drawer — preserves context, don't navigate away) |
| Last-updated / data freshness | `Badge` + `Tooltip` |
| Theme toggle (optional dark) | `next-themes` `ThemeProvider attribute="class"` |

### 6.7 The recommended hero layout (enercast Workbench, modernized for shadcn light)

```
┌───────────────────────────────────────────────────────────────┐
│ Topbar: logo · global search · TZ selector · export · user     │
├──────────┬────────────────────────────────────────────────────┤
│ Sidebar  │ KPI strip:  [MW now ↑] [Cap.factor] [Day-ahead nMAE]│
│ asset    ├────────────────────────────────────────────────────┤
│ tree     │  Main chart  (actual solid · forecast dashed ·      │
│ portfolio│   25% band · Now line)        [1D 1W 1M 3M 1Y All]  │
│ ├ site A │                                                     │
│ │ ├ WTG1 ├────────────────────────────────────────────────────┤
│ │ └ WTG2 │  Quality widget (table): RMSE · MAE · Bias · by-horiz│
│ └ site B │  + forecast grid (dense, tabular-nums, sticky head) │
└──────────┴────────────────────────────────────────────────────┘
```

---

## 7. Evidence-quality caveats

- **Strong (observed):** Montel (screenshots), Meteologica (shipped CSS/JS tokens), enercast layout (release screenshot captions + docs), sector convention values (ONS, ERCOT, HA, Bloomberg, institutional design systems).
- **Inferred (gated behind login):** Meteologica & enercast *chart styling, exact series colors, probabilistic-band rendering, map views* — derived from marketing copy + sector norms, not direct observation. Treat §6 color picks for solar/wind as the sector standard (ERCOT/HA), not as copied-from-enercast.
- No G2/Capterra reviews exist for Meteologica or enercast (confirmed absent).

---

## 8. TL;DR directives

1. **Stay light** — it's the forecasting-sector norm (Meteologica, enercast). Dark is a later optional toggle.
2. **Adopt enercast's Workbench:** left asset/model tree · center forecast chart · bottom RMSE/MAE/Bias quality widget.
3. **Densify tables:** 28–32px rows, 4/8/12px padding, `tabular-nums slashed-zero`, right-aligned, sticky header, density toggle.
4. **Forecast chart idiom:** solid actual + dashed forecast + 20–30% confidence band (funnels wider) + dashed "Now" line + `1D/1W/1M/3M/1Y/All` selector.
5. **Color:** solar = amber `#ff9800`, wind = blue; green/red = above/below target, always with arrows. Primary brand = blue `#1565c0` (Meteologica-aligned) or teal `#0e7490` (Montel-aligned).
6. **Skip** Bloomberg's dark floating-window terminal and OHLC/technical-studies — wrong archetype for power forecasting.
7. Add **CSV/Excel export + last-updated badge + TZ selector** — table stakes the sector expects.

---

### Source index
Montel: [platforms](https://montel.energy/platforms) · [Montel Online](https://montel.energy/platforms/montel-online) · [support](https://support.montel.energy/what-is-montel-online) · [EnAppSys](https://montel.energy/platforms/enappsys) · [insightcommodity](https://insightcommodity.com/montel-online)
Meteologica: [CSS tokens](https://xtraders.meteologica.com/management-tool/assets/index-BTuxgB5e.css) · [app](https://xtraders.meteologica.com/management-tool/) · [services](https://www.meteologica.com/services.html)
enercast: [portal demo](https://www.enercast.de/enercast-portal-demo/) · [Workbench](https://www.enercast.de/magazine/enercast202202/) · [quality eval](https://www.enercast.de/magazine/interactive-forecast-quality-evaluation-in-the-enercast-portal/) · [products](https://www.enercast.de/products/)
Adjacent: [Volue](https://volue.com/data-and-forecasts) · [EQ 2.0](https://www.energyquantified.com/blog/energy-quantified-2-0-launched) · [Nord Pool portal](https://www.nordpoolgroup.com/en/services/power-market-data-services/dataportal-launch-update/) · [EPEX results](https://www.epexspot.com/en/market-results) · [Kpler power](https://www.kpler.com/market/power-market)
Conventions: [data density](https://paulwallas.medium.com/designing-for-data-density-what-most-ui-tutorials-wont-teach-you-091b3e9b51f4) · [institutional design system](https://edwson.com/design-system-showcase.html) · [tabular numbers](https://dev.to/alanwest/tabular-numbers-in-css-font-variant-numeric-vs-monospace-hacks-25cn) · [ONS chart elements](https://service-manual.ons.gov.uk/data-visualisation/build-specifications/chart-elements) · [ERCOT wind+solar](https://www.ercot.com/gridmktinfo/dashboards/combinedwindandsolar) · [HA energy colours](https://community.home-assistant.io/t/energy-panel-colours/368175) · [Bloomberg color accessibility](https://www.bloomberg.com/company/stories/designing-the-terminal-for-color-accessibility/) · [CRD dark matters](https://www.crd.com/insights/importance-of-themes-and-why-dark-matters) · [shadcn dark mode](https://ui.shadcn.com/docs/dark-mode/next)
