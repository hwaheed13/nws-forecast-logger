# Daily Dew Point — Redesign Concept

## Design Language
Inspired by Function Health, Heidi Health, Fortuna Health:
- Clean white backgrounds with subtle gray sections
- Bold sans-serif headings (Inter or similar)
- Generous white space
- Accent color: deep indigo (#4F46E5) for primary actions, amber (#F59E0B) for warnings
- Cards with soft shadows, rounded corners (12-16px)
- Minimal borders — use elevation/shadow instead
- Mobile-first, single-column on small screens, two-column on desktop

---

## Page Structure

### 1. Top Bar (sticky)
```
┌──────────────────────────────────────────────────┐
│  🌡️ Daily Dew Point          NYC ▾    [Dark/Light] │
└──────────────────────────────────────────────────┘
```
- Logo + name left
- City switcher dropdown right
- Optional dark mode toggle
- Clean, minimal, no clutter

---

### 2. Hero Section — "Today's Pick" + "Tomorrow's Pick"
The most important thing on the page. Two large cards side by side (stacked on mobile).

```
┌─────────────────────────┐  ┌─────────────────────────┐
│  TODAY                  │  │  TOMORROW               │
│  Mon, Mar 24            │  │  Tue, Mar 25            │
│                         │  │                         │
│  🎯 48° to 49°          │  │  🎯 54° to 55°          │
│  ML Confidence: 71%     │  │  ML Confidence: 62%     │
│  Runner-up: 50-51 (18%) │  │  Runner-up: 52-53 (24%) │
│                         │  │                         │
│  ┌─────────────────┐    │  │  ┌─────────────────┐    │
│  │ LEAN            │    │  │  │ STRONG BET       │    │
│  └─────────────────┘    │  │  └─────────────────┘    │
│                         │  │                         │
│  Kalshi: 48-49 (53%)    │  │  Kalshi: 54-55 (41%)    │
│  NWS: 48°F              │  │  NWS: 52°F              │
│  AccuWeather: 47°F      │  │  AccuWeather: 52°F      │
│                         │  │                         │
│  ⚠️ Near lower edge     │  │  🌙 Overnight ~53°F     │
│                         │  │  may exceed forecast     │
└─────────────────────────┘  └─────────────────────────┘
```

Each card contains:
- Date
- ML bucket pick (large, prominent)
- Confidence % and runner-up
- Bet signal badge
- Kalshi market leader (for comparison)
- NWS + AccuWeather forecasts (small, reference)
- Warnings if applicable (boundary, overnight)

This replaces 6+ separate sections with 2 unified cards.

---

### 3. Forecast Timeline (collapsible)
Shows how forecasts evolved throughout the day. Currently scattered across multiple areas.

```
┌──────────────────────────────────────────────────┐
│  Forecast Timeline                          ▾    │
│                                                  │
│  Today (Mar 24)                                  │
│  ──●── 1:48am 59°F                              │
│  ──●── 3:21am 56°F                              │
│  ──●── 3:57pm 58°F  ← latest                    │
│                                                  │
│  Tomorrow (Mar 25)                               │
│  ──●── 1:37am 63°F                              │
│  ──●── 3:32am 65°F  ← latest                    │
└──────────────────────────────────────────────────┘
```

Clean timeline view. Replaces the mini forecast chips.

---

### 4. Live Data Bar
Current observations — what's happening right now.

```
┌──────────────────────────────────────────────────┐
│  Observed: 52.0°F (10:51 AM)  │  6hr Max: 52°F  │
│  CLI High: 50°F               │  DSM: 50°F      │
└──────────────────────────────────────────────────┘
```

Compact horizontal bar. Not cards — just data points.

---

### 5. Model Performance
Honest stats about how the system is doing.

```
┌──────────────────────────────────────────────────┐
│  Model Performance                               │
│                                                  │
│  ML Bucket Hit Rate    33% (1/3)                 │
│  Ensemble ±1°F         39%                       │
│  AccuWeather MAE       1.03°F                    │
│  Days Tracked          260                       │
│                                                  │
│  ▸ View detailed accuracy breakdown              │
└──────────────────────────────────────────────────┘
```

Clean, minimal. Detailed stats hidden behind an expand.

---

### 6. Historical Data (expandable/separate page)
The full table and chart. Currently dominates the page.
Move to a separate tab or collapsible section so the main view stays clean.

```
Tabs:  [Overview]  [History]  [About]

History tab:
- Interactive chart
- Filterable table
- Date range picker
```

---

### 7. Footer
```
┌──────────────────────────────────────────────────┐
│  © 2026 Daily Dew Point                         │
│  About · Changelog · Terms · Privacy             │
│  Informational only, not financial advice.       │
└──────────────────────────────────────────────────┘
```

---

## Key Design Principles

1. **The pick is the product.** ML bucket prediction front and center.
   Everything else supports that one number.

2. **Progressive disclosure.** Show the essentials first.
   Details available on click/expand, not cluttering the view.

3. **Two-card hero.** Today + Tomorrow side by side.
   Each card is self-contained — you can make a decision from one card.

4. **Honest performance.** Win rate visible, not hidden.
   Builds trust. Users see the model improve over time.

5. **Mobile-first.** Cards stack vertically.
   Most users will check this on their phone.

---

## Color Palette

- Background: #FFFFFF (light) / #0F172A (dark)
- Cards: #F8FAFC with subtle shadow
- Primary accent: #4F46E5 (indigo)
- Success/WIN: #16A34A (green)
- Warning: #F59E0B (amber)
- Error/MISS: #DC2626 (red)
- Text primary: #0F172A
- Text secondary: #64748B
- Muted: #94A3B8

## Typography

- Headings: Inter (700 weight)
- Body: Inter (400 weight)
- Data/numbers: Inter (600 weight, tabular figures)
- Monospace (for temps): JetBrains Mono or system mono
