# ZENTHROSML CANVAS — Design System

## Overview

ZENTHROSML CANVAS is the design language for the MLOps platform UI.  
Core feel: **neón futurista enterprise** — black background, cyan light filaments, premium control-plane aesthetic.

---

## Color Palette

| Token | Value | Use |
|---|---|---|
| `brand.bg` | `#000000` | Canvas background |
| `brand.surface` | `#03060a` | Body background, node backgrounds |
| `brand.panel` | `#060d14` | Input backgrounds, inline surfaces |
| `neon.cyan` | `#22d3ee` | Primary: borders, labels, focus rings |
| `neon.orange` | `#f97316` | Training nodes, action buttons |
| `neon.emerald` | `#10b981` | Dataset nodes, success states, healthy LEDs |
| `neon.purple` | `#a78bfa` | Serving nodes, estimator nodes |
| `neon.red` | `#f87171` | Errors |
| `slate-100` | `#f1f5f9` | Primary text on dark |
| `slate-500/600` | — | Muted labels |

---

## Typography

| Role | Classes |
|---|---|
| Brand mark | `font-extrabold italic tracking-[0.2em] text-cyan-400` + `text-shadow: 0 0 16px rgba(34,211,238,0.6)` |
| Section title | `text-xs font-extrabold italic uppercase tracking-widest text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]` |
| Sub-heading | `text-xs font-semibold text-cyan-500/70 uppercase tracking-wider` |
| Node label | `text-[10px] font-bold uppercase tracking-widest` |
| Monospace values | `font-mono text-xs` |
| Small muted | `text-[10px] text-slate-500 uppercase tracking-wide` |

---

## Borders & Glow

### CSS classes (defined in `index.css`)

| Class | Effect |
|---|---|
| `.neon-border-cyan` | `border: 1px solid rgba(34,211,238,0.35); box-shadow: 0 0 12px rgba(34,211,238,0.2)` |
| `.neon-border-orange` | Same pattern with orange |
| `.neon-border-emerald` | Same pattern with emerald |
| `.neon-border-purple` | Same pattern with purple |
| `.neon-border-red` | Same pattern with red |
| `.neon-selected-cyan` | Active/selected — stronger border + inner glow |
| `.glass-panel` | `background: rgba(6,13,20,0.7); backdrop-filter: blur(12px); border: 1px solid rgba(34,211,238,0.15)` |
| `.filament-divider` | Gradient `hr` — horizontal light filament separator |

### Tailwind shadow utilities (defined in `tailwind.config.js`)

```
shadow-glow-cyan     0 0 15px rgba(34,211,238,0.35)
shadow-glow-cyan-lg  0 0 30px rgba(34,211,238,0.5)
shadow-glow-orange   0 0 15px rgba(249,115,22,0.35)
shadow-glow-emerald  0 0 15px rgba(16,185,129,0.35)
shadow-glow-purple   0 0 15px rgba(167,139,250,0.35)
shadow-glow-red      0 0 15px rgba(248,113,113,0.35)
```

---

## Shared UI Tokens (`src/lib/uiTokens.ts`)

```typescript
INPUT_CLS    // rounded border-cyan-500/20 bg-brand-panel text-cyan-100 focus:ring-cyan-500/50
SELECT_CLS   // same as INPUT_CLS
BTN_NEUTRAL  // slate border, hover cyan tint
BTN_PRIMARY  // cyan border + bg, shadow-glow-cyan
BTN_DANGER   // orange border + bg
BTN_SUCCESS  // emerald border + bg, shadow-glow-emerald
SUB_HEADING  // text-xs font-semibold text-cyan-500/70 uppercase tracking-wider
```

Always import from `src/lib/uiTokens.ts`. Never define local copies in pages.

---

## Design System Components (`src/design/components/`)

| Component | Purpose |
|---|---|
| `GlassPanel` | Glass surface container with optional glow |
| `NeonBorder` | Glow border wrapper, `color` prop |
| `StatusLED` | Animated pulsing LED indicator, `status` prop |
| `SectionTitle` | Extrabold italic cyan heading |
| `NodeCard` | Rounded capsule base for any node, `color` + `selected` props |
| `CodePanel` | Monospace code/log surface, black background |
| `IconButton` | Small button with variant glow |

All exported from `src/design/components/index.ts`.

---

## Node Colors by Kind

| Kind | Color | LED | Border class |
|---|---|---|---|
| dataset | emerald | green | `neon-border-emerald` |
| processing | cyan | cyan | `neon-border-cyan` |
| training | orange | orange | `neon-border-orange` |
| serving | purple | purple | `neon-border-purple` |
| merge (stub) | slate (disabled) | — | — |
| decision (stub) | slate (disabled) | — | — |

---

## Animations

### CSS keyframes (in `index.css`)

- `led-pulse` — opacity 1↔0.55, 2s ease-in-out infinite. Apply via `.led-pulse` class.  
- `flow-dash` — stroke-dashoffset animation on fiber optic edges. Apply via `.edge-flow` class.

### Principles

- Animations must feel **precise and live**, not playful or caricatured.
- LED pulse: slow, subtle. Not flashy.
- Edge flow: only on active/propagated edges.
- Panel slide-in: CSS `transition` + `width` change only (no Framer Motion dependency).
- Hover effects: `scale` avoided; use `border` / `shadow` / `color` transitions instead.

---

## Layout Structure

```
App
└── OrchestrationCanvasPage (default, full height)
    ├── BrandHeader       (glass, h-11, sticky top)
    ├── Row:
    │   ├── NodePalette   (glass, w-[160px], sticky left)
    │   ├── OrchestrationCanvas  (flex-1, ReactFlow instance)
    │   └── NodeInspector (glass, w-[440px], slides in from right)
    └── StatusBar         (glass, h-8, sticky bottom)
```

DSL Builder layout (unchanged):
```
MainLayout
├── Header  (glass)
├── Row:
│   ├── CatalogPanel
│   ├── CanvasPanel (ReactFlow instance)
│   └── PropertiesPanel
└── ConsolePanel
```

---

## ReactFlow Config

Two independent ReactFlow instances:

| Instance | Provider scope | Store | Background |
|---|---|---|---|
| OrchestrationCanvas | `OrchestrationCanvasPage` | `dagStore` | `#000000` |
| CanvasPanel (DSL) | `App` (dsl-builder page) | `pipelineStore` | `#000000` |

Edge type `fiber` → `FiberOpticEdge` — cyan glowing bezier with optional flow animation when artifact ID is propagated.

---

## Extension Rules

When adding new components or pages:
1. Use `GlassPanel`, `NeonBorder`, `NodeCard`, `StatusLED`, `IconButton` from `src/design/components/`.
2. Use `INPUT_CLS`, `SELECT_CLS`, `BTN_*`, `SUB_HEADING` from `src/lib/uiTokens.ts`.
3. Use `filament-divider` for section separators (not plain `<hr>`).
4. New node types: add to `OrchNodeKind` in `src/types/dag.ts`, create visual in `OrchestrationCanvas/nodes/`, add to `nodeTypes` map in `OrchestrationCanvas.tsx`, add inspector in `inspectors/`, add to `NodePalette`.
5. No gray as dominant UI color. No flat/solid gray separators.
6. No Framer Motion (not installed). Use CSS `transition` only.
