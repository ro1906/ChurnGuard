"""Visual identity: colours, page layout bands and the CSS that skins Streamlit.

The idea: recurring revenue is water and churn is the leak. Pages are stacked full-width
colour bands (paper, deep tide, water) joined by wavy edges, like a shoreline.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import streamlit as st

from churnguard import config

INK = "#0C2E3B"  # deep tide: text and dark bands
PAPER = "#EAF2EE"  # sea glass: page background
CARD = "#FBFDFC"  # card surface
WATER = "#12788A"  # brand / primary actions
WATER_LIGHT = "#7FD1D9"
CORAL = "#F2574B"  # leak: high risk
AMBER = "#F5B83D"  # medium risk
KELP = "#2FA877"  # saved: low risk
MUTED = "#4F6B73"

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

_WAVE = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 60' "
    "preserveAspectRatio='none'%3E%3Cpath d='M0 34 C 160 4 320 4 480 30 S 800 62 960 34 "
    "S 1280 2 1440 28 L1440 60 L0 60 Z'/%3E%3C/svg%3E\")"
)


def money(value: float) -> str:
    return f"{config.CURRENCY}{value:,.0f}"


@lru_cache(maxsize=4)
def font_data_uri(name: str) -> str:
    """Fonts as base64 so the hero (which lives in an iframe) can use them offline."""
    raw = (STATIC_DIR / "fonts" / f"{name}.woff2").read_bytes()
    return "data:font/woff2;base64," + base64.b64encode(raw).decode()


_CSS = """
<style>
:root {
  --ink: __INK__; --paper: __PAPER__; --card: __CARD__; --water: __WATER__;
  --water-light: __WATER_LIGHT__; --coral: __CORAL__; --amber: __AMBER__; --kelp: __KELP__;
  --muted: __MUTED__;
}

/* ---------- page frame ---------- */
[data-testid="stApp"], [data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stHeader"] { background: var(--paper); box-shadow: 0 1px 0 rgba(12,46,59,.08); }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stHeader"] { height: 4.2rem; }
[data-testid="stTopNav"] a, [data-testid="stTopNav"] button { border-radius: 999px !important; font-weight: 600; }
[data-testid="stTopNav"] a[aria-current="page"], [data-testid="stTopNav"] a[data-active="true"] { background: rgba(12,46,59,.12); }
[data-testid="stTopNav"] a:hover { background: rgba(12,46,59,.07); }
[data-testid="stHeaderLogo"] { height: 2.1rem; }
[data-testid="stMainBlockContainer"] { max-width: 100% !important; padding: 0 !important; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stAppViewContainer"] { overflow-x: hidden; }

/* ---------- colour bands joined by a wavy shoreline ---------- */
[class*="st-key-band-"] {
  position: relative;
  padding: 3.25rem max(1.25rem, calc((100% - 1180px) / 2)) 3.75rem;
}
[class*="st-key-band-"] > [data-testid="stVerticalBlock"] { gap: 1.25rem; }
[class*="st-key-band-"]::before {
  content: ""; position: absolute; left: 0; right: 0; top: 1px; height: 46px;
  transform: translateY(-100%); background: inherit; pointer-events: none;
  -webkit-mask: __WAVE__ 0 100% / 100% 100% no-repeat;
  mask: __WAVE__ 0 100% / 100% 100% no-repeat;
}
[class*="st-key-band-"][class*="-first-"]::before { display: none; }
[class*="st-key-band-"][class*="-first-"] { padding-top: 6.25rem; }
[class*="st-key-band-paper"] { background: var(--paper); }
[class*="st-key-band-tide"]  { background: var(--ink); color: #E4F0EC; }
[class*="st-key-band-water"] { background: var(--water); color: #F2FAF8; }
[class*="st-key-band-foam"]  { background: #DCEBE5; }

[class*="st-key-band-tide"] h1, [class*="st-key-band-tide"] h2, [class*="st-key-band-tide"] h3,
[class*="st-key-band-water"] h1, [class*="st-key-band-water"] h2, [class*="st-key-band-water"] h3 { color: #fff; }
[class*="st-key-band-tide"] [data-testid="stCaptionContainer"],
[class*="st-key-band-water"] [data-testid="stCaptionContainer"] { color: #A9C7C4; }

/* ---------- type ---------- */
h1, h2, h3 { letter-spacing: -0.025em; }
h1 { font-weight: 800 !important; line-height: 1.02 !important; }
h2 { font-weight: 800 !important; font-size: clamp(1.9rem, 3vw, 2.6rem) !important; line-height: 1.05 !important; }
h3 { font-weight: 700 !important; }
.cg-lede { font-size: 1.15rem; line-height: 1.55; max-width: 38rem; color: var(--muted); margin: 0; }
[class*="st-key-band-tide"] .cg-lede, [class*="st-key-band-water"] .cg-lede { color: #B9D3D0; }
.cg-title { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; letter-spacing: -0.03em;
  font-size: clamp(2.4rem, 5vw, 4rem); line-height: 1; margin: 0 0 0.9rem; color: #fff; }

/* ---------- metrics become stat cards ---------- */
[data-testid="stMetric"] {
  background: var(--card); border-radius: 22px; padding: 1.1rem 1.3rem;
  border: 1px solid rgba(12,46,59,.08); box-shadow: 0 18px 30px -24px rgba(12,46,59,.45);
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricLabel"] p { font-size: .95rem; font-weight: 500; }
[data-testid="stMetricValue"] { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800;
  letter-spacing: -0.03em; color: var(--ink); }
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricValue"] > div { font-size: clamp(1.5rem, 2.1vw, 2.1rem); line-height: 1.1; }

/* ---------- inputs, buttons, tabs ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a {
  border-radius: 999px; font-weight: 600; padding: .5rem 1.15rem;
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
[data-testid="stPageLink"] a:focus-visible { outline: 3px solid var(--water-light); outline-offset: 2px; }
[data-testid="stTabs"] [role="tablist"] { gap: .45rem; border-bottom: 0; box-shadow: none; padding-bottom: .4rem; }
[data-testid="stTab"] { border-radius: 999px; padding: .4rem 1.1rem; font-weight: 600; border: 1.5px solid rgba(12,46,59,.2);
  color: var(--ink); background: transparent; }
[data-testid="stTab"][data-selected="true"], [data-testid="stTab"][aria-selected="true"] { background: var(--ink); border-color: var(--ink); color: #fff; }
[data-testid="stTab"][data-selected="true"] *, [data-testid="stTab"][aria-selected="true"] * { color: #fff; }
[data-testid="stTabs"] [role="tablist"] > :not([role="tab"]) { display: none; }
[data-testid="stAlert"] { background: var(--card); border: 1px solid rgba(12,46,59,.08); border-radius: 18px; color: var(--ink); }
[class*="st-key-band-paper"] { min-height: 55vh; }

/* ---------- buttons on dark bands ---------- */
[class*="st-key-band-tide"] .stButton > button {
  background: transparent; color: #E4F0EC; border: 1.5px solid rgba(255,255,255,.38);
}
[class*="st-key-band-tide"] .stButton > button:hover { background: rgba(255,255,255,.12); border-color: #fff; color: #fff; }
.st-key-sticky-form { position: sticky; top: 5rem; }

/* ---------- file upload and radio ---------- */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed rgba(18,120,138,.55); border-radius: 24px; background: var(--card); padding: 2rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--water); background: #fff; }
[data-testid="stDataFrame"] { border-radius: 18px; overflow: hidden; }

/* ---------- segmented control and pills ---------- */
[data-testid="stButtonGroup"] button[data-variant] {
  border-radius: 999px !important; font-weight: 600; background: transparent;
  border: 1.5px solid rgba(12,46,59,.28); color: var(--ink); padding: .3rem 1.05rem;
}
[data-testid="stButtonGroup"] button[data-variant][data-selected="true"] {
  background: var(--ink); border-color: var(--ink); color: #fff;
}
[class*="st-key-band-tide"] [data-testid="stButtonGroup"] button[data-variant] { border-color: rgba(255,255,255,.35); color: #E4F0EC; }
[class*="st-key-band-tide"] [data-testid="stButtonGroup"] button[data-variant][data-selected="true"] {
  background: #fff; border-color: #fff; color: var(--ink);
}
[data-testid="stButtonGroup"] [data-testid="stWidgetLabel"] { color: var(--muted); }

/* ---------- bordered containers become cards ---------- */
[class*="st-key-card-"] {
  background: var(--card); border-radius: 24px !important; border: 1px solid rgba(12,46,59,.08) !important;
  box-shadow: 0 18px 30px -26px rgba(12,46,59,.5); padding: 1.4rem 1.5rem !important; color: var(--ink);
}

/* ---------- risk pill ---------- */
.risk-badge { display: inline-block; padding: 5px 16px; border-radius: 999px; font-weight: 700;
  font-size: .95rem; color: #fff; }
.risk-badge.Medium { color: #3b2a00; }

/* ---------- verdict card (gauge) ---------- */
.cg-verdict { background: var(--ink); color: #E4F0EC; border-radius: 28px; padding: 1.6rem 1.6rem 1.3rem;
  position: relative; overflow: hidden; }
.cg-verdict .cg-gauge { width: 100%; max-width: 360px; display: block; margin: 0 auto; }
.cg-verdict .pct { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 3.6rem;
  letter-spacing: -0.04em; line-height: 1; text-align: center; color: #fff; margin-top: -.4rem; }
.cg-verdict .sub { text-align: center; color: #A9C7C4; margin: .35rem 0 .9rem; }
.cg-verdict .row { display: flex; justify-content: space-between; gap: 1rem; border-top: 1px solid rgba(255,255,255,.12);
  padding-top: .9rem; font-size: .95rem; }
.cg-verdict .row b { font-family: "Bricolage Grotesque", sans-serif; font-size: 1.25rem; display: block; color: #fff; }

/* ---------- reasons (driver bars) ---------- */
.cg-driver { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); align-items: center;
  gap: .9rem; padding: .5rem 0; border-bottom: 1px solid rgba(12,46,59,.07); }
.cg-driver:last-child { border-bottom: 0; }
.cg-driver .name { font-weight: 600; line-height: 1.25; }
.cg-driver .name small { display: block; font-weight: 400; color: var(--muted); }
.cg-bar { position: relative; height: 12px; background: rgba(12,46,59,.06); border-radius: 99px; }
.cg-bar::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; background: rgba(12,46,59,.25); }
.cg-bar i { position: absolute; top: 0; bottom: 0; border-radius: 99px; transform-origin: var(--origin);
  animation: cg-grow .8s cubic-bezier(.2,.8,.2,1) both; }
.cg-bar i.up { left: 50%; background: var(--coral); width: var(--w); --origin: left; }
.cg-bar i.down { right: 50%; background: var(--water); width: var(--w); --origin: right; }
@keyframes cg-grow { from { transform: scaleX(0); } }
.cg-axis { display: flex; justify-content: space-between; color: var(--muted); font-size: .85rem; margin-top: .6rem; }

/* ---------- retention play cards ---------- */
.cg-play h4 { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 1.25rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
.cg-tag { display: inline-block; font-size: .8rem; font-weight: 700; padding: 2px 10px; border-radius: 99px;
  background: rgba(242,87,75,.14); color: #A22A20; margin-bottom: .5rem; }
.cg-tag.plain { background: rgba(12,46,59,.07); color: var(--muted); }
.cg-shift { display: flex; align-items: baseline; gap: .6rem; font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; }
.cg-shift .from { font-size: 1.1rem; color: var(--muted); text-decoration: line-through; text-decoration-thickness: 2px; }
.cg-shift .to { font-size: 1.9rem; letter-spacing: -.03em; }
.cg-shift .to.good { color: var(--kelp); }
.cg-shift .note { font-family: Figtree, sans-serif; font-weight: 500; font-size: .9rem; color: var(--muted); }

/* ---------- 100-customer dot grids ---------- */
.cg-waffle { display: grid; grid-template-columns: repeat(10, 1fr); gap: 9px; max-width: 420px; }
.cg-waffle i { aspect-ratio: 1; border-radius: 50%; background: rgba(12,46,59,.12); display: block; }
.cg-waffle i.on { background: var(--water); }
.cg-waffle.coral i.on { background: var(--coral); }
.cg-waffle i.pop { animation: cg-pop .5s cubic-bezier(.3,1.6,.5,1) both; animation-delay: calc(var(--n) * 14ms); }
@keyframes cg-pop { from { transform: scale(0); } }

/* ---------- call-to-action cards ---------- */
.cg-cta h3 { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; color: var(--ink) !important; margin: 0 0 .3rem; font-size: 1.5rem; }
.cg-cta p { margin: 0 0 .6rem; color: var(--muted); }


/* ---------- footer ---------- */
.cg-footer { color: #A9C7C4; font-size: .9rem; display: flex; flex-wrap: wrap; gap: .4rem 1.6rem; }
.cg-footer b { color: #fff; font-family: "Bricolage Grotesque", sans-serif; font-size: 1.15rem; letter-spacing: -.02em; }

@media (max-width: 760px) {
  [class*="st-key-band-"] { padding: 2.4rem 1.1rem 2.8rem; }
  [class*="st-key-band-"][class*="-first-"] { padding-top: 5rem; }
  .cg-driver { grid-template-columns: 1fr; gap: .35rem; }
}
@media (prefers-reduced-motion: reduce) {
  .cg-bar i, .cg-waffle i.pop { animation: none !important; }
}
</style>
"""


def inject_css() -> None:
    css = (
        _CSS.replace("__INK__", INK).replace("__PAPER__", PAPER).replace("__CARD__", CARD)
        .replace("__WATER_LIGHT__", WATER_LIGHT).replace("__WATER__", WATER)
        .replace("__CORAL__", CORAL).replace("__AMBER__", AMBER).replace("__KELP__", KELP)
        .replace("__MUTED__", MUTED).replace("__WAVE__", _WAVE)
    )
    st.html(css)


@contextmanager
def band(name: str, tone: str = "paper", first: bool = False):
    """A full-width colour band. `name` must be unique per page; `tone` picks the colour.

    The first band on a page passes first=True: it clears the top navigation and has no wave.
    """
    marker = "first-" if first else ""
    with st.container(key=f"band-{tone}-{marker}{name}"):
        yield


def page_head(title: str, lede: str) -> None:
    """Dark opening band used at the top of every page except the Overview."""
    st.html(f'<h1 class="cg-title">{title}</h1><p class="cg-lede">{lede}</p>')


def risk_badge(level: str) -> str:
    """HTML pill for a risk level. Only ever fed our own fixed level names."""
    return f'<span class="risk-badge {level}" style="background:{risk_color(level)}">{level} risk</span>'


def risk_color(level: str) -> str:
    return {"Low": KELP, "Medium": AMBER, "High": CORAL}.get(level, "#888")
