"""
MLB Outfield Positioning Tool
-----------------------------
High-end style UI + polished outfield visualization.
Batter + pitcher spray blend (handedness-matched), range from Sprint Speed + OAA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Arc
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings("ignore")

try:
    from pybaseball import (
        playerid_lookup,
        statcast_batter,
        statcast_pitcher,
        statcast_sprint_speed,
        statcast_outs_above_average,
    )
except ImportError:
    st.error("pybaseball is required. Run: `pip install pybaseball`")
    st.stop()

# ---------------------------------------------------------------------------
# Design tokens — dark analytics aesthetic
# ---------------------------------------------------------------------------
BG_FIELD = "#0d1f0d"
GRASS = "#1a3a1a"
DIRT = "#3d2b1f"
FOUL_LINE = "#e8e4d9"
FENCE = "#c4a35a"
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "mlb_heat",
    ["#0d1f0d", "#1a3a2a", "#2d5a3a", "#4a7a2a", "#8a9a20", "#d4a017", "#e85d04", "#c1121f"],
)
POS_COLORS = {"LF": "#00b4d8", "CF": "#90e0ef", "RF": "#48cae4"}
POS_EDGE = {"LF": "#0077b6", "CF": "#00b4d8", "RF": "#0096c7"}
TEXT_LIGHT = "#f0efe8"
TEXT_MUTED = "#a8a49a"
ACCENT = "#d4a017"

LEAGUE_AVG_OF = {
    "R": {
        "LF": {"depth": 295, "angle": -28},
        "CF": {"depth": 325, "angle": 2},
        "RF": {"depth": 295, "angle": 28},
    },
    "L": {
        "LF": {"depth": 295, "angle": -28},
        "CF": {"depth": 325, "angle": -2},
        "RF": {"depth": 295, "angle": 28},
    },
}

DEFAULT_SPRINT = 27.0
DEFAULT_OAA = 0.0


def polar_to_xy(depth: float, angle_deg: float) -> tuple[float, float]:
    rad = np.deg2rad(angle_deg)
    return depth * np.sin(rad), depth * np.cos(rad)


def xy_to_polar(x: float, y: float) -> tuple[float, float]:
    return np.hypot(x, y), np.rad2deg(np.arctan2(x, y))


def transform_statcast_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = 2.5 * (out["hc_x"] - 125.42)
    out["y"] = 2.5 * (198.27 - out["hc_y"])
    return out


def parse_full_name(name: str) -> tuple[str, str] | None:
    """Split 'First Last' or 'First Middle Last' into (first, last)."""
    name = (name or "").strip()
    if not name:
        return None
    parts = name.split()
    if len(parts) < 2:
        return None
    return parts[0], " ".join(parts[1:])


def draw_outfield(ax, foul_distance=330, of_distance=405):
    """Sleek dark-field outfield canvas."""
    ax.set_facecolor(BG_FIELD)

    # Subtle grass gradient rings
    for r, alpha in [(120, 0.15), (200, 0.12), (280, 0.10), (360, 0.08)]:
        theta = np.linspace(-np.pi / 4 - 0.05, np.pi / 4 + 0.05, 120)
        ax.fill(
            np.concatenate([[0], r * np.sin(theta), [0]]),
            np.concatenate([[0], r * np.cos(theta), [0]]),
            color=GRASS,
            alpha=alpha,
            zorder=0,
        )

    # Foul lines
    ax.plot(
        [0, -foul_distance * 0.7071],
        [0, foul_distance * 0.7071],
        color=FOUL_LINE,
        lw=1.4,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        [0, foul_distance * 0.7071],
        [0, foul_distance * 0.7071],
        color=FOUL_LINE,
        lw=1.4,
        alpha=0.85,
        zorder=2,
    )

    # Infield dirt arc
    theta = np.linspace(-np.pi / 4, np.pi / 4, 90)
    ax.plot(95 * np.sin(theta), 95 * np.cos(theta), color=DIRT, lw=2.0, alpha=0.7, zorder=2)

    # Warning track / fence
    ax.plot(
        of_distance * np.sin(theta),
        of_distance * np.cos(theta),
        color=FENCE,
        lw=2.8,
        alpha=0.9,
        zorder=3,
    )

    # Bases + mound (minimal reference)
    ax.scatter([0], [0], c=FOUL_LINE, s=55, marker="p", zorder=4, edgecolors="#2a2a2a", linewidths=0.6)
    ax.scatter(
        [63.64, 0, -63.64],
        [63.64, 127.28, 63.64],
        c=FOUL_LINE,
        s=28,
        marker="s",
        zorder=4,
        edgecolors="#2a2a2a",
        linewidths=0.5,
    )
    ax.scatter([0], [60.5], c=DIRT, s=22, zorder=4)

    # Depth markers (subtle)
    for d in [250, 300, 350]:
        ax.plot(
            d * np.sin(theta),
            d * np.cos(theta),
            color=TEXT_MUTED,
            lw=0.6,
            alpha=0.25,
            linestyle=":",
            zorder=1,
        )
        ax.text(
            0,
            d + 6,
            f"{d}'",
            ha="center",
            va="bottom",
            fontsize=7,
            color=TEXT_MUTED,
            alpha=0.45,
            fontfamily="sans-serif",
            zorder=1,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-355, 355)
    ax.set_ylim(70, 430)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def estimate_range_radius(sprint_speed: float, oaa: float) -> float:
    sprint_factor = sprint_speed / DEFAULT_SPRINT
    base = 56.0 * sprint_factor
    oaa_bonus = np.clip(oaa * 0.45, -10, 14)
    return float(np.clip(base + oaa_bonus, 40, 88))


def get_outfield_bip(bip: pd.DataFrame) -> pd.DataFrame:
    if bip.empty:
        return bip
    return bip[(bip["y"] > 180) & (bip["y"] < 420) & (np.abs(bip["x"]) < bip["y"] + 40)].copy()


@st.cache_data(ttl=7200, show_spinner=False)
def lookup_player(last: str, first: str) -> pd.DataFrame:
    return playerid_lookup(last.strip(), first.strip())


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_batter_bip(player_id: int, start: str, end: str, pitcher_throws: str | None = None) -> pd.DataFrame:
    df = statcast_batter(start, end, player_id)
    if df is None or df.empty:
        return pd.DataFrame()
    bip = df[df["type"] == "X"].dropna(subset=["hc_x", "hc_y"]).copy()
    if pitcher_throws in ("L", "R") and "p_throws" in bip.columns:
        bip = bip[bip["p_throws"] == pitcher_throws]
    return transform_statcast_coords(bip)


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_pitcher_bip(player_id: int, start: str, end: str, batter_stand: str | None = None) -> pd.DataFrame:
    df = statcast_pitcher(start, end, player_id)
    if df is None or df.empty:
        return pd.DataFrame()
    bip = df[df["type"] == "X"].dropna(subset=["hc_x", "hc_y"]).copy()
    if batter_stand in ("L", "R") and "stand" in bip.columns:
        bip = bip[bip["stand"] == batter_stand]
    return transform_statcast_coords(bip)


@st.cache_data(ttl=86400, show_spinner=False)
def load_sprint_speed(year: int) -> pd.DataFrame:
    try:
        return statcast_sprint_speed(year, min_opp=5)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def load_oaa(year: int) -> pd.DataFrame:
    try:
        return statcast_outs_above_average(year, "of", min_att=5)
    except Exception:
        return pd.DataFrame()


def get_player_metrics(mlbam_id: int, sprint_df: pd.DataFrame, oaa_df: pd.DataFrame) -> dict:
    sprint = DEFAULT_SPRINT
    oaa = DEFAULT_OAA
    if not sprint_df.empty and "player_id" in sprint_df.columns:
        row = sprint_df[sprint_df["player_id"] == mlbam_id]
        if not row.empty and "sprint_speed" in row.columns:
            sprint = float(row["sprint_speed"].iloc[0])
    if not oaa_df.empty and "player_id" in oaa_df.columns:
        row = oaa_df[oaa_df["player_id"] == mlbam_id]
        if not row.empty and "outs_above_average" in row.columns:
            oaa = float(row["outs_above_average"].iloc[0])
    return {"sprint": sprint, "oaa": oaa, "radius": estimate_range_radius(sprint, oaa)}


def blend_densities(
    batter_of: pd.DataFrame,
    pitcher_of: pd.DataFrame | None,
    batter_weight: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(-300, 300, 55)
    yi = np.linspace(190, 395, 45)
    X, Y = np.meshgrid(xi, yi)
    positions = np.vstack([X.ravel(), Y.ravel()])

    def _kde_density(df: pd.DataFrame) -> np.ndarray:
        if df is None or len(df) < 12:
            return np.ones(X.size) / X.size
        xs, ys = df["x"].values, df["y"].values
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            z = kde(positions)
            z = np.clip(z, 0, None)
            if z.sum() > 0:
                z /= z.sum()
            return z
        except Exception:
            return np.ones(X.size) / X.size

    bat_z = _kde_density(batter_of)
    if pitcher_of is not None and len(pitcher_of) >= 12:
        pit_z = _kde_density(pitcher_of)
        w = np.clip(batter_weight, 0.3, 0.9)
        blended = w * bat_z + (1 - w) * pit_z
    else:
        blended = bat_z
    blended = blended / blended.sum()
    return X, Y, blended.reshape(X.shape)


def optimize_outfield_positions(
    dens_X: np.ndarray,
    dens_Y: np.ndarray,
    dens: np.ndarray,
    stand: str,
    radii: dict[str, float],
) -> dict[str, tuple[float, float]]:
    stand = stand if stand in ("R", "L") else "R"
    base = LEAGUE_AVG_OF[stand]
    bounds = {
        "LF": [(235, 345), (-44, -10)],
        "CF": [(275, 365), (-16, 16)],
        "RF": [(235, 345), (10, 44)],
    }

    def objective(params):
        positions = {
            "LF": polar_to_xy(params[0], params[1]),
            "CF": polar_to_xy(params[2], params[3]),
            "RF": polar_to_xy(params[4], params[5]),
        }
        coverage = np.zeros_like(dens)
        for pos, (px, py) in positions.items():
            r = radii.get(pos, 58)
            dist = np.sqrt((dens_X - px) ** 2 + (dens_Y - py) ** 2)
            cov = np.clip(1.0 - dist / (1.22 * r), 0, 1)
            coverage = np.maximum(coverage, cov)
        return -np.sum(dens * coverage)

    flat_bounds = []
    for pos in ["LF", "CF", "RF"]:
        flat_bounds.extend(bounds[pos])

    try:
        result = differential_evolution(
            objective,
            bounds=flat_bounds,
            popsize=11,
            mutation=0.55,
            recombination=0.7,
            seed=42,
            maxiter=28,
            tol=1e-3,
            workers=1,
            updating="immediate",
        )
        p = result.x
        return {
            "LF": polar_to_xy(p[0], p[1]),
            "CF": polar_to_xy(p[2], p[3]),
            "RF": polar_to_xy(p[4], p[5]),
        }
    except Exception:
        return {pos: polar_to_xy(info["depth"], info["angle"]) for pos, info in base.items()}


def main():
    st.set_page_config(
        page_title="Outfield Positioning",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Force full dark theme (app + sidebar + widgets)
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg: #0b0f0c;
            --bg-sidebar: #0e140e;
            --bg-input: #1a211a;
            --border: #2a3a2a;
            --text: #f0efe8;
            --text-muted: #b0aca0;
            --accent: #d4a017;
            --green: #2d5a3a;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Main app */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg) !important;
        }
        .main .block-container { background-color: var(--bg) !important; }

        /* Headings & body text */
        h1, h2, h3, h4 {
            color: var(--text) !important;
            font-weight: 600 !important;
        }
        h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
        p, span, label, .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
            color: var(--text-muted) !important;
        }

        /* Sidebar — force dark background + light text */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: var(--bg-sidebar) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text-muted) !important;
        }
        [data-testid="stSidebar"] h3 {
            color: var(--text) !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.5rem;
        }

        /* Inputs */
        .stTextInput input,
        .stTextInput > div > div > input {
            background-color: var(--bg-input) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
        }
        .stTextInput input::placeholder { color: #6a6660 !important; }

        /* Selectbox */
        [data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] > div {
            background-color: var(--bg-input) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }
        [data-baseweb="popover"] { background-color: var(--bg-input) !important; }
        [data-baseweb="menu"] { background-color: var(--bg-input) !important; }
        [data-baseweb="option"] { color: var(--text) !important; }

        /* Slider */
        .stSlider label, [data-testid="stSidebar"] .stSlider label {
            color: var(--text-muted) !important;
        }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: var(--accent) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #1a3a2a 0%, #2d5a3a 100%) !important;
            color: var(--text) !important;
            border: 1px solid #3d6a4a !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em;
        }
        .stButton > button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2d5a3a 0%, #3d7a4a 100%) !important;
        }

        /* Dataframe */
        .stDataFrame, [data-testid="stDataFrame"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        .stDataFrame th, .stDataFrame td {
            color: var(--text) !important;
            background-color: #141a14 !important;
        }

        /* Metric cards */
        .metric-card {
            background: #141a14;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
        }

        /* Help / tooltip icons */
        [data-testid="stTooltipIcon"] { color: var(--text-muted) !important; }

        /* Spinner / status */
        .stSpinner > div { border-top-color: var(--accent) !important; }

        /* Success / info / warning boxes */
        .stAlert { background-color: #141a14 !important; border: 1px solid var(--border) !important; color: var(--text) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Outfield Positioning")
    st.caption("Density-aware alignment · Sprint Speed + OAA range · Batter × Pitcher spray blend")

    with st.sidebar:
        st.markdown("### Matchup")
        batter_name = st.text_input(
            "Batter",
            value="Aaron Judge",
            placeholder="First Last",
            help="Full name, e.g. Aaron Judge",
        )
        pitcher_name = st.text_input(
            "Pitcher (optional)",
            value="",
            placeholder="First Last — leave blank for batter only",
            help="When set, blends pitcher-allowed spray (handedness-matched)",
        )
        batter_weight = st.slider(
            "Batter spray weight",
            0.40,
            0.90,
            0.65,
            0.05,
            help="Higher = more weight on the batter’s tendencies",
            disabled=not bool(pitcher_name.strip()),
        )

        st.markdown("---")
        st.markdown("### Outfielders")
        st.caption("Optional — blank uses league-average range")
        lf_name = st.text_input("LF", value="", placeholder="First Last")
        cf_name = st.text_input("CF", value="", placeholder="First Last")
        rf_name = st.text_input("RF", value="", placeholder="First Last")

        st.markdown("---")
        st.markdown("### Data")
        season = st.selectbox("Season", [2025, 2024, 2023, 2022], index=1)
        min_bip = st.slider("Min OF BIP (batter)", 15, 150, 25)

        st.markdown("")
        run = st.button("Run optimization", type="primary", use_container_width=True)

    if not run:
        st.markdown(
            """
            <div style="color:#a8a49a; font-size:0.95rem; line-height:1.6; max-width:560px;">
            Enter a batter (and optionally a pitcher + outfielders), then run.<br><br>
            <b style="color:#d4a017;">Blend</b> — batter spray + pitcher-allowed spray, matched by handedness<br>
            <b style="color:#d4a017;">Range</b> — Sprint Speed + OAA → coverage radius<br>
            <b style="color:#d4a017;">Optimize</b> — places LF / CF / RF to maximize density-weighted coverage
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ---- Resolve batter ----
    parsed_b = parse_full_name(batter_name)
    if not parsed_b:
        st.error("Enter the batter as **First Last**.")
        return
    b_first, b_last = parsed_b
    with st.spinner("Looking up batter…"):
        b_lookup = lookup_player(b_last, b_first)
    if b_lookup.empty:
        st.error(f"Batter “{batter_name}” not found.")
        return
    b_lookup = b_lookup.sort_values("mlb_played_last", ascending=False)
    batter = b_lookup.iloc[0]
    batter_id = int(batter["key_mlbam"])
    batter_display = f"{batter['name_first']} {batter['name_last']}"

    # ---- Resolve pitcher ----
    pitcher_id = None
    pitcher_display = None
    if pitcher_name.strip():
        parsed_p = parse_full_name(pitcher_name)
        if not parsed_p:
            st.warning("Pitcher name should be **First Last** — continuing with batter only.")
        else:
            p_first, p_last = parsed_p
            with st.spinner("Looking up pitcher…"):
                p_lookup = lookup_player(p_last, p_first)
            if p_lookup.empty:
                st.warning(f"Pitcher “{pitcher_name}” not found — batter spray only.")
            else:
                p_lookup = p_lookup.sort_values("mlb_played_last", ascending=False)
                pitcher = p_lookup.iloc[0]
                pitcher_id = int(pitcher["key_mlbam"])
                pitcher_display = f"{pitcher['name_first']} {pitcher['name_last']}"

    start = f"{season}-03-20"
    end = f"{season}-10-05"

    with st.spinner("Fetching batter BIP…"):
        batter_bip_all = fetch_batter_bip(batter_id, start, end, pitcher_throws=None)
    if batter_bip_all.empty:
        st.warning("No BIP with location data for this batter/season.")
        return

    batter_stand = (
        batter_bip_all["stand"].mode().iloc[0]
        if "stand" in batter_bip_all.columns and not batter_bip_all["stand"].isna().all()
        else "R"
    )

    pitcher_throws = None
    pitcher_of = None
    if pitcher_id is not None:
        with st.spinner("Fetching pitcher-allowed BIP (handedness-matched)…"):
            pitcher_sample = fetch_pitcher_bip(pitcher_id, start, end, batter_stand=None)
            if not pitcher_sample.empty and "p_throws" in pitcher_sample.columns:
                pitcher_throws = pitcher_sample["p_throws"].mode().iloc[0]
            pitcher_bip = fetch_pitcher_bip(pitcher_id, start, end, batter_stand=batter_stand)
            pitcher_of = get_outfield_bip(pitcher_bip)
            if len(pitcher_of) < 15:
                st.warning(
                    f"Only {len(pitcher_of)} OF BIP allowed by pitcher vs {batter_stand}HH — "
                    "blend will lean on the batter."
                )
                if pitcher_of.empty:
                    pitcher_of = None

    if pitcher_throws in ("L", "R"):
        batter_bip = fetch_batter_bip(batter_id, start, end, pitcher_throws=pitcher_throws)
    else:
        batter_bip = batter_bip_all

    batter_of = get_outfield_bip(batter_bip)
    if len(batter_of) < min_bip:
        st.warning(
            f"Only {len(batter_of)} outfield BIP for the batter"
            + (f" vs {pitcher_throws}HP" if pitcher_throws else "")
            + ". Try another season or lower the minimum."
        )
        if batter_of.empty:
            return

    # Status strip
    cols = st.columns(4)
    cols[0].markdown(
        f"<div class='metric-card'><div style='font-size:0.7rem;color:#a8a49a;text-transform:uppercase;letter-spacing:0.06em;'>Batter</div>"
        f"<div style='font-size:1.05rem;color:#f0efe8;font-weight:600;'>{batter_display}</div></div>",
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f"<div class='metric-card'><div style='font-size:0.7rem;color:#a8a49a;text-transform:uppercase;letter-spacing:0.06em;'>Pitcher</div>"
        f"<div style='font-size:1.05rem;color:#f0efe8;font-weight:600;'>{pitcher_display or '—'}</div></div>",
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f"<div class='metric-card'><div style='font-size:0.7rem;color:#a8a49a;text-transform:uppercase;letter-spacing:0.06em;'>Batter OF BIP</div>"
        f"<div style='font-size:1.05rem;color:#f0efe8;font-weight:600;'>{len(batter_of)}"
        f"<span style='font-size:0.8rem;color:#a8a49a;font-weight:400;'>"
        f"{' · vs ' + pitcher_throws + 'HP' if pitcher_throws else ''}</span></div></div>",
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        f"<div class='metric-card'><div style='font-size:0.7rem;color:#a8a49a;text-transform:uppercase;letter-spacing:0.06em;'>Pitcher OF BIP</div>"
        f"<div style='font-size:1.05rem;color:#f0efe8;font-weight:600;'>"
        f"{len(pitcher_of) if pitcher_of is not None else '—'}"
        f"<span style='font-size:0.8rem;color:#a8a49a;font-weight:400;'>"
        f"{' · vs ' + batter_stand + 'HH' if pitcher_of is not None else ''}</span></div></div>",
        unsafe_allow_html=True,
    )

    # Metrics
    with st.spinner("Loading Sprint Speed & OAA…"):
        sprint_df = load_sprint_speed(season)
        oaa_df = load_oaa(season)

    of_inputs = {"LF": lf_name, "CF": cf_name, "RF": rf_name}
    metrics = {}
    for pos, name in of_inputs.items():
        parsed = parse_full_name(name)
        if parsed:
            first, last = parsed
            lk = lookup_player(last, first)
            if not lk.empty:
                lk = lk.sort_values("mlb_played_last", ascending=False)
                pid = int(lk.iloc[0]["key_mlbam"])
                m = get_player_metrics(pid, sprint_df, oaa_df)
                m["name"] = f"{lk.iloc[0]['name_first']} {lk.iloc[0]['name_last']}"
                metrics[pos] = m
            else:
                metrics[pos] = {
                    "sprint": DEFAULT_SPRINT,
                    "oaa": DEFAULT_OAA,
                    "radius": estimate_range_radius(DEFAULT_SPRINT, DEFAULT_OAA),
                    "name": "League avg (not found)",
                }
        else:
            metrics[pos] = {
                "sprint": DEFAULT_SPRINT,
                "oaa": DEFAULT_OAA,
                "radius": estimate_range_radius(DEFAULT_SPRINT, DEFAULT_OAA),
                "name": "League avg",
            }

    radii = {pos: m["radius"] for pos, m in metrics.items()}

    with st.spinner("Blending densities & optimizing…"):
        dens_X, dens_Y, dens = blend_densities(
            batter_of,
            pitcher_of,
            batter_weight=batter_weight if pitcher_of is not None else 1.0,
        )
        positions = optimize_outfield_positions(dens_X, dens_Y, dens, batter_stand, radii)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10.5, 10), facecolor=BG_FIELD)
    draw_outfield(ax)

    # Heatmap
    ax.contourf(dens_X, dens_Y, dens, levels=16, cmap=HEAT_CMAP, alpha=0.72, zorder=1)

    # BIP scatter (very subtle)
    ax.scatter(
        batter_of["x"],
        batter_of["y"],
        c="#f0efe8",
        s=6,
        alpha=0.18,
        zorder=3,
        linewidths=0,
    )
    if pitcher_of is not None and len(pitcher_of) > 0:
        ax.scatter(
            pitcher_of["x"],
            pitcher_of["y"],
            c="#48cae4",
            s=5,
            alpha=0.14,
            zorder=3,
            linewidths=0,
        )

    # Positions + range
    for pos in ["LF", "CF", "RF"]:
        x, y = positions[pos]
        color = POS_COLORS[pos]
        edge = POS_EDGE[pos]
        r = radii[pos]
        # Soft fill under range
        ax.add_patch(
            Circle((x, y), r, facecolor=color, edgecolor="none", alpha=0.08, zorder=4)
        )
        ax.add_patch(
            Circle(
                (x, y),
                r,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                linestyle=(0, (4, 3)),
                alpha=0.95,
                zorder=5,
            )
        )
        ax.scatter(
            x,
            y,
            c=color,
            s=200,
            edgecolors=edge,
            linewidths=2.2,
            zorder=7,
        )
        ax.annotate(
            pos,
            (x, y),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=10,
            fontweight="600",
            color=color,
            fontfamily="sans-serif",
            zorder=8,
        )

    # Title block
    title = batter_display
    if pitcher_display:
        title += f"  ·  {pitcher_display}"
    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="600",
        color=TEXT_LIGHT,
        fontfamily="sans-serif",
    )
    ax.text(
        0.02,
        0.945,
        f"{season}  ·  density-weighted coverage  ·  Sprint Speed + OAA range",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=TEXT_MUTED,
        fontfamily="sans-serif",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#c1121f", alpha=0.6, label="BIP density"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f0efe8", markersize=5, alpha=0.5, label="Batter BIP", linestyle="None"),
    ]
    if pitcher_of is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#48cae4", markersize=5, alpha=0.5, label="Pitcher BIP", linestyle="None")
        )
    leg = ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        frameon=True,
        facecolor="#141a14",
        edgecolor="#2a3a2a",
        labelcolor=TEXT_MUTED,
    )
    leg.get_frame().set_alpha(0.92)

    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Table
    st.markdown("#### Recommended alignment")
    rows = []
    for pos in ["LF", "CF", "RF"]:
        m = metrics[pos]
        x, y = positions[pos]
        depth, angle = xy_to_polar(x, y)
        rows.append(
            {
                "Pos": pos,
                "Player": m["name"],
                "Sprint (ft/s)": round(m["sprint"], 1),
                "OAA": round(m["oaa"], 1),
                "Range (ft)": round(m["radius"], 1),
                "Depth (ft)": round(depth, 1),
                "Angle (°)": round(angle, 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "Range ≈ f(Sprint Speed, OAA). Density blends batter spray with pitcher-allowed spray when a pitcher is set. "
        "Handedness-matched on both sides. Public-data approximation."
    )


if __name__ == "__main__":
    main()
