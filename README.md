# ⚾ MLB Outfield Positioning Tool

A Streamlit app that recommends where the LF, CF, and RF should stand against a specific batter (and optionally a specific pitcher), using Statcast batted-ball data and each fielder's range.

The tool blends the batter's spray tendencies with the pitcher's allowed spray (matched by handedness), estimates each outfielder's coverage radius from Sprint Speed and Outs Above Average (OAA), then runs an optimizer to place the three outfielders where they cover the most expected batted balls.

## Features

- **Batter × pitcher spray blend**: combines the batter's outfield balls in play with the pitcher-allowed outfield balls, with an adjustable batter weight (0.40–0.90).
- **Handedness matching**: the pitcher's data is filtered to the batter's stance, and the batter's data is filtered to the pitcher's throwing hand.
- **Range from athleticism**: coverage radius is derived from Sprint Speed and OAA for each named outfielder, falling back to league-average values when no player is given.
- **Density-weighted optimization**: a Gaussian KDE of the blended spray distribution is maximized with `scipy`'s differential evolution to place LF/CF/RF.
- **Dark-themed visualization**: heatmap of batted-ball density, subtle BIP scatter, range circles, and labeled fielder positions on an outfield diagram.
- **Alignment table**: recommended depth (ft) and angle (°) for each position alongside sprint speed, OAA, and range.

## How It Works

1. **Player lookup**: names are resolved to MLBAM IDs via `pybaseball.playerid_lookup` (the most recently active match is used).
2. **Data fetch**: Statcast balls in play (`type == "X"`) with hit coordinates are pulled for the selected season and converted from Statcast `hc_x`/`hc_y` to field coordinates in feet.
3. **Outfield filter**: only balls landing in the outfield region are kept.
4. **Density blend**: a KDE is fit for each source and combined: `w · batter + (1 − w) · pitcher`. Samples under 12 balls fall back to a uniform density.
5. **Range estimate**: `radius = 56 ft × (sprint_speed / 27) + clip(0.45 × OAA, −10, 14)`, clipped to 40–88 ft.
6. **Optimization**: each fielder's coverage falls off linearly with distance (out to 1.22 × radius). The optimizer maximizes total density-weighted coverage, with positions bounded to realistic depth and angle ranges for each spot.
7. **Fallback**: if optimization fails, league-average alignments (by batter handedness) are used.

## Installation

```bash
git clone https://github.com/MichaelBaud22/MLB-Outfield-Positioning.git
cd MLB-Outfield-Positioning
pip install streamlit pandas numpy matplotlib scipy pybaseball
```

Python 3.10+ is required (the code uses `X | Y` type-union syntax).

## Usage

```bash
streamlit run mlb_outfield_positioning.py
```

Then, in the sidebar:

| Input | Description |
|---|---|
| **Batter** | Full name, e.g. `Aaron Judge` |
| **Pitcher (optional)** | Full name; enables the pitcher blend and handedness matching |
| **Batter spray weight** | Weight on the batter's tendencies (only active with a pitcher) |
| **LF / CF / RF** | Optional outfielder names for personalized range; blank uses league average |
| **Season** | 2022–2025 |
| **Min OF BIP (batter)** | Warns if the batter has fewer outfield balls in play than this |

Click **Run optimization** to generate the heatmap, fielder positions, and alignment table.

## Output

- A field plot showing blended batted-ball density, the underlying batter (white) and pitcher (blue) balls in play, and each fielder's position with a dashed range circle.
- A **Recommended alignment** table with player, Sprint Speed, OAA, range, depth, and angle.

## Notes and Limitations

- This is a **public-data approximation**, not a production-grade defensive model. Range is a simplified function of Sprint Speed and OAA and does not account for ball hang time, route efficiency, park dimensions, or wall effects.
- Coverage uses a simple linear falloff and ignores fly-ball vs. line-drive vs. ground-ball differences.
- Small samples produce noisy density estimates; the app warns when samples are thin.
- Player lookups take the most recently active match, so common names may resolve to the wrong player.
- Statcast queries can be slow on first run; results are cached (2 hours for player data, 24 hours for sprint speed and OAA).
- Requires network access to Baseball Savant via `pybaseball`.

## Tech Stack

[Streamlit](https://streamlit.io/) · [pybaseball](https://github.com/jldbc/pybaseball) · pandas · NumPy · SciPy · Matplotlib

## Acknowledgments

Data from [Baseball Savant / Statcast](https://baseballsavant.mlb.com/), accessed through `pybaseball`.
