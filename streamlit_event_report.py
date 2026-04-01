from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale


@dataclass
class ColumnMapping:
    player: str
    time: str
    x: str
    y: str
    speed: str


@dataclass
class DataOptions:
    speed_unit: str
    auto_scale_xy: bool
    clamp_to_pitch: bool


DEFAULT_SPRINT_THRESHOLD = 25.0
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _find_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_lookup:
            return lower_lookup[cand]
    return None


def suggest_mapping(df: pd.DataFrame) -> ColumnMapping | None:
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    if not cols:
        return None

    player = _find_first(
        cols,
        ["player", "player_name", "player display name", "joueur", "athlete", "name"],
    )
    time = _find_first(
        cols,
        ["timestamp", "time", "datetime", "event_time", "session time", "clock"],
    )
    x = _find_first(cols, ["x", "posx", "position_x", "x_position", "longitude", "lon"])
    y = _find_first(cols, ["y", "posy", "position_y", "y_position", "latitude", "lat"])
    speed = _find_first(cols, ["speed", "vitesse", "velocity", "kmh", "speed_kmh"])

    # Fallback heuristics by partial name
    if player is None:
        for c in cols:
            if any(k in c.lower() for k in ["player", "joueur", "athlete", "name"]):
                player = c
                break
    if time is None:
        for c in cols:
            if any(k in c.lower() for k in ["time", "stamp", "date"]):
                time = c
                break
    if x is None:
        for c in cols:
            if any(k in c.lower() for k in ["_x", "x_", " x", "lon"]):
                x = c
                break
    if y is None:
        for c in cols:
            if any(k in c.lower() for k in ["_y", "y_", " y", "lat"]):
                y = c
                break
    if speed is None:
        for c in cols:
            if any(k in c.lower() for k in ["speed", "vitesse", "km/h", "kmh", "velocity"]):
                speed = c
                break

    if all([player, time, x, y, speed]):
        return ColumnMapping(player=player, time=time, x=x, y=y, speed=speed)
    return None


def load_csv(uploaded_file, delimiter: str, decimal_comma: bool) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=delimiter)
    if decimal_comma:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace("nan", np.nan)
                )
    return df


def normalize_data(df: pd.DataFrame, mapping: ColumnMapping, options: DataOptions) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            mapping.player: "player",
            mapping.time: "time",
            mapping.x: "x",
            mapping.y: "y",
            mapping.speed: "speed",
        }
    )
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out["speed"] = pd.to_numeric(out["speed"], errors="coerce")
    if options.speed_unit == "m/s":
        out["speed"] = out["speed"] * 3.6

    if options.auto_scale_xy:
        x_min, x_max = out["x"].min(), out["x"].max()
        y_min, y_max = out["y"].min(), out["y"].max()
        if pd.notna(x_min) and pd.notna(x_max) and x_max > x_min:
            out["x"] = ((out["x"] - x_min) / (x_max - x_min)) * PITCH_LENGTH
        if pd.notna(y_min) and pd.notna(y_max) and y_max > y_min:
            out["y"] = ((out["y"] - y_min) / (y_max - y_min)) * PITCH_WIDTH

    if options.clamp_to_pitch:
        out["x"] = out["x"].clip(lower=0, upper=PITCH_LENGTH)
        out["y"] = out["y"].clip(lower=0, upper=PITCH_WIDTH)

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["player", "x", "y", "speed"])
    out = out.sort_values(["player", "time"])
    return out


def build_pitch_figure(length: float, width: float) -> go.Figure:
    fig = go.Figure()
    # Outer lines
    fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color="white", width=2))
    # Halfway line
    fig.add_shape(type="line", x0=length / 2, y0=0, x1=length / 2, y1=width, line=dict(color="white", width=2))
    # Center circle
    fig.add_shape(type="circle", x0=length / 2 - 9.15, y0=width / 2 - 9.15, x1=length / 2 + 9.15, y1=width / 2 + 9.15, line=dict(color="white", width=2))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        xaxis=dict(range=[0, length], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, width], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=20, b=10),
        height=650,
    )
    return fig


def add_heatmap(fig: go.Figure, player_df: pd.DataFrame) -> None:
    fig.add_trace(
        go.Histogram2d(
            x=player_df["x"],
            y=player_df["y"],
            colorscale="YlOrRd",
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="Densité"),
            xbins=dict(size=2.5),
            ybins=dict(size=2.5),
        )
    )


def compute_zone_occupancy(player_df: pd.DataFrame, length: float, width: float, bins_x: int = 12, bins_y: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = player_df.copy()
    if df.empty:
        x_edges = np.linspace(0, length, bins_x + 1)
        y_edges = np.linspace(0, width, bins_y + 1)
        return np.zeros((bins_y, bins_x)), x_edges, y_edges

    if "time" in df.columns and df["time"].notna().any():
        df = df.sort_values("time")
        dt = df["time"].diff().dt.total_seconds().fillna(0.2)
        dt = dt.clip(lower=0.0, upper=5.0)
    else:
        dt = pd.Series(np.full(len(df), 0.2), index=df.index)

    x_edges = np.linspace(0, length, bins_x + 1)
    y_edges = np.linspace(0, width, bins_y + 1)

    x_idx = np.digitize(df["x"], x_edges) - 1
    y_idx = np.digitize(df["y"], y_edges) - 1

    zone = np.zeros((bins_y, bins_x), dtype=float)
    for xi, yi, dti in zip(x_idx, y_idx, dt):
        if 0 <= xi < bins_x and 0 <= yi < bins_y:
            zone[yi, xi] += float(dti)
    return zone, x_edges, y_edges


def add_zone_heatmap(fig: go.Figure, zone: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=zone,
            colorscale="YlOrRd",
            opacity=0.7,
            colorbar=dict(title="Temps (s)"),
            hovertemplate="Zone x: %{x:.1f}m<br>Zone y: %{y:.1f}m<br>Temps: %{z:.1f}s<extra></extra>",
        )
    )


def sprint_vectors(player_df: pd.DataFrame, speed_threshold: float) -> pd.DataFrame:
    df = player_df.copy()
    if "time" in df.columns and df["time"].notna().any():
        df = df.sort_values("time")
    df["next_x"] = df["x"].shift(-1)
    df["next_y"] = df["y"].shift(-1)
    if "time" in df.columns and df["time"].notna().any():
        df["dt_s"] = df["time"].shift(-1).sub(df["time"]).dt.total_seconds()
    else:
        df["dt_s"] = 0.2
    df["distance_m"] = np.sqrt((df["next_x"] - df["x"]) ** 2 + (df["next_y"] - df["y"]) ** 2)
    df = df[df["speed"] >= speed_threshold].copy()
    df = df.dropna(subset=["next_x", "next_y", "speed", "distance_m"])
    df = df[df["distance_m"] > 0]
    return df


def add_sprint_arrows(fig: go.Figure, sprint_df: pd.DataFrame, max_arrows: int = 400) -> None:
    if sprint_df.empty:
        return

    if len(sprint_df) > max_arrows:
        sprint_df = sprint_df.nlargest(max_arrows, "speed")

    smin = float(sprint_df["speed"].min())
    smax = float(sprint_df["speed"].max())
    rng = smax - smin if smax > smin else 1.0

    for row in sprint_df.itertuples(index=False):
        scale_val = (float(row.speed) - smin) / rng
        color = sample_colorscale("Turbo", scale_val)[0]
        fig.add_annotation(
            x=row.next_x,
            y=row.next_y,
            ax=row.x,
            ay=row.y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=1.0 + min(4.0, float(row.distance_m) / 4.0),
            arrowcolor=color,
            opacity=0.9,
            hovertext=f"Vitesse: {row.speed:.2f} km/h<br>Distance: {row.distance_m:.2f} m",
        )
    fig.add_trace(
        go.Scatter(
            x=sprint_df["x"],
            y=sprint_df["y"],
            mode="markers",
            marker=dict(
                size=np.clip(sprint_df["distance_m"], 5, 14),
                color=sprint_df["speed"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Vitesse sprint (km/h)"),
            ),
            name="Départs sprint",
            hovertemplate="Vitesse: %{marker.color:.2f} km/h<br>Distance: %{marker.size:.2f} (taille)<extra></extra>",
        )
    )


def build_report_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "events": len(g),
                "max_speed_kmh": g["speed"].max(),
                "avg_speed_kmh": g["speed"].mean(),
                f"sprints_>{threshold:.0f}": int((g["speed"] >= threshold).sum()),
            }
        )

    report = df.groupby("player", dropna=False).apply(_agg).reset_index()
    return report.sort_values("max_speed_kmh", ascending=False)


def main() -> None:
    st.set_page_config(page_title="Rapports événements GPS", layout="wide")
    st.title("📊 Rapport événementiel GPS (Heatmap + Sprints)")
    st.caption("Charge un CSV événementiel avec au minimum: joueur, temps, x, y, vitesse.")

    with st.sidebar:
        st.header("Paramètres")
        delimiter = st.selectbox("Séparateur CSV", [",", ";", "\\t"], index=0)
        decimal_comma = st.checkbox("Décimales avec virgule (,)", value=True)
        sprint_threshold = st.slider("Seuil sprint (km/h)", 10.0, 40.0, DEFAULT_SPRINT_THRESHOLD, 0.5)
        speed_unit = st.selectbox("Unité vitesse", ["km/h", "m/s"], index=0)
        auto_scale_xy = st.checkbox("Auto-remettre X/Y à l'échelle du terrain", value=True)
        clamp_to_pitch = st.checkbox("Forcer les positions dans le terrain", value=True)
        pitch_length = st.number_input("Longueur terrain (m)", value=PITCH_LENGTH, min_value=40.0, max_value=130.0)
        pitch_width = st.number_input("Largeur terrain (m)", value=PITCH_WIDTH, min_value=20.0, max_value=100.0)

    uploaded_files = st.file_uploader("CSV tracking / événements (1 fichier = 1 joueur)", type=["csv"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Ajoute un ou plusieurs CSV pour générer le rapport.")
        return

    frames: list[pd.DataFrame] = []
    for f in uploaded_files:
        df_f = load_csv(f, delimiter, decimal_comma)
        df_f["__source_file"] = f.name
        frames.append(df_f)
    raw = pd.concat(frames, ignore_index=True)
    st.write("Aperçu brut", raw.head(5))

    auto_mapping = suggest_mapping(raw)
    st.subheader("Mapping colonnes")
    cols = list(raw.columns)
    default_player = cols.index(auto_mapping.player) if auto_mapping and auto_mapping.player in cols else 0
    default_time = cols.index(auto_mapping.time) if auto_mapping and auto_mapping.time in cols else 0
    default_x = cols.index(auto_mapping.x) if auto_mapping and auto_mapping.x in cols else 0
    default_y = cols.index(auto_mapping.y) if auto_mapping and auto_mapping.y in cols else 0
    default_speed = cols.index(auto_mapping.speed) if auto_mapping and auto_mapping.speed in cols else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        player_col = st.selectbox("Joueur", cols, index=default_player)
    with c2:
        time_col = st.selectbox("Temps", cols, index=default_time)
    with c3:
        x_col = st.selectbox("X", cols, index=default_x)
    with c4:
        y_col = st.selectbox("Y", cols, index=default_y)
    with c5:
        speed_col = st.selectbox("Vitesse", cols, index=default_speed)

    mapping = ColumnMapping(player=player_col, time=time_col, x=x_col, y=y_col, speed=speed_col)
    options = DataOptions(speed_unit=speed_unit, auto_scale_xy=auto_scale_xy, clamp_to_pitch=clamp_to_pitch)
    data = normalize_data(raw, mapping, options)
    if data.empty:
        st.error("Aucune donnée exploitable après nettoyage. Vérifie le mapping et le format numérique.")
        return

    players = sorted(data["player"].dropna().unique().tolist())
    selected_player = st.selectbox("Choisir un joueur", players, index=0)
    player_df = data[data["player"] == selected_player].copy()
    sprints = sprint_vectors(player_df, sprint_threshold)
    max_arrows = st.slider("Nombre max de flèches sprint", min_value=50, max_value=1200, value=400, step=50)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Événements", f"{len(player_df):,}".replace(",", " "))
    k2.metric("Sprints détectés", f"{len(sprints):,}".replace(",", " "))
    k3.metric("Vitesse max", f"{player_df['speed'].max():.2f} km/h")
    k4.metric("Vitesse moyenne", f"{player_df['speed'].mean():.2f} km/h")

    fig = build_pitch_figure(length=pitch_length, width=pitch_width)
    zone, x_edges, y_edges = compute_zone_occupancy(player_df, pitch_length, pitch_width, bins_x=12, bins_y=8)
    add_zone_heatmap(fig, zone, x_edges, y_edges)
    add_sprint_arrows(fig, sprints, max_arrows=max_arrows)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Joueur sélectionné: {selected_player} | "
        f"Plage vitesse: {player_df['speed'].min():.2f} à {player_df['speed'].max():.2f} km/h."
    )

    st.subheader("Résumé multi-joueurs")
    report = build_report_table(data, sprint_threshold)
    st.dataframe(report, use_container_width=True)

    csv_data = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger le résumé (CSV)",
        data=csv_data,
        file_name="report_summary_players.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
