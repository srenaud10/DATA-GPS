from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class ColumnMapping:
    player: str
    time: str
    x: str
    y: str
    speed: str


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


def normalize_data(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
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
        go.Histogram2dContour(
            x=player_df["x"],
            y=player_df["y"],
            colorscale="YlOrRd",
            contours=dict(showlabels=False),
            ncontours=18,
            opacity=0.9,
            showscale=True,
            colorbar=dict(title="Densité"),
            hovertemplate="x: %{x:.1f}m<br>y: %{y:.1f}m<extra></extra>",
        )
    )


def sprint_vectors(player_df: pd.DataFrame, speed_threshold: float) -> pd.DataFrame:
    df = player_df.copy()
    df = df[df["speed"] >= speed_threshold].copy()
    if df.empty:
        return df
    df["next_x"] = df["x"].shift(-1)
    df["next_y"] = df["y"].shift(-1)
    df = df.dropna(subset=["next_x", "next_y"])
    return df


def add_sprint_arrows(fig: go.Figure, sprint_df: pd.DataFrame) -> None:
    if sprint_df.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=sprint_df["x"],
            y=sprint_df["y"],
            mode="markers",
            marker=dict(size=7, color="#34D399"),
            name="Départs sprint",
            hovertemplate="Vitesse: %{customdata:.2f} km/h<extra></extra>",
            customdata=sprint_df["speed"],
        )
    )
    for row in sprint_df.itertuples():
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
            arrowwidth=1.5,
            arrowcolor="#34D399",
            opacity=0.85,
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
        sprint_threshold = st.slider("Seuil sprint (km/h)", 15.0, 40.0, DEFAULT_SPRINT_THRESHOLD, 0.5)
        pitch_length = st.number_input("Longueur terrain (m)", value=PITCH_LENGTH, min_value=40.0, max_value=130.0)
        pitch_width = st.number_input("Largeur terrain (m)", value=PITCH_WIDTH, min_value=20.0, max_value=100.0)

    uploaded = st.file_uploader("CSV tracking / événements", type=["csv"])
    if uploaded is None:
        st.info("Ajoute un fichier CSV pour générer le rapport.")
        return

    raw = load_csv(uploaded, delimiter, decimal_comma)
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
    data = normalize_data(raw, mapping)
    if data.empty:
        st.error("Aucune donnée exploitable après nettoyage. Vérifie le mapping et le format numérique.")
        return

    players = sorted(data["player"].dropna().unique().tolist())
    selected_player = st.selectbox("Choisir un joueur", players, index=0)
    player_df = data[data["player"] == selected_player].copy()
    sprints = sprint_vectors(player_df, sprint_threshold)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Événements", f"{len(player_df):,}".replace(",", " "))
    k2.metric("Sprints détectés", f"{len(sprints):,}".replace(",", " "))
    k3.metric("Vitesse max", f"{player_df['speed'].max():.2f} km/h")
    k4.metric("Vitesse moyenne", f"{player_df['speed'].mean():.2f} km/h")

    fig = build_pitch_figure(length=pitch_length, width=pitch_width)
    add_heatmap(fig, player_df)
    add_sprint_arrows(fig, sprints)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Résumé multi-joueurs")
    report = build_report_table(data, sprint_threshold)
    st.dataframe(report, use_container_width=True)

    csv_data = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger le résumé (CSV)",
        data=csv_data,
        file_name=f"report_summary_{Path(uploaded.name).stem}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
