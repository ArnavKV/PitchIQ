"""Fast PitchIQ player comparison. Uses cleaned Transfermarkt/Understat data.

Searches compact season summaries without modifying source datasets.
"""
import pandas as pd
import altair as alt
import streamlit as st

from pitchiq.ui import PALETTE, configure_page, download_csv, footer, page_header
from pitchiq.data import load_player_frames, player_data_versions, using_demo

configure_page("Player comparison")
page_header("Player comparison", "Put two players side by side. Compare scoring, creativity, and performance across recorded seasons.", "Explore / Player stats")

KNOWN = {
    "Lionel Messi": ("transfermarkt:28003", "understat:2097"),
    "Cristiano Ronaldo": ("transfermarkt:8198", "understat:2371"),
    "Erling Haaland": ("transfermarkt:418560", "understat:8260"),
    "Kylian Mbappé": ("transfermarkt:342229", "understat:3423"),
    "Lamine Yamal": ("transfermarkt:937958", "understat:11527"),
}
LEAGUES = {
    "ES1": "La Liga", "La_liga": "La Liga",
    "GB1": "Premier League", "EPL": "Premier League",
    "L1": "Bundesliga", "Bundesliga": "Bundesliga",
    "FR1": "Ligue 1", "Ligue_1": "Ligue 1",
    "IT1": "Serie A", "Serie_A": "Serie A",
    "CL": "Champions League", "RFPL": "Russian Premier League",
}
METRICS = [
    ("Games", "Appearances", "core"),
    ("Minutes", "Minutes played", "core"),
    ("Goals", "Goals", "core"),
    ("Assists", "Assists", "core"),
    ("GoalsPer90", "Goals per 90", "core"),
    ("AssistsPer90", "Assists per 90", "core"),
    ("Shots", "Shots", "advanced"),
    ("xG", "Expected goals (xG)", "advanced"),
    ("xA", "Expected assists (xA)", "advanced"),
    ("GoalsMinusxG", "Goals − reference xG" if using_demo() else "Understat goals − xG", "advanced"),
    ("Conversion", "Shot conversion (%)" if using_demo() else "Understat shot conversion (%)", "advanced"),
]

@st.cache_resource(show_spinner="Preparing searchable player index…")
def prepare_index(tm_mtime, us_mtime):
    """Expensive work runs once per data-file version, not on every widget change."""
    tm, us = load_player_frames()

    def normalize(df, provider):
        year_field = "season" if provider == "tm" else "Season"
        if year_field not in df.columns and provider == "us" and "year" in df.columns:
            year_field = "year"
        league_field = "competition_id" if provider == "tm" else "league"
        needed = {"Player", "ProviderPlayerID", "Games", "Goals", "Assists", "Minutes", year_field, league_field}
        if provider == "us":
            needed.update(["Shots", "xG", "xA"])
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing {provider} columns: {sorted(missing)}")
        df["Year"] = pd.to_numeric(df[year_field], errors="coerce")
        df = df.dropna(subset=["Year"]).copy()
        df["Year"] = df["Year"].astype(int)
        df["League"] = df[league_field].astype(str).replace(LEAGUES)
        df["PID"] = df["ProviderPlayerID"].astype(str)
        for column in ["Games", "Goals", "Assists", "Minutes", "Shots", "xG", "xA"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    tm = normalize(tm, "tm")
    us = normalize(us, "us")
    tm_ids = set(tm["PID"].unique())
    us_ids = set(us["PID"].unique())

    known_tm = {}
    known_us = {}
    labels = {}
    for name, (tm_id, us_id) in KNOWN.items():
        if tm_id in tm_ids or us_id in us_ids:
            key = "verified:" + name
            labels[key] = name
            known_tm[tm_id] = key
            known_us[us_id] = key

    # Unlike the old page, this does NOT loop over every full player group.
    tm_names = tm[["PID", "Player"]].drop_duplicates(subset="PID")
    us_names = us[["PID", "Player"]].drop_duplicates(subset="PID")
    for pid, name in tm_names.itertuples(index=False, name=None):
        labels.setdefault(known_tm.get(pid, pid), str(name))
    for pid, name in us_names.itertuples(index=False, name=None):
        labels.setdefault(known_us.get(pid, pid), str(name))

    tm["PersonKey"] = tm["PID"].replace(known_tm)
    us["PersonKey"] = us["PID"].replace(known_us)

    # Summarize once. All subsequent filtering works over compact grouped tables.
    grouping = ["PersonKey", "Year", "League"]
    core_fields = ["Games", "Goals", "Assists", "Minutes"]
    tm_small = tm.groupby(grouping, as_index=False, sort=False)[core_fields].sum(min_count=1)
    us_small = us.groupby(grouping, as_index=False, sort=False)[core_fields + ["Shots", "xG", "xA"]].sum(min_count=1)
    search_index = pd.DataFrame([{"Key": k, "Name": v, "Search": v.casefold()} for k, v in labels.items()])
    return tm_small, us_small, search_index


try:
    tm, us, player_index = prepare_index(*player_data_versions())
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    leagues = ["All competitions"] + sorted(set(tm["League"]) | set(us["League"]))
    selected_league = st.selectbox("Competition", leagues)
with col2:
    years = ["All seasons"] + sorted(set(tm["Year"]) | set(us["Year"]), reverse=True)
    selected_year = st.selectbox(
        "Season", years,
        format_func=lambda v: f"{v}–{str(v + 1)[-2:]}" if isinstance(v, int) else v,
    )


def filtered(df):
    if selected_league != "All competitions":
        df = df.loc[df["League"].eq(selected_league)]
    if selected_year != "All seasons":
        df = df.loc[df["Year"].eq(selected_year)]
    return df


tm_f, us_f = filtered(tm), filtered(us)
available = set(tm_f["PersonKey"]) | set(us_f["PersonKey"])
if len(available) < 2:
    st.warning("Fewer than two players for this season and competition.")
    st.stop()

names = player_index[player_index["Key"].isin(available)].copy()
# Featured players are immediately selectable without loading 30,000 choices.
featured = ["verified:" + name for name in KNOWN]

st.subheader("Choose players")
a_col, b_col = st.columns(2)


def select_player(parent, number, preferred):
    with parent:
        query = st.text_input(
            f"Search player {number}",
            placeholder="Type a name to narrow the list…",
            key=f"player_search_{number}",
        ).strip().casefold()
        if query:
            result = names[names["Search"].str.contains(query, regex=False, na=False)]
            result = result.sort_values(["Name", "Key"]).head(150)
        else:
            result = names[names["Key"].isin(featured)].copy()
            if result.empty:
                result = names.sort_values(["Name", "Key"]).head(150)
        options = result["Key"].tolist()
        if not options:
            st.info("No matching player. Try a different name.")
            return None
        lookup = dict(zip(result["Key"], result["Name"]))
        default = "verified:" + preferred
        # A unique key per search/league/season avoids invalid widget state.
        return st.selectbox(
            f"Player {number}", options,
            index=options.index(default) if default in options else min(number - 1, len(options) - 1),
            format_func=lambda key: f"{lookup[key]} — {key}" if not key.startswith("verified:") else lookup[key],
            key=f"player_choice_{number}_{selected_league}_{selected_year}_{query}",
        )


a = select_player(a_col, 1, "Lionel Messi")
b = select_player(b_col, 2, "Cristiano Ronaldo")
if a is None or b is None:
    st.stop()
if a == b:
    st.info("Select two different players to compare.")
    st.stop()


def stats_for(key):
    core = tm_f[tm_f["PersonKey"].eq(key)]
    advanced = us_f[us_f["PersonKey"].eq(key)]
    main = core if not core.empty else advanced
    result = {
        "CoreSource": "Synthetic sample" if using_demo() else "Transfermarkt" if not core.empty else "Understat" if not main.empty else "N/A",
        "AdvancedSource": "Synthetic sample" if using_demo() else "Understat" if not advanced.empty else "N/A",
        "CoreRows": len(main), "AdvancedRows": len(advanced),
    }
    for metric in ["Games", "Goals", "Assists", "Minutes"]:
        result[metric] = main[metric].sum(min_count=1) if not main.empty else float("nan")
    for metric in ["Shots", "xG", "xA"]:
        result[metric] = advanced[metric].sum(min_count=1) if not advanced.empty else float("nan")
    mins = result["Minutes"]
    for metric in ["Goals", "Assists"]:
        result[metric + "Per90"] = result[metric] * 90 / mins if pd.notna(mins) and mins > 0 else float("nan")
    if not advanced.empty:
        understat_goals = advanced["Goals"].sum(min_count=1)
        result["GoalsMinusxG"] = understat_goals - result["xG"]
        result["Conversion"] = 100 * understat_goals / result["Shots"] if result["Shots"] > 0 else float("nan")
    else:
        result["GoalsMinusxG"] = float("nan")
        result["Conversion"] = float("nan")
    return result


sa, sb = stats_for(a), stats_for(b)
name_map = dict(zip(player_index["Key"], player_index["Name"]))
name_a, name_b = name_map[a], name_map[b]


def fmt(value, metric):
    if pd.isna(value):
        return "N/A"
    if metric in ["Games", "Minutes", "Goals", "Assists", "Shots"]:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


st.subheader("Head-to-head statistics")
for column, name, stats in zip(st.columns(2), (name_a, name_b), (sa, sb)):
    with column, st.container(border=True):
        st.subheader(name)
        goals, assists = st.columns(2)
        goals.metric("Goals", fmt(stats["Goals"], "Goals"))
        assists.metric("Assists", fmt(stats["Assists"], "Assists"))
        st.caption(f"{stats['CoreSource']} · {fmt(stats['Minutes'], 'Minutes')} minutes recorded")
rows = []
for metric, label, kind in METRICS:
    source_key = "CoreSource" if kind == "core" else "AdvancedSource"
    pa, pb = sa[source_key], sb[source_key]
    rows.append({
        "Metric": label,
        "Player 1": fmt(sa[metric], metric),
        "Player 2": fmt(sb[metric], metric),
        "Source": pa if pa == pb else f"{pa} / {pb}",
    })
comparison = pd.DataFrame(rows).rename(columns={"Player 1": name_a, "Player 2": name_b})
st.dataframe(comparison, hide_index=True, width="stretch")
download_csv(comparison, "pitchiq_player_comparison.csv", "Download comparison")
source_note = "All totals and xG values are fictional." if using_demo() else "Core totals prefer Transfermarkt; xG/xA/shots use Understat."
st.caption(f"Player 1: {name_a} · Player 2: {name_b}. {source_note}")

valid = [(metric, label) for metric, label, _ in METRICS if pd.notna(sa[metric]) or pd.notna(sb[metric])]
if valid:
    metric_labels = dict(valid)
    selected_metric = st.selectbox("Chart metric", [m for m, _ in valid], format_func=lambda k: metric_labels[k])
    plot = pd.DataFrame({
        "Player": [name_a, name_b],
        "Value": [sa[selected_metric], sb[selected_metric]],
    }).dropna(subset=["Value"])
    if not plot.empty:
        chart = alt.Chart(plot).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Player:N", title=None, sort=None),
            y=alt.Y("Value:Q", title=metric_labels[selected_metric]),
            color=alt.Color("Player:N", legend=None, scale=alt.Scale(domain=[name_a, name_b], range=[PALETTE[0], PALETTE[2]])),
            tooltip=["Player:N", alt.Tooltip("Value:Q", format=".2f")],
        ).properties(height=280)
        st.altair_chart(chart, width="stretch")

with st.expander("Coverage and source limitations"):
    for name, record in [(name_a, sa), (name_b, sb)]:
        st.write(f"**{name}** — core: {record['CoreSource']} ({record['CoreRows']} rows); "
                 f"advanced: {record['AdvancedSource']} ({record['AdvancedRows']} rows)")
    if using_demo():
        st.write("The same generated player IDs link the two fictional tables. These values do not come from Transfermarkt or Understat and do not describe real players.")
    else:
        st.write("Only the five explicitly reviewed player identities are linked across providers. "
                 "Other players remain provider-specific to avoid mistaken identity matches.")
        st.write("Provider statistics may differ. Goals−xG and conversion are computed solely "
                 "from Understat numbers. The datasets do not guarantee complete career coverage.")
        st.write("Review upstream licensing and redistribution terms before publishing raw data.")

footer("Player statistics: Transfermarkt / Understat. Recorded coverage varies by provider.")
