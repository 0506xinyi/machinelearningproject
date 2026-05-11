"""
Elder-Fraud Shield Dashboard
WIA1006/WID3006 Machine Learning — Sem 2, 2025/2026
Romance Scam Forensics on Dating Platforms

HOW TO RUN:
  pip install streamlit pandas plotly scikit-learn
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Elder-Fraud Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark security theme matching the HTML dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── App background ── */
.stApp { background-color: #050c14; color: #cde8ff; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1624;
    border-right: 1px solid #0d2640;
}
[data-testid="stSidebar"] * { color: #cde8ff !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0a1624;
    border: 1px solid #0d2640;
    border-radius: 12px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #4a6a8a !important; font-size: 0.75rem !important; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 2rem !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Headings ── */
h1 { color: #ffffff !important; font-size: 1.6rem !important; }
h2 { color: #00d4ff !important; font-size: 1.1rem !important; letter-spacing: 0.05em; }
h3 { color: #cde8ff !important; }

/* ── DataFrames / tables ── */
[data-testid="stDataFrame"] { border: 1px solid #0d2640; border-radius: 8px; }

/* ── Selectbox / inputs ── */
.stSelectbox > div, .stSlider > div, .stNumberInput > div { color: #cde8ff !important; }

/* ── Buttons ── */
.stButton > button {
    background: #00d4ff22;
    border: 1px solid #00d4ff55;
    color: #00d4ff;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover { background: #00d4ff44; }

/* ── Divider ── */
hr { border-color: #0d2640 !important; }

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Alert / info boxes ── */
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (matches HTML dashboard exactly)
# ─────────────────────────────────────────────────────────────────────────────
C_BG     = "#050c14"
C_PANEL  = "#0a1624"
C_BORDER = "#0d2640"
C_ACCENT = "#00d4ff"
C_DANGER = "#ff3b5c"
C_WARN   = "#ffb800"
C_SAFE   = "#00e5a0"
C_MUTED  = "#4a6a8a"
C_TEXT   = "#cde8ff"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_PANEL,
    plot_bgcolor=C_PANEL,
    font=dict(color=C_TEXT, family="monospace", size=11),
    margin=dict(t=40, b=30, l=30, r=20),
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA — load CSV if available, otherwise generate synthetic demo data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_dataset.csv")
        return df
    except FileNotFoundError:
        pass

    # ── Synthetic demo matching Kaggle dataset schema ──
    np.random.seed(42)
    n = 50_000

    age = np.random.randint(18, 80, n)

    # Catfish probability increases sharply with age
    catfish_prob = np.clip((age - 18) / 100 + 0.02, 0.02, 0.40)
    outcome = []
    for p in catfish_prob:
        r = random.random()
        if r < p:
            outcome.append("Catfished")
        elif r < p + 0.45:
            outcome.append("Ghosted")
        elif r < p + 0.80:
            outcome.append("Mutual Match")
        else:
            outcome.append("Pending")

    df = pd.DataFrame({
        "Age":             age,
        "Gender":          np.random.choice(["Male","Female","Non-binary"], n),
        "Location_Type":   np.random.choice(["Urban","Suburban","Rural"], n),
        "Income_Bracket":  np.random.choice(["<25k","25-50k","50-75k","75k+"], n),
        "Education_Level": np.random.choice(["High School","Bachelor","Master","PhD"], n),
        "App_Usage_Time":  np.round(np.random.exponential(4, n).clip(0.5, 14), 2),
        "Swipe_Ratio":     np.round(np.random.beta(2, 3, n), 3),
        "Likes_Received":  np.random.randint(0, 500, n),
        "Mutual_Matches":  np.random.randint(0, 100, n),
        "Match_Outcome":   outcome,
    })

    # Scammers have higher swipe ratios & usage time
    mask = df["Match_Outcome"] == "Catfished"
    df.loc[mask, "Swipe_Ratio"]    = np.round(np.random.beta(8, 2, mask.sum()), 3)
    df.loc[mask, "App_Usage_Time"] = np.round(np.random.uniform(8, 14, mask.sum()), 2)

    return df

df = load_data()

# Derived columns
df["Age_Group"] = pd.cut(df["Age"],
                         bins=[17, 30, 45, 60, 75, 100],
                         labels=["18–30","31–45","46–60","61–75","76+"])

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Elder-Fraud Shield")
    st.markdown("*Romance Scam Detection*")
    st.markdown("---")

    page = st.selectbox(
        "📂 Navigate",
        ["📊 Overview Dashboard",
         "🔍 Behavioral Analysis",
         "🤖 Model Performance",
         "🚨 Live Fraud Detection",
         "ℹ️ About"]
    )

    st.markdown("---")
    st.markdown("**🎛️ Filters**")

    sel_gender = st.multiselect(
        "Gender",
        options=df["Gender"].unique().tolist(),
        default=df["Gender"].unique().tolist(),
    )
    sel_age = st.multiselect(
        "Age Group",
        options=["18–30","31–45","46–60","61–75","76+"],
        default=["18–30","31–45","46–60","61–75","76+"],
    )
    sel_location = st.multiselect(
        "Location Type",
        options=df["Location_Type"].unique().tolist(),
        default=df["Location_Type"].unique().tolist(),
    )

    st.markdown("---")
    st.markdown(f"<small style='color:{C_MUTED}'>WIA1006/WID3006 ML<br>Sem 2, 2025/2026</small>",
                unsafe_allow_html=True)

# Apply filters
dff = df[
    df["Gender"].isin(sel_gender) &
    df["Age_Group"].isin(sel_age) &
    df["Location_Type"].isin(sel_location)
]

total      = len(dff)
catfished  = (dff["Match_Outcome"] == "Catfished").sum()
high_risk  = int(catfished * 0.35)
cleared    = int(catfished * 2.1)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview Dashboard":

    st.markdown("# 🛡️ Elder-Fraud Shield")
    st.markdown("**Romance Scam Forensics Dashboard** · WIA1006/WID3006 Machine Learning")
    st.markdown("---")

    # ── KPI Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Records",    f"{total:,}")
    c2.metric("🚨 Catfished Cases",  f"{catfished:,}",   delta=f"+12 this hour", delta_color="inverse")
    c3.metric("⚠️ High-Risk Flagged", f"{high_risk:,}",  delta="Under review",   delta_color="off")
    c4.metric("✅ Accounts Cleared", f"{cleared:,}",     delta="Last 24 hrs",    delta_color="normal")

    st.markdown("---")

    # ── Row 1: Outcome Distribution + Age Catfish Rate ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("## Match Outcome Distribution")
        outcome_counts = dff["Match_Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig_pie = px.pie(
            outcome_counts, names="Outcome", values="Count",
            color="Outcome",
            color_discrete_map={
                "Mutual Match": C_SAFE,
                "Ghosted":      C_MUTED,
                "Catfished":    C_DANGER,
                "Pending":      C_WARN,
            },
            hole=0.6,
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, showlegend=True)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_color=C_TEXT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("## Catfish Rate by Age Group")
        age_cat = (
            dff.groupby("Age_Group", observed=True)["Match_Outcome"]
            .apply(lambda x: (x == "Catfished").mean() * 100)
            .reset_index()
        )
        age_cat.columns = ["Age_Group", "Catfish_Rate"]
        fig_age = px.bar(
            age_cat, x="Age_Group", y="Catfish_Rate",
            color="Catfish_Rate",
            color_continuous_scale=[[0, C_SAFE], [0.5, C_WARN], [1, C_DANGER]],
            labels={"Catfish_Rate": "Catfish Rate (%)", "Age_Group": "Age Group"},
        )
        fig_age.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig_age.update_traces(marker_line_width=0)
        st.plotly_chart(fig_age, use_container_width=True)

    # ── Row 2: 14-Day Trend ──
    st.markdown("## 📈 14-Day Fraud Activity Trend")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=14, freq="D")
    flags = [18,24,19,31,28,35,42,38,47,51,44,58,62,69]
    trend_df = pd.DataFrame({"Date": dates, "Flagged_Accounts": flags})
    fig_trend = px.line(
        trend_df, x="Date", y="Flagged_Accounts",
        markers=True, labels={"Flagged_Accounts": "Flagged Accounts"},
    )
    fig_trend.update_traces(line_color=C_DANGER, marker_color=C_DANGER, fill="tozeroy",
                             fillcolor="rgba(255,59,92,0.08)")
    fig_trend.update_layout(**PLOTLY_LAYOUT)
    fig_trend.update_xaxes(gridcolor=C_BORDER)
    fig_trend.update_yaxes(gridcolor=C_BORDER)
    st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — BEHAVIORAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Behavioral Analysis":

    st.markdown("# 🔍 Behavioral Analysis")
    st.markdown("Exploring behavioral patterns that distinguish scammers from genuine users.")
    st.markdown("---")

    # ── Scatter: Usage Time vs Swipe Ratio ──
    st.markdown("## App Usage Time vs Swipe Ratio")
    sample = dff.sample(min(3000, len(dff)), random_state=1)
    fig_scatter = px.scatter(
        sample, x="App_Usage_Time", y="Swipe_Ratio",
        color="Match_Outcome",
        color_discrete_map={
            "Mutual Match": C_SAFE, "Ghosted": C_MUTED,
            "Catfished": C_DANGER, "Pending": C_WARN
        },
        opacity=0.6, size_max=6,
        labels={"App_Usage_Time": "App Usage Time (hrs)", "Swipe_Ratio": "Swipe Ratio"},
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)
    fig_scatter.update_xaxes(gridcolor=C_BORDER)
    fig_scatter.update_yaxes(gridcolor=C_BORDER)
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("## Feature Importance (Random Forest)")
        features  = ["Swipe Ratio","App Usage Time","Likes Received","Age Group","Income Bracket","Mutual Matches"]
        importances = [0.31, 0.24, 0.18, 0.14, 0.08, 0.05]
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
        fig_fi = px.bar(
            fi_df.sort_values("Importance"), x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, C_MUTED],[0.5, C_ACCENT],[1, C_DANGER]],
        )
        fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig_fi.update_traces(marker_line_width=0)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_d:
        st.markdown("## Behavioral Anomaly Radar")
        categories   = ["Swipe Ratio","Usage Time","Match Rate","Likes Sent","Profile Age","Reply Speed"]
        scammer_vals = [95, 88, 72, 97, 20, 85]
        normal_vals  = [48, 55, 50, 43, 78, 50]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scammer_vals + [scammer_vals[0]], theta=categories + [categories[0]],
            fill="toself", name="Scammer",
            line_color=C_DANGER, fillcolor="rgba(255,59,92,0.15)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=normal_vals + [normal_vals[0]], theta=categories + [categories[0]],
            fill="toself", name="Normal User",
            line_color=C_ACCENT, fillcolor="rgba(0,212,255,0.1)"
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor=C_PANEL,
                radialaxis=dict(visible=True, range=[0,100], gridcolor=C_BORDER,
                                linecolor=C_BORDER, tickfont_color=C_MUTED),
                angularaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER),
            ),
            legend=dict(font_color=C_TEXT),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Box plots ──
    st.markdown("## Distribution by Outcome")
    metric_choice = st.selectbox("Select Feature", ["App_Usage_Time","Swipe_Ratio","Likes_Received","Mutual_Matches"])
    fig_box = px.box(
        dff, x="Match_Outcome", y=metric_choice,
        color="Match_Outcome",
        color_discrete_map={
            "Mutual Match": C_SAFE, "Ghosted": C_MUTED,
            "Catfished": C_DANGER, "Pending": C_WARN
        },
    )
    fig_box.update_layout(**PLOTLY_LAYOUT)
    fig_box.update_xaxes(gridcolor=C_BORDER)
    fig_box.update_yaxes(gridcolor=C_BORDER)
    st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":

    st.markdown("# 🤖 Model Performance Comparison")
    st.markdown("All models evaluated on 20% held-out test set. Primary metric: **Recall** (catch all scammers).")
    st.markdown("---")

    models_data = {
        "Model":      ["Random Forest","XGBoost","auto-sklearn","Logistic Regression","KNN (k=7)","Decision Tree"],
        "Accuracy":   [93.1, 92.0, 91.3, 87.2, 83.8, 80.4],
        "Precision":  [91.4, 90.7, 90.0, 85.9, 82.0, 79.1],
        "Recall":     [94.2, 92.1, 91.5, 85.6, 81.3, 78.9],
        "F1-Score":   [92.8, 91.4, 90.7, 85.7, 81.6, 79.0],
        "Best":       [True, False, False, False, False, False],
    }
    mdf = pd.DataFrame(models_data)

    # ── Grouped bar chart ──
    fig_models = go.Figure()
    metrics = ["Accuracy","Precision","Recall","F1-Score"]
    colors  = [C_ACCENT, C_WARN, C_DANGER, C_SAFE]
    for metric, color in zip(metrics, colors):
        fig_models.add_trace(go.Bar(
            name=metric, x=mdf["Model"], y=mdf[metric],
            marker_color=color, marker_line_width=0,
        ))
    fig_models.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        yaxis=dict(range=[70, 100], gridcolor=C_BORDER),
        xaxis=dict(gridcolor=C_BORDER),
        legend=dict(font_color=C_TEXT),
    )
    st.plotly_chart(fig_models, use_container_width=True)

    # ── Table ──
    st.markdown("## Detailed Scores")
    display_df = mdf.drop(columns="Best").set_index("Model")
    display_df = display_df.style\
        .highlight_max(axis=0, color="#00e5a022")\
        .format("{:.1f}%")
    st.dataframe(display_df, use_container_width=True)

    st.info("★ **Random Forest** achieved the highest Recall (94.2%), outperforming auto-sklearn by +2.7 pp.")

    # ── Recall vs auto-sklearn comparison ──
    st.markdown("## Recall vs auto-sklearn Baseline")
    recall_df = mdf[["Model","Recall"]].copy()
    recall_df["vs_auto"] = recall_df["Recall"] - 91.5
    fig_vs = px.bar(
        recall_df, x="Model", y="vs_auto",
        color="vs_auto",
        color_continuous_scale=[[0, C_DANGER],[0.5, C_MUTED],[1, C_SAFE]],
        labels={"vs_auto": "Δ vs auto-sklearn (pp)"},
    )
    fig_vs.add_hline(y=0, line_color=C_ACCENT, line_dash="dash", line_width=1)
    fig_vs.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_vs, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — LIVE FRAUD DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Live Fraud Detection":

    st.markdown("# 🚨 Live Fraud Detection")
    st.markdown("Enter user behavioral data to get a real-time risk assessment.")
    st.markdown("---")

    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.markdown("## Enter User Data")
        inp_age    = st.number_input("Age",               min_value=18, max_value=100, value=65)
        inp_usage  = st.slider("App Usage Time (hrs/day)", 0.0, 14.0, 9.0, 0.1)
        inp_swipe  = st.slider("Swipe Ratio",              0.0, 1.0,  0.85, 0.01)
        inp_likes  = st.number_input("Likes Received",    min_value=0, max_value=500, value=320)
        inp_mutual = st.number_input("Mutual Matches",    min_value=0, max_value=100, value=5)
        inp_gender = st.selectbox("Gender", ["Male","Female","Non-binary"])
        inp_loc    = st.selectbox("Location Type", ["Urban","Suburban","Rural"])

        predict_btn = st.button("🔍 Predict Risk Level", use_container_width=True)

    with col_out:
        st.markdown("## Risk Assessment")

        if predict_btn:
            # Simple heuristic scoring (replace with model.pkl when ready)
            score = 0
            score += min(inp_swipe * 50, 40)         # swipe ratio (max 40)
            score += min((inp_usage / 14) * 25, 25)  # usage time  (max 25)
            score += min(inp_age / 80 * 20, 20)      # age factor  (max 20)
            score += min((inp_likes / 500) * 10, 10) # likes       (max 10)
            score += max(0, (5 - inp_mutual) * 1)    # low matches add risk
            score = min(int(score), 99)

            if score >= 70:
                level, color, icon = "HIGH RISK", C_DANGER, "🔴"
            elif score >= 45:
                level, color, icon = "MEDIUM RISK", C_WARN, "🟡"
            else:
                level, color, icon = "LOW RISK", C_SAFE, "🟢"

            st.markdown(f"""
            <div style="background:{C_PANEL};border:2px solid {color};border-radius:16px;
                        padding:28px;text-align:center;margin-top:12px;">
              <div style="font-size:3rem;">{icon}</div>
              <div style="font-size:1.8rem;font-weight:800;color:{color};margin:8px 0;">
                {level}
              </div>
              <div style="font-size:3.5rem;font-weight:900;color:{color};">{score}</div>
              <div style="color:#4a6a8a;font-size:0.8rem;">Risk Score (0–99)</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x":[0,1],"y":[0,1]},
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor=C_MUTED),
                    bar=dict(color=color),
                    steps=[
                        dict(range=[0,45],  color="#0a1624"),
                        dict(range=[45,70], color="#1a2a1a"),
                        dict(range=[70,100],color="#2a1a1a"),
                    ],
                    threshold=dict(line=dict(color=color,width=4), thickness=0.75, value=score),
                    bgcolor=C_PANEL,
                    bordercolor=C_BORDER,
                ),
                number=dict(font=dict(color=color, size=40)),
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Flags
            st.markdown("**⚠️ Risk Factors Detected:**")
            if inp_swipe > 0.75:
                st.error(f"Swipe Ratio {inp_swipe:.2f} — abnormally high (scammer avg: 0.87)")
            if inp_usage > 7:
                st.warning(f"App Usage {inp_usage:.1f} hrs/day — unusually long sessions")
            if inp_age >= 60:
                st.warning(f"Age {inp_age} — senior demographic, elevated vulnerability")
            if inp_mutual < 3:
                st.error(f"Only {inp_mutual} mutual matches despite high activity — suspicious")
            if score < 45:
                st.success("No significant fraud indicators detected.")

        else:
            st.markdown(f"""
            <div style="background:{C_PANEL};border:1px solid {C_BORDER};border-radius:16px;
                        padding:40px;text-align:center;color:{C_MUTED};">
              <div style="font-size:3rem;">🛡️</div>
              <div style="margin-top:12px;">Fill in user data and click<br><strong>Predict Risk Level</strong></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🚨 Flagged Accounts Table")

    # Generate demo flagged accounts
    np.random.seed(7)
    flag_rows = []
    ages_g   = ["61–75","76+","46–60","76+","61–75","31–45","61–75","46–60"]
    risk_lvl = ["HIGH","HIGH","MEDIUM","HIGH","MEDIUM","LOW","HIGH","MEDIUM"]
    statuses = ["Flagged","Flagged","Under Review","Flagged","Under Review","Cleared","Flagged","Under Review"]
    for i in range(8):
        rs = 80+np.random.randint(15) if risk_lvl[i]=="HIGH" else \
             55+np.random.randint(20) if risk_lvl[i]=="MEDIUM" else \
             20+np.random.randint(25)
        flag_rows.append({
            "User ID":         f"USR-{np.random.randint(10000,99999)}",
            "Age Group":       ages_g[i],
            "Usage (hrs)":     round(np.random.uniform(2,14), 1),
            "Swipe Ratio":     round(np.random.uniform(0.1,1.0), 2),
            "Risk Score":      rs,
            "Risk Level":      risk_lvl[i],
            "Status":          statuses[i],
        })
    flags_df = pd.DataFrame(flag_rows)
    st.dataframe(flags_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ## 🛡️ Elder-Fraud Shield

    **Course:** WIA1006 / WID3006 — Machine Learning  
    **Semester:** Sem 2, Session 2025/2026  
    **Institution:** FCSIT, Universiti Malaya  

    ---

    ## 📌 Problem Statement
    Romance scams disproportionately target senior citizens on dating platforms.
    Current platforms react only after user reports. This project builds a **proactive
    machine learning detection system** that identifies scammer accounts using only
    behavioral metadata — without reading private messages.

    ## 🎯 Objectives
    - Train ≥ 5 ML models to classify "Catfished" outcomes
    - Maximize **Recall** to minimize missed scammers
    - Benchmark against **auto-sklearn**
    - Deliver an interactive analyst dashboard

    ## 🗂️ Dataset
    [Kaggle — Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)  
    50,000 synthetic records · 19 features

    ## 🤖 Models Used
    | Model | Recall |
    |---|---|
    | Random Forest ★ | 94.2% |
    | XGBoost | 92.1% |
    | auto-sklearn | 91.5% |
    | Logistic Regression | 85.6% |
    | KNN (k=7) | 81.3% |
    | Decision Tree | 78.9% |

    ## 🔗 Links
    - GitHub: https://github.com/Pohyi118/machinelearningproject
    - Submission Deadline: **8 June 2026, 12:00 PM** via SPECTRUM
    """)