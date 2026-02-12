from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import (
    AdvisorBot,
    BayesianLifeModel,
    DecisionSimulator,
    LifeForecastModel,
    RLDecisionAgent,
    build_life_roadmap_pdf,
)


st.set_page_config(page_title="AI Life Decision Engine", page_icon="🚀", layout="wide")


THEME = {
    "bg_start": "#0A0F1F",
    "bg_end": "#111827",
    "primary_text": "#FFFFFF",
    "secondary_text": "#B8C1EC",
    "accent": "#007AFF",
    "highlight": "#00E5FF",
}


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --primary: {THEME['primary_text']};
            --secondary: {THEME['secondary_text']};
            --accent: {THEME['accent']};
            --highlight: {THEME['highlight']};
            --glass: rgba(255,255,255,0.05);
            --glass-strong: rgba(255,255,255,0.08);
            --border: rgba(184,193,236,0.22);
          }}

          html, body, [class*="css"], .stApp, p, div, span, label, h1, h2, h3, h4, h5 {{
            color: var(--primary) !important;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif !important;
          }}

          .stApp {{
            background: linear-gradient(160deg, {THEME['bg_start']} 0%, {THEME['bg_end']} 100%);
          }}

          .block-container {{
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }}

          [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: transparent;
          }}

          .hero-card {{
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.1rem;
            background: linear-gradient(125deg, rgba(0,122,255,0.26), rgba(0,229,255,0.12) 60%, rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 20px 56px rgba(0,0,0,0.34);
          }}

          .hero-card::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 85% 0%, rgba(0,229,255,0.18), transparent 45%);
            pointer-events: none;
          }}

          .hero-title {{
            font-size: clamp(2rem, 4.2vw, 3rem);
            line-height: 1.1;
            font-weight: 740;
            margin: 0;
            letter-spacing: -0.02em;
          }}

          .hero-subtitle {{
            margin: 0.45rem 0 0 0;
            color: var(--secondary) !important;
            font-size: 1.05rem;
          }}

          .glass-panel {{
            border-radius: 20px;
            padding: 1.2rem;
            background: var(--glass);
            border: 1px solid var(--border);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
          }}

          .section-heading {{
            font-size: 1.28rem;
            font-weight: 650;
            margin: 0 0 0.8rem 0;
            color: var(--primary) !important;
            letter-spacing: -0.01em;
          }}

          .metric-card {{
            border-radius: 18px;
            padding: 1rem;
            background: linear-gradient(140deg, rgba(255,255,255,0.09), rgba(255,255,255,0.04));
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 10px 24px rgba(0,0,0,0.24);
            min-height: 136px;
            transition: transform 220ms ease, box-shadow 220ms ease;
          }}

          .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 30px rgba(0,122,255,0.22);
          }}

          .metric-label {{
            font-size: 0.88rem;
            color: var(--secondary) !important;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
          }}

          .metric-value {{
            font-size: clamp(1.9rem, 2.2vw, 2.35rem);
            margin: 0;
            font-weight: 700;
            color: var(--primary) !important;
          }}

          .insight-strip {{
            margin-top: 0.8rem;
            padding: 0.95rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(0,229,255,0.28);
            background: linear-gradient(120deg, rgba(0,122,255,0.13), rgba(0,229,255,0.08));
            color: var(--primary) !important;
          }}

          .stButton > button {{
            background: linear-gradient(120deg, #007AFF, #00A2FF) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.72rem 1.1rem !important;
            font-weight: 640 !important;
            box-shadow: 0 10px 24px rgba(0,122,255,0.35);
            transition: all 220ms ease;
          }}

          .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(0,122,255,0.45);
            filter: brightness(1.06);
          }}

          .stButton > button:disabled {{
            opacity: 0.55 !important;
            color: #e5efff !important;
            background: linear-gradient(120deg, rgba(0,122,255,0.55), rgba(0,162,255,0.45)) !important;
            cursor: not-allowed !important;
          }}

          [data-testid="stMetricValue"],
          .stMarkdown,
          [data-testid="stWidgetLabel"],
          .stSlider label,
          .stRadio label,
          .stSelectbox label,
          .stTextInput label,
          .stNumberInput label,
          .stTextArea label,
          .stDownloadButton span {{
            color: var(--primary) !important;
            opacity: 1 !important;
          }}

          .stCaption, .stMarkdown small, .stAlert p {{
            color: var(--secondary) !important;
          }}

          .stTextInput > div > div > input,
          .stNumberInput > div > div > input,
          .stTextArea textarea,
          .stSelectbox div[data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.9) !important;
            color: #0D1328 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.55) !important;
          }}

          .stRadio [role="radiogroup"] > label,
          .stSlider [data-baseweb="slider"] {{
            background: transparent !important;
          }}

          .stPlotlyChart, .chart-wrap {{
            border-radius: 20px;
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border);
            padding: 0.35rem;
          }}

          .separator {{
            height: 1px;
            width: 100%;
            margin: 1.1rem 0;
            background: linear-gradient(90deg, transparent, rgba(0,229,255,0.55), transparent);
          }}

          @media (max-width: 1200px) {{
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
            .hero-card {{ padding: 1.6rem; }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(title: str) -> dict:
    return {
        "title": {"text": title, "font": {"size": 22, "color": THEME["primary_text"]}, "x": 0.01},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": THEME["secondary_text"], "family": "Inter, sans-serif"},
        "margin": {"l": 36, "r": 20, "t": 52, "b": 36},
    }


def build_profile(age: int, education: str, skills: str, income: int, savings: int, city: str, stress_level: int, career_goals: str, family_support: int) -> dict:
    return {
        "age": age,
        "education": education,
        "skills": skills,
        "income": income,
        "savings": savings,
        "city": city,
        "stress_level": stress_level,
        "career_goals": career_goals,
        "family_support": family_support,
    }


def build_risk_map(savings: float, income: float, stress_level: int, decisions: dict[str, str]) -> pd.DataFrame:
    scores = {
        "Financial": 100 - min(100, (savings / max(income, 1)) * 30 + 20),
        "Career": 55 + (15 if decisions["career_path"] == "Startup" else -8),
        "Emotional": stress_level * 9,
        "Execution": 45 + (20 if decisions["location"] == "Abroad" else 5),
    }
    clipped = {k: float(np.clip(v, 0, 100)) for k, v in scores.items()}
    return pd.DataFrame({"Risk": list(clipped.keys()), "Score": list(clipped.values())})


def build_decision_alignment(decisions: dict[str, str]) -> float:
    risk_flags = [
        decisions["career_path"] == "Startup",
        decisions["location"] == "Abroad",
        decisions["study_work"] == "Higher Study",
        decisions["invest_save"] == "Invest",
    ]
    return float(np.clip((4 - sum(risk_flags)) / 4, 0, 1))


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <p class='metric-value'>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_results(profile: dict, decisions: dict[str, str]) -> dict:
    simulator = DecisionSimulator()
    rl_agent = RLDecisionAgent(n_features=9)
    bayes_model = BayesianLifeModel()
    forecast_model = LifeForecastModel()
    advisor = AdvisorBot()

    state = simulator.encode_profile(profile, decisions)
    rl_success, happiness = rl_agent.train(state)
    bayes = bayes_model.predict(state)
    success_probability = float(np.clip((rl_success * 0.62 + bayes["bayesian_success"] * 0.38), 0, 1))

    wealth_df = forecast_model.project_wealth(profile["income"], profile["savings"], decisions["invest_save"], years=12)
    wealth_quality = forecast_model.forecast_quality(wealth_df)
    risk_map = build_risk_map(profile["savings"], profile["income"], profile["stress_level"], decisions)
    advice = advisor.suggest(profile, decisions)
    decision_alignment = build_decision_alignment(decisions)

    metrics_for_pdf = {
        "success_probability": f"{success_probability*100:.2f}%",
        "happiness_score": f"{happiness:.2f}",
        "wealth_after_12y": f"₹{wealth_df['Projected Wealth'].iloc[-1]:,.0f}",
        "risk_peak": f"{risk_map.loc[risk_map['Score'].idxmax(), 'Risk']} ({risk_map['Score'].max():.1f})",
        "confidence": f"{bayes['confidence']*100:.2f}%",
    }

    pdf_bytes = None
    try:
        pdf_bytes = build_life_roadmap_pdf(profile, decisions, metrics_for_pdf, advice)
    except Exception:
        pdf_bytes = None

    return {
        "success_probability": success_probability,
        "happiness": happiness,
        "wealth_quality": wealth_quality,
        "wealth_df": wealth_df,
        "risk_map": risk_map,
        "advice": advice,
        "decision_alignment": decision_alignment,
        "decisions": decisions,
        "profile": profile,
        "pdf_bytes": pdf_bytes,
    }


def render_dashboard(result: dict) -> None:
    sp = result["success_probability"]
    happiness = result["happiness"]
    wealth_quality = result["wealth_quality"]
    decisions = result["decisions"]
    profile = result["profile"]

    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric("Success Probability", f"{sp * 100:.1f}%")
    with m2:
        render_metric("Happiness Score", f"{happiness:.1f}/100")
    with m3:
        render_metric("Forecast Confidence", f"{wealth_quality * 100:.1f}%")

    insight = (
        f"<b>Executive Insight:</b> Your profile indicates <b>{'high upside' if sp > 0.7 else 'moderate upside'}</b> "
        f"with a <b>{'balanced' if happiness > 60 else 'high-pressure'}</b> trajectory. Recommended posture: "
        f"<b>{'aggressive growth' if result['decision_alignment'] < 0.5 else 'risk-calibrated execution'}</b>."
    )
    st.markdown(f"<div class='insight-strip'>{insight}</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-heading'>Advanced Visual Analytics</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=np.linspace(1, 10, 20),
                    y=np.linspace(1, 10, 20) ** 0.8,
                    z=np.linspace(sp * 40, happiness, 20),
                    mode="markers+lines",
                    marker=dict(size=5, color=np.linspace(0, 1, 20), colorscale="Turbo"),
                    line=dict(color=THEME["highlight"], width=4),
                )
            ]
        )
        fig3d.update_layout(
            **plotly_layout("3D Life Trajectory"),
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Capability",
                zaxis_title="Outcome",
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with c2:
        treemap_df = pd.DataFrame(
            {
                "Node": ["Life Path", "Career", "Location", "Finance", decisions["career_path"], decisions["location"], decisions["invest_save"]],
                "Parent": ["", "Life Path", "Life Path", "Life Path", "Career", "Location", "Finance"],
                "Value": [100, 33, 33, 34, 20, 18, 22],
            }
        )
        tree_fig = px.treemap(
            treemap_df,
            names="Node",
            parents="Parent",
            values="Value",
            color="Value",
            color_continuous_scale=[[0, "#007AFF"], [1, "#00E5FF"]],
            title="Life Path Tree",
        )
        tree_fig.update_layout(**plotly_layout("Life Path Tree"))
        st.plotly_chart(tree_fig, use_container_width=True)

    radar_scores = pd.DataFrame(
        {
            "Dimension": ["Career", "Finance", "Wellbeing", "Execution", "Support"],
            "Score": [
                72 + (12 if decisions["career_path"] == "Startup" else -6),
                70 + (14 if decisions["invest_save"] == "Invest" else -4),
                85 - profile["stress_level"] * 5,
                68 + (8 if decisions["study_work"] == "Higher Study" else 3),
                profile["family_support"] * 10,
            ],
        }
    )
    radar_fig = px.line_polar(radar_scores, r="Score", theta="Dimension", line_close=True, range_r=[0, 100], title="Decision Readiness Radar")
    radar_fig.update_traces(fill="toself", line_color=THEME["highlight"], fillcolor="rgba(0,229,255,0.35)")
    radar_fig.update_layout(**plotly_layout("Decision Readiness Radar"))
    st.plotly_chart(radar_fig, use_container_width=True)

    timeline = px.line(result["wealth_df"], x="Year", y="Projected Wealth", title="Timeline Forecast: Wealth Projection", markers=True)
    timeline.update_traces(line=dict(color=THEME["highlight"], width=4))
    timeline.update_layout(**plotly_layout("Timeline Forecast: Wealth Projection"))
    st.plotly_chart(timeline, use_container_width=True)

    risk_fig = px.bar(result["risk_map"], x="Risk", y="Score", color="Score", title="Risk Map", color_continuous_scale=[[0, "#007AFF"], [1, "#00E5FF"]])
    risk_fig.update_layout(**plotly_layout("Risk Map"))
    st.plotly_chart(risk_fig, use_container_width=True)

    st.markdown("<h3 class='section-heading'>Advisor Bot</h3>", unsafe_allow_html=True)
    for tip in result["advice"]:
        st.info(tip)

    if result["pdf_bytes"]:
        st.download_button(
            label="Download Life Roadmap PDF",
            data=result["pdf_bytes"],
            file_name="ai_life_roadmap.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("PDF generation is temporarily unavailable for this scenario. Please adjust inputs and retry.")


inject_global_styles()
st.markdown(
    """
    <section class="hero-card">
      <h1 class="hero-title">AI Life Decision Engine</h1>
      <p class="hero-subtitle">Neural strategy cockpit for your next decade.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None

simulator = DecisionSimulator()
left, right = st.columns([1.02, 1.55], gap="large")

with left:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-heading'>Profile & Strategy Inputs</h3>", unsafe_allow_html=True)

    age = st.slider("Age", 16, 60, 25)
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "MBA", "PhD"])
    skills = st.text_input("Skills", "Python, Product, Communication")
    income = st.number_input("Annual Income (₹)", min_value=0, value=800000, step=50000)
    savings = st.number_input("Current Savings (₹)", min_value=0, value=400000, step=50000)
    city = st.text_input("City", "Bengaluru")
    stress_level = st.slider("Stress Level", 1, 10, 5)
    career_goals = st.text_area("Career Goals", "Build a high-impact AI startup with global reach.")
    family_support = st.slider("Family Support", 1, 10, 7)

    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
    st.markdown("<h4 class='section-heading'>Strategic Scenario Simulator</h4>", unsafe_allow_html=True)

    decisions: dict[str, str] = {}
    for scenario in simulator.scenarios:
        decisions[scenario.key] = st.radio(scenario.label, scenario.options, horizontal=True)

    run = st.button("Run AI Decision Engine ⚡", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if run:
        profile = build_profile(age, education, skills, int(income), int(savings), city, stress_level, career_goals, family_support)
        st.session_state.simulation_result = build_results(profile, decisions)

    if st.session_state.simulation_result:
        render_dashboard(st.session_state.simulation_result)
    else:
        st.markdown(
            """
            <div class='glass-panel'>
              <h3 class='section-heading'>Awaiting simulation</h3>
              <p style='color: var(--secondary); margin-bottom: 0;'>
                Complete your profile and click <b>Run AI Decision Engine ⚡</b> to generate a premium decision dashboard.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
