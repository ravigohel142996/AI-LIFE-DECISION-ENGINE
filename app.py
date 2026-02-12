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

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 15% 20%, rgba(38,87,255,0.28), transparent 30%),
          radial-gradient(circle at 85% 0%, rgba(255,0,209,0.22), transparent 25%),
          linear-gradient(140deg, #060812 0%, #0a1021 45%, #04040a 100%);
        color: #eaf2ff;
      }
      .hero {
        padding: 1.2rem;
        border-radius: 22px;
        border: 1px solid rgba(0,255,252,0.36);
        background: linear-gradient(135deg, rgba(0,255,252,0.12), rgba(255,0,200,0.1));
        box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 0 34px rgba(67,190,255,0.1);
        transform: perspective(1200px) rotateX(2deg);
      }
      .glass {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(119, 169, 255, 0.35);
        padding: 1rem;
        box-shadow: 0 14px 42px rgba(0, 0, 0, 0.3);
      }
      .metric {
        border-radius: 20px;
        padding: 0.85rem 1rem;
        background: linear-gradient(145deg, rgba(28,37,60,0.95), rgba(17,23,42,0.95));
        border: 1px solid rgba(84,228,255,0.35);
        box-shadow: 0 16px 28px rgba(0,255,255,0.09);
      }
      .neon-divider {
        border-top: 1px solid rgba(0,255,252,0.35);
        box-shadow: 0 0 24px rgba(0,255,252,0.45);
        margin: 0.8rem 0 1rem 0;
      }
      h1, h2, h3 { color: #b8d9ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>AI Life Decision Engine</h1><p><b>AI-powered strategic cockpit for your next 10 years.</b></p></div>',
    unsafe_allow_html=True,
)

simulator = DecisionSimulator()
rl_agent = RLDecisionAgent(n_features=9)
bayes_model = BayesianLifeModel()
forecast_model = LifeForecastModel()
advisor = AdvisorBot()

left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("User Profile Input")
    age = st.slider("Age", 16, 60, 24)
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "MBA", "PhD"])
    skills = st.text_input("Skills", "Python, AI, Product, Communication")
    income = st.number_input("Annual Income (₹)", min_value=0, value=900000, step=50000)
    savings = st.number_input("Current Savings (₹)", min_value=0, value=450000, step=50000)
    city = st.text_input("City", "Bengaluru")
    stress_level = st.slider("Stress Level", 1, 10, 5)
    career_goals = st.text_area("Career Goals", "Build an AI company with global impact while staying financially resilient.")
    family_support = st.slider("Family Support", 1, 10, 7)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.subheader("Decision Simulator")
    decisions = {}
    for scenario in simulator.scenarios:
        decisions[scenario.key] = st.radio(scenario.label, scenario.options, horizontal=True)

    run = st.button("Run AI Decision Engine ⚡", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if not run:
        st.markdown(
            '<div class="glass"><h3>Awaiting simulation...</h3><p>Complete your profile and run the engine to generate strategy-grade projections.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        profile = {
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
        state = simulator.encode_profile(profile, decisions)

        rl_success, happiness = rl_agent.train(state)
        bayes = bayes_model.predict(state)
        success_probability = float(np.clip(rl_success * 0.6 + bayes["bayesian_success"] * 0.4, 0, 1))

        wealth_df = forecast_model.project_wealth(income, savings, decisions["invest_save"], years=12)
        forecast_conf = forecast_model.forecast_quality(wealth_df)

        risk_map = pd.DataFrame(
            {
                "Risk": ["Financial", "Career", "Emotional", "Execution"],
                "Score": [
                    100 - min(100, (savings / max(income, 1)) * 28 + 22),
                    48 + (22 if decisions["career_path"] == "Startup" else 5),
                    stress_level * 8.5,
                    42 + (22 if decisions["location"] == "Abroad" else 8),
                ],
            }
        )
        advice = advisor.suggest(profile, decisions)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric"><h3>Success</h3><h2>{success_probability*100:.1f}%</h2></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric"><h3>Happiness</h3><h2>{happiness:.1f}/100</h2></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric"><h3>Confidence</h3><h2>{forecast_conf*100:.1f}%</h2></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric"><h3>Bayesian Certainty</h3><h2>{bayes["confidence"]*100:.1f}%</h2></div>', unsafe_allow_html=True)

        tab_dash, tab_viz, tab_bot, tab_report = st.tabs(["Dashboard", "Visual Intelligence", "Advisor Bot", "Report"])

        with tab_dash:
            st.subheader("Output Dashboard")
            c1, c2 = st.columns(2)
            with c1:
                risk_fig = px.bar(risk_map, x="Risk", y="Score", color="Score", title="Risk Map")
                risk_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(risk_fig, use_container_width=True)
            with c2:
                decision_strength = pd.DataFrame(
                    {
                        "Dimension": ["Career", "Location", "Study/Work", "Finance"],
                        "Strength": [
                            0.72 if decisions["career_path"] == "Startup" else 0.56,
                            0.68 if decisions["location"] == "Abroad" else 0.59,
                            0.64 if decisions["study_work"] == "Higher Study" else 0.61,
                            0.74 if decisions["invest_save"] == "Invest" else 0.57,
                        ],
                    }
                )
                radar = go.Figure()
                radar.add_trace(
                    go.Scatterpolar(
                        r=decision_strength["Strength"],
                        theta=decision_strength["Dimension"],
                        fill="toself",
                        line=dict(color="#33f6ff", width=3),
                        name="Decision Strength",
                    )
                )
                radar.update_layout(title="Strategic Decision Profile", polar=dict(bgcolor="rgba(0,0,0,0)"), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(radar, use_container_width=True)

        with tab_viz:
            st.subheader("Visualization")
            c1, c2 = st.columns(2)
            with c1:
                fig3d = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=np.linspace(1, 12, 24),
                            y=np.linspace(0.3, 1.2, 24) ** 1.2,
                            z=np.linspace(success_probability * 40, happiness, 24),
                            mode="markers+lines",
                            marker=dict(size=6, color=np.linspace(0, 1, 24), colorscale="Turbo"),
                            line=dict(color="#00f6ff", width=4),
                        )
                    ]
                )
                fig3d.update_layout(
                    title="3D Life Trajectory",
                    scene=dict(xaxis_title="Time", yaxis_title="Capability", zaxis_title="Outcome", bgcolor="rgba(0,0,0,0)"),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig3d, use_container_width=True)

            with c2:
                tree = pd.DataFrame(
                    {
                        "Node": ["Life Path", "Career", "Location", "Finance", decisions["career_path"], decisions["location"], decisions["invest_save"]],
                        "Parent": ["", "Life Path", "Life Path", "Life Path", "Career", "Location", "Finance"],
                        "Value": [100, 33, 33, 34, 21, 19, 23],
                    }
                )
                tree_fig = px.treemap(tree, names="Node", parents="Parent", values="Value", color="Value", color_continuous_scale="Bluered", title="Life Path Tree")
                tree_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(tree_fig, use_container_width=True)

            timeline = px.line(wealth_df, x="Year", y="Projected Wealth", markers=True, title="Timeline Wealth Forecast")
            timeline.update_traces(line=dict(color="#00e5ff", width=4))
            timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(timeline, use_container_width=True)

        with tab_bot:
            st.subheader("Advisor Bot: NLP Guidance")
            for i, tip in enumerate(advice, start=1):
                st.info(f"{i}. {tip}")

        with tab_report:
            st.subheader("Report Generator")
            metrics_for_pdf = {
                "success_probability": f"{success_probability*100:.2f}%",
                "happiness_score": f"{happiness:.2f}",
                "wealth_after_12y": f"₹{wealth_df['Projected Wealth'].iloc[-1]:,.0f}",
                "peak_risk": f"{risk_map.loc[risk_map['Score'].idxmax(), 'Risk']} ({risk_map['Score'].max():.1f})",
                "bayesian_confidence": f"{bayes['confidence']*100:.2f}%",
            }
            pdf_bytes = build_life_roadmap_pdf(profile, decisions, metrics_for_pdf, advice)
            st.download_button(
                label="Download Life Roadmap PDF",
                data=pdf_bytes,
                file_name="ai_life_roadmap.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
