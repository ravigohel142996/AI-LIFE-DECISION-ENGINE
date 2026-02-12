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
        background: radial-gradient(circle at 10% 20%, #111430 0%, #07070f 45%, #020203 100%);
        color: #eaf2ff;
      }
      .glass {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(100, 170, 255, 0.35);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 0 24px rgba(81, 160, 255, 0.07);
        border-radius: 20px;
        padding: 1rem;
      }
      .hero {
        text-align: center;
        padding: 1.1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(0,255,252,0.13), rgba(255,0,212,0.1));
        border: 1px solid rgba(0,255,252,0.35);
        box-shadow: 0 15px 50px rgba(0,255,252,0.09);
        transform: perspective(1000px) rotateX(2deg);
      }
      .metric-card {
        background: linear-gradient(145deg, rgba(24,24,40,0.95), rgba(18,30,57,0.9));
        border: 1px solid rgba(37, 240, 255, 0.35);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 18px 35px rgba(0, 255, 255, 0.08);
      }
      h1, h2, h3 { color: #b5d5ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero"><h1>AI Life Decision Engine</h1><p>Neural strategy cockpit for your next decade.</p></div>', unsafe_allow_html=True)

simulator = DecisionSimulator()
rl_agent = RLDecisionAgent(n_features=9)
bayes_model = BayesianLifeModel()
forecast_model = LifeForecastModel()
advisor = AdvisorBot()

left, right = st.columns([1.05, 1.6], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("User Profile Input")
    age = st.slider("Age", 16, 60, 25)
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "MBA", "PhD"])
    skills = st.text_input("Skills", "Python, Product, Communication")
    income = st.number_input("Annual Income (₹)", min_value=0, value=800000, step=50000)
    savings = st.number_input("Current Savings (₹)", min_value=0, value=400000, step=50000)
    city = st.text_input("City", "Bengaluru")
    stress_level = st.slider("Stress level", 1, 10, 5)
    career_goals = st.text_area("Career goals", "Build a high-impact AI startup with global reach.")
    family_support = st.slider("Family support", 1, 10, 7)

    st.subheader("Decision Simulator")
    decisions = {}
    for scenario in simulator.scenarios:
        decisions[scenario.key] = st.radio(scenario.label, scenario.options, horizontal=True)

    run = st.button("Run AI Decision Engine ⚡", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if run:
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
        success_probability = float(np.clip((rl_success * 0.62 + bayes["bayesian_success"] * 0.38), 0, 1))

        wealth_df = forecast_model.project_wealth(income, savings, decisions["invest_save"], years=12)
        wealth_quality = forecast_model.forecast_quality(wealth_df)

        risk_map = pd.DataFrame(
            {
                "Risk": ["Financial", "Career", "Emotional", "Execution"],
                "Score": [
                    100 - min(100, (savings / max(income, 1)) * 30 + 20),
                    55 + (15 if decisions["career_path"] == "Startup" else -8),
                    stress_level * 9,
                    45 + (20 if decisions["location"] == "Abroad" else 5),
                ],
            }
        )
        advice = advisor.suggest(profile, decisions)

        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card"><h3>Success Probability</h3><h2>{success_probability*100:.1f}%</h2></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><h3>Happiness Score</h3><h2>{happiness:.1f}/100</h2></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><h3>Forecast Confidence</h3><h2>{wealth_quality*100:.1f}%</h2></div>', unsafe_allow_html=True)

        st.subheader("Visualization")
        c1, c2 = st.columns(2)

        with c1:
            fig3d = go.Figure(
                data=[
                    go.Scatter3d(
                        x=np.linspace(1, 10, 20),
                        y=np.linspace(1, 10, 20) ** 0.8,
                        z=np.linspace(success_probability * 40, happiness, 20),
                        mode="markers+lines",
                        marker=dict(size=6, color=np.linspace(0, 1, 20), colorscale="Turbo"),
                        line=dict(color="#00f6ff", width=4),
                    )
                ]
            )
            fig3d.update_layout(
                title="3D Life Trajectory",
                scene=dict(
                    xaxis_title="Time",
                    yaxis_title="Capability",
                    zaxis_title="Outcome",
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
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
                color_continuous_scale="Bluered",
                title="Life Path Tree",
            )
            tree_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(tree_fig, use_container_width=True)

        timeline = px.line(
            wealth_df,
            x="Year",
            y="Projected Wealth",
            title="Timeline Forecast: Wealth Projection",
            markers=True,
        )
        timeline.update_traces(line=dict(color="#00f0ff", width=4))
        timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(timeline, use_container_width=True)

        st.subheader("Advisor Bot")
        for tip in advice:
            st.info(tip)

        metrics_for_pdf = {
            "success_probability": f"{success_probability*100:.2f}%",
            "happiness_score": f"{happiness:.2f}",
            "wealth_after_12y": f"₹{wealth_df['Projected Wealth'].iloc[-1]:,.0f}",
            "risk_peak": f"{risk_map.loc[risk_map['Score'].idxmax(), 'Risk']} ({risk_map['Score'].max():.1f})",
            "confidence": f"{bayes['confidence']*100:.2f}%",
        }

        pdf_bytes = build_life_roadmap_pdf(profile, decisions, metrics_for_pdf, advice)
        st.download_button(
            label="Download Life Roadmap PDF",
            data=pdf_bytes,
            file_name="ai_life_roadmap.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        risk_fig = px.bar(risk_map, x="Risk", y="Score", color="Score", title="Risk Map")
        risk_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(risk_fig, use_container_width=True)
    else:
        st.markdown('<div class="glass"><h3>Awaiting simulation...</h3><p>Complete your profile and click <b>Run AI Decision Engine ⚡</b>.</p></div>', unsafe_allow_html=True)
