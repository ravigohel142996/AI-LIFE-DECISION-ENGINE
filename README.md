# AI Life Decision Engine

A futuristic Streamlit application for simulating major life decisions with a blended AI stack and executive-style visual analytics.

## Highlights

- **Futuristic UI**: glassmorphism, neon glow accents, 3D-style cards, dark futuristic gradient theme
- **User Profile Input**:
  - Age, Education, Skills
  - Income, Savings, City
  - Stress Level, Career Goals, Family Support
- **Decision Simulator**:
  - Job vs Startup
  - India vs Abroad
  - Higher Study vs Work
  - Invest vs Save
- **AI Engine**:
  - Reinforcement-learning-inspired decision agent (PyTorch with deterministic fallback)
  - Bayesian probability model (Scikit-learn with fallback)
  - Time-series wealth projection
- **Output Dashboard**:
  - Success Probability
  - Happiness Score
  - Forecast/Bayesian confidence
  - Risk Map
  - Life Path Tree
- **Visualization**:
  - Plotly 3D trajectory chart
  - Decision tree map
  - Timeline wealth forecast
  - Strategic radar profile
- **Advisor Bot**:
  - NLP-style recommendation retrieval
  - Personalized suggestions based on profile + selected decisions
- **Report Generator**:
  - Downloadable life roadmap PDF

## Architecture

```text
app.py
engine/
  __init__.py
  advisor.py
  models.py
  report.py
  simulator.py
requirements.txt
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL (typically `http://localhost:8501`).
