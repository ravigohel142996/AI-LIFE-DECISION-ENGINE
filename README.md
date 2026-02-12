# AI Life Decision Engine

A futuristic Streamlit app that simulates major life decisions with a blended AI stack:
- Reinforcement-learning-inspired agent (PyTorch)
- Bayesian probability model (Scikit-learn)
- Time-series wealth forecasting (Pandas + NumPy)
- Interactive analytics dashboard (Plotly)
- NLP advisor bot and downloadable PDF roadmap

## Features

- Glassmorphism/neon UI and 3D-style metric cards
- User profile cockpit (age, education, skills, income, savings, city, stress, goals, family support)
- Decision simulator:
  - Job vs Startup
  - India vs Abroad
  - Higher Study vs Work
  - Invest vs Save
- Dashboard outputs:
  - Success Probability
  - Happiness Score
  - Wealth Projection
  - Risk Map
  - Life Path Tree
- Visualizations:
  - Plotly 3D trajectory chart
  - Decision tree map
  - Timeline forecast
- Advisor Bot for personalized guidance
- PDF life roadmap download

## Project Structure

```text
app.py
engine/
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

Open the local URL shown by Streamlit (usually http://localhost:8501).
