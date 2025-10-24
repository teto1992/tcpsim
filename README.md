# tcpsim

A compact, didactic TCP congestion-control simulator (Streamlit app) for visualizing and comparing algorithms.

Why this repo
- Small, readable discrete‑RTT model to illustrate qualitative behaviour of TCP algorithms.
- Good for teaching, quick prototyping, and intuition-building (not a packet‑accurate simulator).

Features
- Multiple algorithms: Tahoe, Reno, CUBIC, Vegas, BBR.
- Configurable path: bandwidth, RTT, per-packet loss, MSS, queue/BDP intuition.
- Simple time‑series outputs: cwnd, send rate, goodput, and loss events.
- Interactive UI using Streamlit (app.py).

Requirements
- Python 3.8+
- streamlit, pandas, numpy, plotly (and any extras listed in requirements.txt if present)

Quick start
1. Create a virtualenv and install dependencies:
   python -m venv .venv
   source .venv/bin/activate
   pip install streamlit pandas numpy plotly

   (Or: pip install -r requirements.txt if the file exists.)

2. Run the app:
   streamlit run app.py

   The UI exposes the link parameters and algorithm picker in the sidebar. Select one or more algorithms and view cwnd / send rate / goodput time series.

Notes on the model
- The app uses a simple, per‑RTT step model (see app.py). Losses are i.i.d. per packet and time is discretized to RTTs — this is intentional for didactic clarity.

Extending
- Add a new step_<algo> function following the CCState / step_fn pattern in app.py.
- Register the name in the ALGORITHMS mapping to expose it to the UI.
- Add tests for increase/decrease behavior and expected KPIs.

