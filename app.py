# streamlit_app.py
# TCP congestion control toy simulator with algorithm picker
# NOTE: This is a *didactic* discrete‑RTT model. It is not ns-3 nor Linux TCP.
# It aims to visualize trends and relative behaviors under configurable path params.

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Path & simulation parameters
# ------------------------------
@dataclass
class Path:
    bandwidth_mbps: float = 100.0  # link capacity
    rtt_ms: float = 100.0          # base RTT (no queue)
    loss_prob: float = 0.05         # i.i.d. per-packet loss probability
    mss_bytes: int = 1460

    @property
    def bdp_packets(self) -> float:
        bits_per_rtt = self.bandwidth_mbps * 1e6 * (self.rtt_ms/1000.0)
        return bits_per_rtt / (8 * self.mss_bytes)

# RNG helper for reproducibility
class RNG:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
    def bernoulli(self, p: float) -> bool:
        return self.rng.random() < p

# ------------------------------
# Utility: one-RTT loss event probability given cwnd & per-pkt loss p
# For Reno-like loss signaling via duplicate ACKs, approximate the prob
# that at least one pkt in window is lost.
# ------------------------------

def window_loss_event_prob(cwnd_pkts: float, p_pkt: float) -> float:
    cwnd = max(1.0, cwnd_pkts)
    # P(no loss in window) = (1-p)^cwnd  => P(loss) = 1 - (1-p)^cwnd
    return 1.0 - (1.0 - p_pkt) ** cwnd

# ------------------------------
# Discrete-RTT toy models for congestion control variants
# Each returns new cwnd, ssthresh and meta (dict)
# ------------------------------

@dataclass
class CCState:
    cwnd: float
    ssthresh: float
    # Cubic state
    Wmax: float = 0.0
    epoch: int = 0
    # BBR state
    btlbw_mbps: float = 0.0

# Tahoe

def step_tahoe(state: CCState, path: Path, rng: RNG):
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    if state.cwnd < state.ssthresh:  # slow start (SS)
        state.cwnd *= 2.0
    else:                            # congestion avoidance (CA)
        state.cwnd += 1.0
    if loss_evt:
        state.ssthresh = max(2.0, state.cwnd / 2.0)
        state.cwnd = 1.0  # Tahoe goes back to 1 MSS
    return state, {"loss": loss_evt}

# Reno

def step_reno(state: CCState, path: Path, rng: RNG):
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    if state.cwnd < state.ssthresh:
        state.cwnd *= 2.0
    else:
        state.cwnd += 1.0
    if loss_evt:
        state.ssthresh = max(2.0, state.cwnd / 2.0)
        state.cwnd = max(1.0, state.ssthresh)  # fast recovery approx
    return state, {"loss": loss_evt}

# NewReno (very similar in this toy model)

def step_newreno(state: CCState, path: Path, rng: RNG):
    # Slightly less drastic cwnd deflation on loss vs Reno
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    if state.cwnd < state.ssthresh:
        state.cwnd *= 2.0
    else:
        state.cwnd += 1.0
    if loss_evt:
        state.ssthresh = max(2.0, state.cwnd / 2.0)
        state.cwnd = max(1.0, 0.7 * state.ssthresh)  # gentler deflation
    return state, {"loss": loss_evt}

# CUBIC (high-level approximation)
# W_cubic(t) = C*(t-K)^3 + Wmax, where K=(Wmax*β/C)^{1/3}; β≈0.7, C≈0.4
# We discretize time by RTTs; on loss, Wmax <- cwnd, cwnd <- cwnd*(1-β)

def step_cubic(state: CCState, path: Path, rng: RNG):
    beta = 0.3  # multiplicative decrease ~ 1-β in Linux docs (β~0.3)
    C = 0.4
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    if loss_evt:
        state.Wmax = max(state.Wmax, state.cwnd)
        state.cwnd = max(1.0, state.cwnd * (1.0 - beta))
        state.epoch = 0
    else:
        # cubic growth from last Wmax
        t = state.epoch
        K = (state.Wmax * beta / C) ** (1.0/3.0) if state.Wmax > 0 else 0.0
        Wt = C * (t - K) ** 3 + state.Wmax
        state.cwnd = max(state.cwnd + 1.0, Wt) if state.Wmax > 0 else state.cwnd + 1.0
        state.epoch += 1
    return state, {"loss": loss_evt}

# Vegas (delay-based): try to keep ~α..β packets queued; adjust cwnd toward BDP

def step_vegas(state: CCState, path: Path, rng: RNG):
    alpha, beta = 3.0, 6.0
    # Estimate baseRTT as path.rtt_ms; queuing ~ (cwnd - BDP)
    target_low = path.bdp_packets + alpha
    target_high = path.bdp_packets + beta
    if state.cwnd < target_low:
        state.cwnd += 1.0
    elif state.cwnd > target_high:
        state.cwnd = max(1.0, state.cwnd - 1.0)
    # Vegas reacts little to random loss; if loss, small decrease
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    if loss_evt:
        state.cwnd = max(1.0, state.cwnd * 0.9)
    return state, {"loss": loss_evt}

# "NewVegas" (placeholder: same spirit, tighter target)

def step_newvegas(state: CCState, path: Path, rng: RNG):
    alpha, beta = 2.0, 4.0
    target_low = path.bdp_packets + alpha
    target_high = path.bdp_packets + beta
    if state.cwnd < target_low:
        state.cwnd += 1.0
    elif state.cwnd > target_high:
        state.cwnd = max(1.0, state.cwnd - 1.0)
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    if loss_evt:
        state.cwnd = max(1.0, state.cwnd * 0.92)
    return state, {"loss": loss_evt}

# BBR (model): pace at BtlBw, cwnd ≈ 2*BDP; ignore random loss unless severe

def step_bbr(state: CCState, path: Path, rng: RNG):
    # In this toy, assume measured bottleneck bandwidth = link capacity
    state.btlbw_mbps = path.bandwidth_mbps
    target_cwnd = max(4.0, 2.0 * path.bdp_packets)
    # Probe: oscillate slightly around target
    state.epoch += 1
    oscillation = 0.1 * math.sin(state.epoch / 5.0)
    state.cwnd = max(1.0, target_cwnd * (1.0 + oscillation))
    # Loss does not drive cwnd directly here
    loss_evt = rng.bernoulli(window_loss_event_prob(state.cwnd, path.loss_prob))
    return state, {"loss": loss_evt}

# Map names to step funcs and initializers
ALGORITHMS: Dict[str, Callable[[CCState, Path, RNG], tuple]] = {
    "Tahoe": step_tahoe,
    "Reno": step_reno,
    "NewReno": step_newreno,
    "CUBIC": step_cubic,
    "Vegas": step_vegas,
    "NewVegas": step_newvegas,
    "BBR": step_bbr,
}

# ------------------------------
# Simulation engine
# ------------------------------

def simulate(algo: str, path: Path, duration_s: float, seed: int) -> pd.DataFrame:
    rng = RNG(seed + hash(algo) % 100000)
    # Initialize per‑algo state
    if algo in ("Tahoe", "Reno", "NewReno"):
        state = CCState(cwnd=1.0, ssthresh=path.bdp_packets)  # start in slow start
    elif algo == "CUBIC":
        state = CCState(cwnd=1.0, ssthresh=1e9, Wmax=0.0, epoch=0)
    elif algo in ("Vegas", "NewVegas"):
        state = CCState(cwnd=max(2.0, path.bdp_packets/2), ssthresh=1e9)
    elif algo == "BBR":
        state = CCState(cwnd=max(4.0, 2.0*path.bdp_packets/3), ssthresh=1e9, btlbw_mbps=path.bandwidth_mbps)
    else:
        raise ValueError("Unknown algorithm")

    rtts = int(max(1, duration_s / (path.rtt_ms/1000.0)))
    times = []
    cwnds = []
    send_rates = []
    goodputs = []
    losses = []

    step_fn = ALGORITHMS[algo]

    for r in range(rtts):
        # Throughput limited by cwnd/RTT and link capacity
        send_rate_mbps = min((state.cwnd * path.mss_bytes * 8) / (path.rtt_ms/1000.0) / 1e6,
                             path.bandwidth_mbps)
        # i.i.d per‑packet loss => expect goodput scaled by (1-p)
        goodput_mbps = send_rate_mbps * (1.0 - path.loss_prob)

        times.append(r * path.rtt_ms/1000.0)
        cwnds.append(state.cwnd)
        send_rates.append(send_rate_mbps)
        goodputs.append(goodput_mbps)

        # Advance CC by one RTT
        state, meta = step_fn(state, path, rng)
        losses.append(1 if meta.get("loss", False) else 0)

    df = pd.DataFrame({
        "time_s": times,
        "cwnd_pkts": cwnds,
        "send_mbps": send_rates,
        "goodput_mbps": goodputs,
        "loss_event": losses,
        "algo": algo,
    })
    return df

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="TCP CC toy simulator", layout="wide")
st.title("TCP Congestion Control – simulatore didattico")

with st.sidebar:
    st.header("Parametri del link")
    bw = st.number_input("Banda (Mbps)", 1.0, 10000.0, 100.0, 1.0)
    rtt = st.number_input("RTT (ms)", 1.0, 2000.0, 100.0, 1.0)
    loss = st.slider("Tasso di perdita per pacchetto", 0.0, 0.2, 0.05, 0.005)
    mss = st.selectbox("MSS (bytes)", [1200, 1360, 1460, 1500], index=2)

    st.header("Simulazione")
    duration = st.number_input("Durata (s)", 0.5, 600.0, 30.0, 0.5)
    seed = st.number_input("Seed", 0, 10_000_000, 42, 1)

    st.header("Algoritmi da plottare")
    default_selection = ["CUBIC", "BBR", "Reno", "Vegas"]
    algos_selected = st.multiselect(
        "Scegli uno o più algoritmi",
        list(ALGORITHMS.keys()),
        default=default_selection,
    )

    st.header("Cosa visualizzare")
    what = st.radio("Serie temporali", ["Goodput (Mbps)", "cwnd (pacchetti)", "Send rate (Mbps)"])
    show_losses = st.checkbox("Evidenzia eventi di perdita", value=False)

path = Path(bandwidth_mbps=bw, rtt_ms=rtt, loss_prob=loss, mss_bytes=mss)

if not algos_selected:
    st.info("Seleziona almeno un algoritmo nella sidebar per iniziare.")
    st.stop()

# Run simulations
frames: List[pd.DataFrame] = []
for name in algos_selected:
    frames.append(simulate(name, path, duration, seed))
all_df = pd.concat(frames, ignore_index=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("BDP (pacchetti)", f"{path.bdp_packets:.1f}")
with col2:
    st.metric("Capacità link", f"{path.bandwidth_mbps:.1f} Mbps")
with col3:
    st.metric("RTT", f"{path.rtt_ms:.0f} ms")
with col4:
    st.metric("Perdita", f"{100*path.loss_prob:.1f}%")

# Plot
metric_map = {
    "Goodput (Mbps)": ("goodput_mbps", "Goodput (Mbps)"),
    "cwnd (pacchetti)": ("cwnd_pkts", "cwnd (pacchetti)"),
    "Send rate (Mbps)": ("send_mbps", "Send rate (Mbps)"),
}
col, ylab = metric_map[what]

fig = px.line(all_df, x="time_s", y=col, color="algo", render_mode="svg")
fig.update_layout(xaxis_title="Tempo (s)", yaxis_title=ylab, legend_title_text="Algoritmo",
                  template="plotly_white")

if show_losses:
    # Overlay loss events as semi-transparent markers at the current metric value
    loss_df = all_df[all_df["loss_event"] == 1]
    if not loss_df.empty:
        fig.add_trace(go.Scatter(
            x=loss_df["time_s"],
            y=loss_df[col],
            mode="markers",
            marker=dict(color="red", size=6, opacity=0.4),
            name="Perdita",
            showlegend=True,
        ))

st.plotly_chart(fig, use_container_width=True)

# Aggregate stats
st.subheader("Statistiche riassuntive")
summary = (all_df
           .groupby("algo")
           .agg(avg_goodput_Mbps=("goodput_mbps", "mean"),
                avg_send_Mbps=("send_mbps", "mean"),
                avg_cwnd_pkts=("cwnd_pkts", "mean"),
                loss_events=("loss_event", "sum"))
           .reset_index())

# Jain's fairness index over average goodput of selected flows
x = summary["avg_goodput_Mbps"].to_numpy()
if len(x) > 0:
    jain = (x.sum() ** 2) / (len(x) * (x**2).sum()) if (x**2).sum() > 0 else 0.0
else:
    jain = float("nan")

st.dataframe(summary.style.format({
    "avg_goodput_Mbps": "{:.2f}",
    "avg_send_Mbps": "{:.2f}",
    "avg_cwnd_pkts": "{:.2f}",
    "loss_events": "{:d}",
}))

st.caption(f"Indice di Jain (goodput medio): {jain:.3f} — più vicino a 1 ⇒ più equità tra algoritmi selezionati.")

st.markdown(
    """
**Avvertenze didattiche**  
Questo modello è intenzionalmente semplificato (passo = 1 RTT, perdite i.i.d., code trascurate,
assenza di SACK/ACK clock reali, pacing semplificato per BBR). Serve per *intuire* le differenze qualitative
tra TCP Tahoe/Reno/NewReno, CUBIC e le famiglie Vegas/BBR in presenza di:
- banda = 100 Mbps, RTT = 100 ms, perdita = 5% (valori modificabili nella sidebar)

Per esperimenti più fedeli: ns‑3, Mininet, o Linux con `tc` e `netem`.
"""
)
