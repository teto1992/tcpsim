# streamlit_app.py
# TCP congestion control toy simulator with algorithm picker
# NOTE: This is a *didactic* discrete-RTT model. It is not ns-3 nor Linux TCP.
# It aims to visualize trends and relative behaviors under configurable path params.

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Helpers
# ------------------------------
def compute_bdp_packets(bandwidth_mbps: float, rtt_ms: float, mss_bytes: int) -> float:
    bits_per_rtt = bandwidth_mbps * 1e6 * (rtt_ms / 1000.0)
    return bits_per_rtt / (8 * mss_bytes)

# ------------------------------
# Path & simulation parameters
# ------------------------------
@dataclass
class Path:
    bandwidth_mbps: float = 100.0   # base link capacity
    rtt_ms: float = 100.0           # base RTT (no queue)
    loss_prob: float = 0.01         # i.i.d. per-packet loss probability
    mss_bytes: int = 1460
    bw_variation_frac: float = 0.0  # +/- fractional variation per RTT (e.g., 0.5 => ±50%)
    rtt_variation_frac: float = 0.0 # +/- fractional variation per RTT
    rwnd_bytes: float | None = None # receiver advertised window (bytes); None = disattivo

    @property
    def bdp_packets(self) -> float:
        return compute_bdp_packets(self.bandwidth_mbps, self.rtt_ms, self.mss_bytes)

def effective_cwnd_pkts(cwnd_pkts: float, path: Path) -> float:
    """cwnd effettiva considerando il limite rwnd (in MSS)."""
    cw = max(1.0, cwnd_pkts)
    if path.rwnd_bytes is None or path.rwnd_bytes <= 0:
        return cw
    rwnd_pkts = max(1.0, path.rwnd_bytes / path.mss_bytes)
    return max(1.0, min(cw, rwnd_pkts))

# RNG helper for reproducibility
class RNG:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
    def bernoulli(self, p: float) -> bool:
        return self.rng.random() < p
    def uniform(self, a: float, b: float) -> float:
        return self.rng.uniform(a, b)

# ------------------------------
# Utility: one-RTT loss event probability given cwnd & per-pkt loss p
# For Reno-like loss signaling via duplicate ACKs, approximate the prob
# that at least one pkt in window is lost.
# ------------------------------
def window_loss_event_prob(cwnd_pkts: float, p_pkt: float) -> float:
    cwnd = max(1.0, cwnd_pkts)
    # P(no loss in window) = (1-p)^cwnd  => P(loss) = 1 - (1-p)^cwnd
    return 1.0 - (1.0 - p_pkt) ** cwnd

# Classify loss event into dupACK vs timeout (didactic heuristic)
def classify_loss_event(cwnd_pkts: float, rng: RNG) -> str:
    """
    Returns 'dup3ack', 'timeout', or 'none'.
    Heuristic:
      - If cwnd >= 4: mostly dup3ack, small chance timeout (e.g., burst losses).
      - If cwnd < 4: likely timeout (no room for 3 duplicate ACKs).
    """
    if cwnd_pkts >= 4.0:
        # 85% dupACK, 15% timeout
        return "dup3ack" if rng.bernoulli(0.85) else "timeout"
    else:
        # 80% timeout, 20% dupACK (rare ma possibile)
        return "timeout" if rng.bernoulli(0.80) else "dup3ack"

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
    # Vegas state (per Expected-Actual)
    base_rtt_ms: float = float("inf")

# Tahoe (loss-based)
def step_tahoe(state: CCState, path: Path, rng: RNG) -> Tuple[CCState, Dict]:
    eff_cw = effective_cwnd_pkts(state.cwnd, path)
    loss_happened = rng.bernoulli(window_loss_event_prob(eff_cw, path.loss_prob))
    event = "none"
    if state.cwnd < state.ssthresh:  # slow start (SS)
        state.cwnd *= 2.0
    else:                            # congestion avoidance (CA)
        state.cwnd += 1.0
    if loss_happened:
        event = classify_loss_event(eff_cw, rng)
        # Tahoe: su dup3ack e su timeout si torna a cwnd=1, ssthresh = cwnd/2
        pre = state.cwnd
        state.ssthresh = max(2.0, pre / 2.0)
        state.cwnd = 1.0
    return state, {"loss": loss_happened, "event": event}

# Reno (loss-based)
def step_reno(state: CCState, path: Path, rng: RNG) -> Tuple[CCState, Dict]:
    eff_cw = effective_cwnd_pkts(state.cwnd, path)
    loss_happened = rng.bernoulli(window_loss_event_prob(eff_cw, path.loss_prob))
    event = "none"
    if state.cwnd < state.ssthresh:
        state.cwnd *= 2.0
    else:
        state.cwnd += 1.0
    if loss_happened:
        event = classify_loss_event(eff_cw, rng)
        pre = state.cwnd
        if event == "dup3ack":
            # Fast retransmit + fast recovery (approx)
            state.ssthresh = max(2.0, pre / 2.0)
            state.cwnd = max(1.0, state.ssthresh)
        else:  # timeout
            state.ssthresh = max(2.0, pre / 2.0)
            state.cwnd = 1.0
    return state, {"loss": loss_happened, "event": event}

# CUBIC (loss-based, high-level approximation)
# W_cubic(t) = C*(t-K)^3 + Wmax, where K=(Wmax*β/C)^{1/3}; β≈0.3, C≈0.4
def step_cubic(state: CCState, path: Path, rng: RNG) -> Tuple[CCState, Dict]:
    beta = 0.3  # multiplicative decrease ~ (1-β)
    C = 0.4
    eff_cw = effective_cwnd_pkts(state.cwnd, path)
    loss_happened = rng.bernoulli(window_loss_event_prob(eff_cw, path.loss_prob))
    event = "none"
    if loss_happened:
        event = classify_loss_event(eff_cw, rng)
        state.Wmax = max(state.Wmax, state.cwnd)
        if event == "dup3ack":
            # classico backoff moltiplicativo
            state.cwnd = max(1.0, state.cwnd * (1.0 - beta))
            state.epoch = 0
        else:  # timeout: più severo
            state.ssthresh = max(2.0, state.cwnd / 2.0)
            state.cwnd = 1.0
            state.Wmax = 0.0
            state.epoch = 0
    else:
        # cubic growth from last Wmax
        t = state.epoch
        K = (state.Wmax * beta / C) ** (1.0 / 3.0) if state.Wmax > 0 else 0.0
        Wt = C * (t - K) ** 3 + state.Wmax if state.Wmax > 0 else state.cwnd + 1.0
        state.cwnd = max(state.cwnd + 1.0, Wt)
        state.epoch += 1
    return state, {"loss": loss_happened, "event": event}

# Vegas (delay-inspired; Expected vs Actual con baseRTT)
def step_vegas(state: CCState, path: Path, rng: RNG) -> Tuple[CCState, Dict]:
    # aggiorna baseRTT
    state.base_rtt_ms = min(state.base_rtt_ms, path.rtt_ms)
    # expected/actual (in pkts/RTT)
    expected = max(1e-9, state.cwnd)
    actual = expected * (state.base_rtt_ms / max(1e-9, path.rtt_ms))
    diff = expected - actual  # MSS "in coda"
    alpha, beta = 3.0, 6.0
    if diff < alpha:
        state.cwnd += 1.0
    elif diff > beta:
        state.cwnd = max(1.0, state.cwnd - 1.0)
    # reazione blanda alla perdita
    eff_cw = effective_cwnd_pkts(state.cwnd, path)
    loss_evt = rng.bernoulli(window_loss_event_prob(eff_cw, path.loss_prob))
    if loss_evt:
        state.cwnd = max(1.0, state.cwnd * 0.95)
    return state, {"loss": loss_evt, "event": "none"}

# BBR (model-based)
def step_bbr(state: CCState, path: Path, rng: RNG) -> Tuple[CCState, Dict]:
    state.btlbw_mbps = path.bandwidth_mbps
    target_cwnd = max(4.0, 2.0 * path.bdp_packets)
    state.epoch += 1
    oscillation = 0.1 * math.sin(state.epoch / 5.0)
    state.cwnd = max(1.0, target_cwnd * (1.0 + oscillation))
    eff_cw = effective_cwnd_pkts(state.cwnd, path)
    loss_evt = rng.bernoulli(window_loss_event_prob(eff_cw, path.loss_prob))
    return state, {"loss": loss_evt, "event": "none"}

# Map names to step funcs and initializers (LIMITED SET)
ALGORITHMS: Dict[str, Callable[[CCState, Path, RNG], tuple]] = {
    "Tahoe": step_tahoe,
    "Reno": step_reno,
    "CUBIC": step_cubic,
    "Vegas": step_vegas,
    "BBR": step_bbr,
}

# Fixed colors per algorithm (consistent palette)
COLOR_MAP = {
    "Tahoe": "#1f77b4",  # blue
    "Reno":  "#ff7f0e",  # orange
    "CUBIC": "#2ca02c",  # green
    "Vegas": "#d62728",  # red
    "BBR":   "#9467bd",  # purple
}

LOSS_BASED = {"Tahoe", "Reno", "CUBIC"}

# Event markers (fixed across algorithms)
EVENT_STYLE = {
    "dup3ack": dict(symbol="x", size=10, line=dict(width=1.8), color="#e377c2"),  # magenta
    "timeout": dict(symbol="x", size=10, line=dict(width=2.0), color="#7f7f7f"),  # grey
}

# ------------------------------
# Simulation engine
# ------------------------------
def simulate(algo: str, base_path: Path, duration_s: float, seed: int) -> pd.DataFrame:
    rng = RNG(seed + hash(algo) % 100000)
    # Initialize per-algo state
    if algo in ("Tahoe", "Reno"):
        state = CCState(cwnd=1.0, ssthresh=compute_bdp_packets(base_path.bandwidth_mbps, base_path.rtt_ms, base_path.mss_bytes))
    elif algo == "CUBIC":
        state = CCState(cwnd=1.0, ssthresh=1e9, Wmax=0.0, epoch=0)
    elif algo == "Vegas":
        state = CCState(cwnd=max(2.0, compute_bdp_packets(base_path.bandwidth_mbps, base_path.rtt_ms, base_path.mss_bytes) / 2), ssthresh=1e9)
    elif algo == "BBR":
        state = CCState(cwnd=max(4.0, 2.0 * compute_bdp_packets(base_path.bandwidth_mbps, base_path.rtt_ms, base_path.mss_bytes) / 3),
                        ssthresh=1e9, btlbw_mbps=base_path.bandwidth_mbps)
    else:
        raise ValueError("Unknown algorithm")

    times = []
    cwnds = []
    eff_cwnds = []
    send_rates = []
    goodputs = []
    loss_flags = []
    event_types = []  # 'none' | 'dup3ack' | 'timeout'
    algo_bases = []   # NEW: nome algoritmo base (senza etichette rwnd)

    step_fn = ALGORITHMS[algo]

    # working copy of path to allow per-RTT variation
    path = Path(
        bandwidth_mbps=base_path.bandwidth_mbps,
        rtt_ms=base_path.rtt_ms,
        loss_prob=base_path.loss_prob,
        mss_bytes=base_path.mss_bytes,
        bw_variation_frac=base_path.bw_variation_frac,
        rtt_variation_frac=base_path.rtt_variation_frac,
        rwnd_bytes=base_path.rwnd_bytes,
    )

    # cumulative time (seconds) to handle variable RTT per step
    t_s = 0.0
    max_iters = int(1e6)  # safety guard

    iters = 0
    while t_s < duration_s and iters < max_iters:
        iters += 1
        # Bandwidth variation per RTT (uniform in [-frac, +frac])
        if path.bw_variation_frac > 0:
            delta_bw = rng.uniform(-path.bw_variation_frac, path.bw_variation_frac)
            inst_bw = max(1e-6, base_path.bandwidth_mbps * (1.0 + delta_bw))
        else:
            inst_bw = base_path.bandwidth_mbps

        # RTT variation per RTT (uniform in [-frac, +frac])
        if base_path.rtt_variation_frac > 0:
            delta_rtt = rng.uniform(-base_path.rtt_variation_frac, base_path.rtt_variation_frac)
            inst_rtt_ms = max(0.1, base_path.rtt_ms * (1.0 + delta_rtt))  # clamp for stability
        else:
            inst_rtt_ms = base_path.rtt_ms

        # Expose instantaneous values to algorithms
        path.bandwidth_mbps = inst_bw
        path.rtt_ms = inst_rtt_ms

        # Effective cwnd given rwnd
        eff_cw = effective_cwnd_pkts(state.cwnd, path)

        # Throughput limited by eff_cwnd/RTT and instantaneous link capacity
        send_rate_mbps = min((eff_cw * path.mss_bytes * 8) / (inst_rtt_ms / 1000.0) / 1e6,
                             inst_bw)
        # i.i.d per-packet loss => expect goodput scaled by (1-p)
        goodput_mbps = send_rate_mbps * (1.0 - path.loss_prob)

        times.append(t_s)
        cwnds.append(state.cwnd)
        eff_cwnds.append(eff_cw)
        send_rates.append(send_rate_mbps)
        goodputs.append(goodput_mbps)
        algo_bases.append(algo)

        # Advance CC by one RTT
        state, meta = step_fn(state, path, rng)
        loss_flags.append(1 if meta.get("loss", False) else 0)
        event_types.append(meta.get("event", "none"))

        # Advance cumulative time by the instantaneous RTT
        t_s += inst_rtt_ms / 1000.0

    df = pd.DataFrame({
        "time_s": times,
        "cwnd_pkts": cwnds,
        "eff_cwnd_pkts": eff_cwnds,  # NEW: cwnd effettiva limitata da rwnd
        "send_mbps": send_rates,
        "goodput_mbps": goodputs,
        "loss_event": loss_flags,
        "event_type": event_types,  # 'none' | 'dup3ack' | 'timeout'
        "algo": algo,               # verrà eventualmente sostituito con etichetta
        "algo_base": algo_bases,    # NEW: sempre il nome "puro"
    })
    df["is_loss_based"] = df["algo_base"].isin(list(LOSS_BASED))  # NEW
    return df

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="TCP CC toy simulator", layout="wide")
st.title("Controllo della congestione in TCP")

with st.sidebar:
    st.header("Parametri del collegamento bottleneck")
    bw = st.number_input("Banda (Mbps)", 1.0, 10000.0, 100.0, 1.0)
    rtt = st.number_input("RTT (ms)", 1.0, 2000.0, 100.0, 1.0)

    # Flow control (rwnd) — singolo valore in KB
    use_rwnd = st.checkbox(
        "Applica limite finestra ricevitore (rwnd)", value=False,
        help="Se attivo, la finestra di invio è limitata dal rwnd pubblicizzato dal ricevitore."
    )
    rwnd_kb = None
    if use_rwnd:
        rwnd_kb = st.number_input(
            "rwnd (KB)", min_value=1.0, max_value=1048576.0, value=64.0, step=1.0,
            help="Valore singolo in kilobyte (KB)."
        )

    # Percent input for very low loss rates
    loss_pct = st.number_input(
        "Tasso di perdita (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.01,
        format="%.4f",
        help="Inserisci la percentuale di perdita per pacchetto (es. 0.05 = 0.05%)"
    )
    loss = loss_pct / 100.0

    mss = st.selectbox("MSS (bytes)", [1200, 1360, 1460, 1500], index=2)

    st.header("Variazioni del collegamento bottleneck")
    bw_var_pct = st.slider("Banda ± (%)", 0, 50, 0, 1,
                           help="Se >0, ad ogni RTT la banda varia in modo uniforme casuale nell'intervallo selezionato.")
    bw_var_frac = bw_var_pct / 100.0

    rtt_var_pct = st.slider("RTT ± (%)", 0, 50, 0, 1,
                            help="Se >0, ad ogni RTT il RTT varia in modo uniforme casuale nell'intervallo selezionato.")
    rtt_var_frac = rtt_var_pct / 100.0

    st.header("Simulazione")
    duration = st.number_input("Durata (s)", 0.5, 600.0, 30.0, 0.5)
    seed = st.number_input("Seed", 0, 10_000_000, 42, 1)

    st.header("Algoritmi da plottare")
    # Restricted set
    default_selection = ["CUBIC", "BBR", "Reno", "Vegas"]
    algos_selected = st.multiselect(
        "Scegli uno o più algoritmi",
        ["Tahoe", "Reno", "CUBIC", "Vegas", "BBR"],
        default=default_selection,
    )

    st.header("Cosa visualizzare")
    what = st.radio("Serie temporali", ["Goodput (Mbps)", "cwnd (MSS)", "Send rate (Mbps)"])
    show_events = st.checkbox("Evidenzia eventi di perdita", value=True)
    show_eff_cwnd = st.checkbox("Mostra finestra effettiva in tabella", value=True)

# Build base path (variation fractions included) — rwnd rimane None, gestito in sweep
path = Path(bandwidth_mbps=bw, rtt_ms=rtt, loss_prob=loss, mss_bytes=mss,
            bw_variation_frac=bw_var_frac, rtt_variation_frac=rtt_var_frac,
            rwnd_bytes=None)

if not algos_selected:
    st.info("Seleziona almeno un algoritmo nella sidebar per iniziare.")
    st.stop()

# Run simulations (singolo valore di rwnd se attivo)
frames: List[pd.DataFrame] = []

for name in algos_selected:
    sim_path = Path(**{**path.__dict__})
    label = name
    if use_rwnd and (rwnd_kb is not None):
        sim_path.rwnd_bytes = rwnd_kb * 1024.0
        label = f"{name} (rwnd {int(rwnd_kb)}KB)"
    df = simulate(name, sim_path, duration, seed)
    df["algo"] = label                 # etichetta mostrata
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df["color_key"] = all_df["algo_base"]


# KPIs
base_bdp = compute_bdp_packets(bw, rtt, mss)
cols = st.columns(6)
with cols[0]:
    st.metric("BDP base (MSS)", f"{base_bdp:.1f}")
with cols[1]:
    st.metric("Capacità base", f"{bw:.1f} Mbps")
with cols[2]:
    st.metric("RTT base", f"{rtt:.0f} ms")
with cols[3]:
    st.metric("Perdita", f"{loss_pct:.4f}%")
with cols[4]:
    st.metric("Var. banda ±", f"{100*path.bw_variation_frac:.0f}%")
with cols[5]:
    st.metric("Var. RTT ±", f"{100*path.rtt_variation_frac:.0f}%")

# Plot
metric_map = {
    "Goodput (Mbps)": ("goodput_mbps", "Goodput (Mbps)"),
    "cwnd (MSS)": ("cwnd_pkts", "cwnd (MSS)"),
    "Send rate (Mbps)": ("send_mbps", "Send rate (Mbps)"),
}
col, ylab = metric_map[what]

fig = px.line(
    all_df, x="time_s", y=col, color="color_key", render_mode="svg",
    color_discrete_map=COLOR_MAP, line_group="algo",
)
fig.update_layout(xaxis_title="Tempo (s)", yaxis_title=ylab, legend_title_text="Algoritmo",
                  template="plotly_white")

if show_events:
    # Solo algoritmi loss-based (indipendente dall'etichetta visualizzata)
    events_df = all_df[all_df["is_loss_based"]]
    # 3dupACK
    dup_df = events_df[events_df["event_type"] == "dup3ack"]
    if not dup_df.empty:
        fig.add_trace(go.Scatter(
            x=dup_df["time_s"],
            y=dup_df[col],
            mode="markers",
            marker=EVENT_STYLE["dup3ack"],
            name="3dupACK",
            showlegend=True,
            hovertemplate="Evento: 3dupACK<br>t=%{x:.3f}s<br>"+ylab+": %{y:.3f}<extra></extra>",
        ))
    # Timeout
    to_df = events_df[events_df["event_type"] == "timeout"]
    if not to_df.empty:
        fig.add_trace(go.Scatter(
            x=to_df["time_s"],
            y=to_df[col],
            mode="markers",
            marker=EVENT_STYLE["timeout"],
            name="timeout",
            showlegend=True,
            hovertemplate="Evento: Timeout<br>t=%{x:.3f}s<br>"+ylab+": %{y:.3f}<extra></extra>",
        ))

st.plotly_chart(fig, use_container_width=True)

# Aggregate stats
st.subheader("Statistiche riassuntive")
agg_cols = {
    "avg_goodput_Mbps": ("goodput_mbps", "mean"),
    "avg_send_Mbps": ("send_mbps", "mean"),
    "avg_cwnd_pkts": ("cwnd_pkts", "mean"),
    "avg_eff_cwnd_pkts": ("eff_cwnd_pkts", "mean"),
    "loss_events": ("loss_event", "sum"),
}
summary = (all_df
           .groupby("algo", as_index=False)
           .agg(**{k: pd.NamedAgg(*v) for k, v in agg_cols.items()}))

# Aggiungi colonne per dup3ack/timeout solo per loss-based (usiamo la maschera)
dup_counts = (all_df[all_df["is_loss_based"]]
              .groupby("algo")["event_type"].apply(lambda s: int((s == "dup3ack").sum())))
to_counts = (all_df[all_df["is_loss_based"]]
             .groupby("algo")["event_type"].apply(lambda s: int((s == "timeout").sum())))

summary["dup3ack"] = summary["algo"].map(dup_counts).fillna(0).astype(int)
summary["timeout"] = summary["algo"].map(to_counts).fillna(0).astype(int)

# Jain's fairness index over average goodput of selected flows
x = summary["avg_goodput_Mbps"].to_numpy()
if len(x) > 0:
    jain = (x.sum() ** 2) / (len(x) * (x**2).sum()) if (x**2).sum() > 0 else 0.0
else:
    jain = float("nan")

# Mostra tabella (con o senza eff_cwnd)
fmt = {
    "avg_goodput_Mbps": "{:.2f}",
    "avg_send_Mbps": "{:.2f}",
    "avg_cwnd_pkts": "{:.2f}",
    "avg_eff_cwnd_pkts": "{:.2f}",
    "loss_events": "{:d}",
    "dup3ack": "{:d}",
    "timeout": "{:d}",
}
cols_order = ["algo", "avg_goodput_Mbps", "avg_send_Mbps", "avg_cwnd_pkts"]
if show_eff_cwnd:
    cols_order.append("avg_eff_cwnd_pkts")
cols_order += ["loss_events", "dup3ack", "timeout"]

st.dataframe(summary[cols_order].style.format(fmt))

st.caption(f"Indice di Jain (goodput medio): {jain:.3f} — più vicino a 1 ⇒ più equità tra algoritmi selezionati.")

st.markdown(
    """
**Avvertenze didattiche**  
Questo modello è intenzionalmente semplificato (passo = 1 RTT, perdite i.i.d., code trascurate,
assenza di SACK/ACK clock reali, pacing semplificato per BBR). Serve per *intuire* le differenze qualitative
tra TCP Tahoe/Reno, CUBIC, Vegas e BBR in presenza di:
- banda e RTT base impostabili nella sidebar
- (opzionale) **variazione di banda e/o RTT** per-RTT uniforme nell’intervallo ±X%
- perdita impostata in **percentuale** (es. 0.05 = 0.05%)
- **flow control**: se attivo, `rwnd` limita l'invio a `min(cwnd, rwnd)`; puoi inserire **più valori in KB**
- negli algoritmi **loss-based** si distinguono **3dupACK** e **Timeout** con euristica semplificata

Per esperimenti più fedeli: ns-3, Mininet, o Linux con `tc` e `netem`.
"""
)
