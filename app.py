"""TCP congestion-control laboratory, independent of the original app.py.

Run: python -m streamlit run new_app.py
Requires Python >=3.8; Streamlit/pandas/Plotly are imported only by the UI or
by the backwards-compatible simulate() DataFrame adapter.

This is a single-flow packet/ACK event model, not a TCP stack. Packet sizes
are payload bytes; capacity is therefore payload-equivalent. The receiver
uses cumulative ACKs without SACK. Reno recovery is deliberately classic
Reno, including its weakness with multiple losses. CUBIC uses that same
recovery. Vegas and BBR are explicitly approximate teaching variants.

References and model limitations are also displayed in the application.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


ALGORITHMS = ("Tahoe", "Reno", "CUBIC", "Vegas", "BBR")
COLOR_MAP = {"Tahoe": "#1f77b4", "Reno": "#ff7f0e", "CUBIC": "#2ca02c",
             "Vegas": "#d62728", "BBR": "#9467bd"}
ALGORITHM_LABELS = {name: name for name in ALGORITHMS}
ALGORITHM_LABELS.update({"Tahoe": "Tahoe (storico)",
                         "Vegas": "Vegas (didattico)",
                         "BBR": "BBRv1-inspired (didattico)"})


def compute_bdp_packets(bandwidth_mbps: float, rtt_ms: float, mss_bytes: int) -> float:
    return bandwidth_mbps * 1e6 * rtt_ms / 1000.0 / (8 * mss_bytes)


@dataclass(frozen=True)
class Path:
    bandwidth_mbps: float = 100.0
    rtt_ms: float = 100.0
    loss_prob: float = 0.001
    mss_bytes: int = 1460
    queue_packets: int = 64  # Waiting slots, in addition to one in service.
    bw_variation_frac: float = 0.0
    rtt_variation_frac: float = 0.0
    rwnd_bytes: Optional[int] = None
    variation_interval_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("bandwidth_mbps", "rtt_ms", "variation_interval_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if not isinstance(self.mss_bytes, int) or isinstance(self.mss_bytes, bool) or self.mss_bytes <= 0:
            raise ValueError("mss_bytes must be a positive integer")
        if not isinstance(self.queue_packets, int) or isinstance(self.queue_packets, bool) or self.queue_packets < 0:
            raise ValueError("queue_packets must be a nonnegative integer")
        if not math.isfinite(self.loss_prob) or not 0 <= self.loss_prob <= 1:
            raise ValueError("loss_prob must be in [0, 1]")
        for name in ("bw_variation_frac", "rtt_variation_frac"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1:
                raise ValueError("%s must be in [0, 1)" % name)
        if self.rwnd_bytes is not None:
            if not isinstance(self.rwnd_bytes, int) or isinstance(self.rwnd_bytes, bool) or self.rwnd_bytes < 0:
                raise ValueError("rwnd_bytes must be a nonnegative integer or None")

    @property
    def bdp_packets(self) -> float:
        return compute_bdp_packets(self.bandwidth_mbps, self.rtt_ms, self.mss_bytes)


def effective_cwnd_pkts(cwnd_pkts: float, path: Path) -> float:
    """A zero advertised window really stops new data; never round up rwnd."""
    return max(0.0, cwnd_pkts) if path.rwnd_bytes is None else max(
        0.0, min(cwnd_pkts, path.rwnd_bytes / path.mss_bytes))


def initial_window_packets(mss_bytes: int, policy: str = "rfc5681") -> float:
    if policy == "legacy1":
        return 1.0
    if policy == "rfc5681":
        return min(4 * mss_bytes, max(2 * mss_bytes, 4380)) / mss_bytes
    if policy == "rfc6928":
        return min(10 * mss_bytes, max(2 * mss_bytes, 14600)) / mss_bytes
    raise ValueError("Unknown initial window policy: " + policy)


@dataclass
class RTOEstimator:
    """RFC 6298 sections 2 and 5, in seconds; Karn is enforced by the sender."""
    srtt: Optional[float] = None
    rttvar: Optional[float] = None
    rto: float = 1.0
    granularity: float = 0.001
    minimum: float = 1.0
    maximum: float = 60.0

    def observe(self, sample_s: float) -> None:
        if not math.isfinite(sample_s) or sample_s <= 0:
            raise ValueError("RTT sample must be finite and positive")
        if self.srtt is None:
            self.srtt, self.rttvar = sample_s, sample_s / 2.0
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - sample_s)
            self.srtt = 0.875 * self.srtt + 0.125 * sample_s
        self.rto = min(self.maximum, max(self.minimum,
                                        self.srtt + max(self.granularity, 4 * self.rttvar)))

    def backoff(self) -> None:
        self.rto = min(self.maximum, 2 * self.rto)


class PathTrace:
    """Exogenous trace indexed by WALL time, identical across algorithms.

    Capacity and propagation RTT are piecewise constant, independently drawn
    every variation_interval_s. A packet retains the capacity selected when
    its service begins and the propagation delay selected when it is sent.
    """
    def __init__(self, path: Path, seed: int):
        self.path = path
        self._bw_rng = random.Random("tcpsim:bandwidth:%d" % seed)
        self._rtt_rng = random.Random("tcpsim:rtt:%d" % seed)
        self._values = []  # type: List[Tuple[float, float]]

    def at(self, time_s: float) -> Tuple[float, float]:
        index = int(max(0.0, time_s) / self.path.variation_interval_s)
        while len(self._values) <= index:
            bw = self.path.bandwidth_mbps * (1 + self._bw_rng.uniform(
                -self.path.bw_variation_frac, self.path.bw_variation_frac))
            rtt = self.path.rtt_ms / 1000 * (1 + self._rtt_rng.uniform(
                -self.path.rtt_variation_frac, self.path.rtt_variation_frac))
            self._values.append((bw, rtt))
        return self._values[index]


@dataclass
class Segment:
    seq: int
    size: int
    first_sent: float
    last_sent: float
    transmissions: int = 1
    timer_transmissions: int = 0
    delivered_at_send: int = 0
    delivered_time_at_send: float = 0.0
    first_flight_send: float = 0.0


@dataclass
class SimulationResult:
    samples: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    summary: Dict[str, Any]
    complete: bool
    stop_reason: str

    def dataframe(self):
        import pandas as pd
        frame = pd.DataFrame(self.samples)
        frame.attrs.update(summary=self.summary, complete=self.complete,
                           stop_reason=self.stop_reason, events=self.events)
        return frame


class CongestionController:
    """Only ACK/loss observations enter this object; no path parameters do.

    The transport implements loss detection, retransmission and classic Reno
    fast recovery. It must suppress normal cwnd growth during recovery.
    For BBR only, ``cwnd_limited`` means network-limited (not constrained by
    receiver/application); BBR can otherwise be intentionally pacing-limited.
    ``round_end`` marks an ACK crossing a saved send-window sequence boundary.
    """

    CUBIC_C = 0.4
    CUBIC_BETA = 0.7
    BBR_HIGH_GAIN = 2.885
    BBR_GAINS = (1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def __init__(self, algo: str, mss_bytes: int, initial_cwnd: float):
        if algo not in ("Tahoe", "Reno", "CUBIC", "Vegas", "BBR"):
            raise ValueError("Unknown congestion controller: " + str(algo))
        if mss_bytes <= 0 or not math.isfinite(initial_cwnd) or initial_cwnd < 1:
            raise ValueError("MSS must be positive and initial cwnd at least one")
        self.algo = algo
        self.mss_bytes = mss_bytes
        self.initial_cwnd = float(initial_cwnd)
        self.cwnd = float(initial_cwnd)
        self.ssthresh = math.inf
        self.mode = "Startup" if algo == "BBR" else "Slow start"
        self.pacing_rate_pps = None  # type: Optional[float]
        self.srtt_s = None  # type: Optional[float]
        self._ca_credit = 0.0

        self.fast_convergence = False
        self._cubic_epoch = None  # type: Optional[float]
        self._cubic_pause_at = None  # type: Optional[float]
        self._cubic_wmax = None  # type: Optional[float]
        self._cubic_prior = self.cwnd
        self._cubic_west = self.cwnd
        self._cubic_k = 0.0
        self._cubic_after_timeout = False

        self.base_rtt_s = math.inf
        self._vegas_min_rtt = math.inf
        self._vegas_samples = 0
        self._vegas_round_cwnd = self.cwnd
        self._vegas_round_limited = False
        self._vegas_grow_round = True
        self._vegas_restart = False

        self.btlbw_pps = 0.0
        self.min_rtt_s = math.inf
        self._bbr_min_rtt_stamp = None  # type: Optional[float]
        self._bbr_round = 0
        self._bbr_delivered = 0.0
        self._bbr_has_measured_rate = False
        self._bbr_bw_samples = deque()
        self._bbr_full_bw = 0.0
        self._bbr_full_bw_rounds = 0
        self._bbr_full_pipe = False
        self._bbr_round_limited = False
        self._bbr_cycle_index = 0
        self._bbr_cycle_stamp = 0.0
        self._bbr_probe_done_at = None  # type: Optional[float]
        self._bbr_probe_round = None  # type: Optional[int]
        self._bbr_prior_cwnd = self.cwnd
        self._bbr_loss_in_cycle = False
        if algo == "BBR":
            # Unmeasured bootstrap assumption, as in Linux's fallback: 1 ms.
            # The first usable RTT/delivery observations replace this guess.
            self.pacing_rate_pps = self.BBR_HIGH_GAIN * self.cwnd / 0.001

    def _reno_increase(self, acked_packets: float) -> None:
        if self.cwnd < self.ssthresh:
            # RFC 5681: at most one SMSS per ACK, including delayed ACKs.
            self.cwnd += min(acked_packets, 1.0)
            self.mode = "Slow start"
        else:
            # RFC 5681 byte counting: grow by one SMSS after cwnd SMSSs
            # have been acknowledged. One accumulated-window increase/ACK.
            self._ca_credit += acked_packets
            if self._ca_credit >= self.cwnd:
                self._ca_credit -= self.cwnd
                self.cwnd += 1.0
            self.mode = "Congestion avoidance"

    def _cubic_increase(self, now: float, acked_packets: float) -> None:
        if self.cwnd < self.ssthresh:
            self._reno_increase(acked_packets)
            return
        if self._cubic_epoch is None:
            self._cubic_epoch = now
            if self._cubic_wmax is None or self._cubic_after_timeout:
                self._cubic_wmax = self.cwnd
                self._cubic_prior = self.cwnd
            delta = (self._cubic_wmax - self.cwnd) / self.CUBIC_C
            self._cubic_k = math.copysign(abs(delta) ** (1.0 / 3.0), delta)
            self._cubic_west = self.cwnd
            self._cubic_after_timeout = False
        elapsed = max(0.0, now - self._cubic_epoch)
        rtt = self.srtt_s if self.srtt_s is not None else 0.0
        wcubic = self.CUBIC_C * (elapsed - self._cubic_k) ** 3 + self._cubic_wmax
        target = self.CUBIC_C * (elapsed + rtt - self._cubic_k) ** 3 + self._cubic_wmax
        target = max(self.cwnd, min(target, 1.5 * self.cwnd))
        alpha = 1.0 if self._cubic_west >= self._cubic_prior else (
            3.0 * (1.0 - self.CUBIC_BETA) / (1.0 + self.CUBIC_BETA)
        )
        self._cubic_west += alpha * acked_packets / self.cwnd
        if wcubic < self._cubic_west:
            self.cwnd = max(self.cwnd, self._cubic_west)
            self.mode = "CUBIC Reno-friendly"
        else:
            # RFC 9438 sections 4.4/4.5 specify this increment per new ACK.
            self.cwnd += (target - self.cwnd) / self.cwnd
            self.mode = "CUBIC concave" if self.cwnd < self._cubic_wmax else "CUBIC convex"

    def _vegas_ack(self, acked_packets: float, rtt_s: Optional[float],
                   cwnd_limited: bool, round_end: bool) -> None:
        if rtt_s is not None and rtt_s > 0:
            self.base_rtt_s = min(self.base_rtt_s, rtt_s)
            self._vegas_min_rtt = min(self._vegas_min_rtt, rtt_s)
            self._vegas_samples += 1
        self._vegas_round_limited |= cwnd_limited
        was_slow_start = self.cwnd < self.ssthresh
        # Original Vegas alternates growth and observation rounds in SS.
        if cwnd_limited and was_slow_start and self._vegas_grow_round:
            self._reno_increase(acked_packets)
        if not round_end:
            return
        valid = self._vegas_samples >= 3 and math.isfinite(self.base_rtt_s)
        if self._vegas_restart:
            # After recovery, wait for a fresh complete round before using
            # delay feedback recorded against a different congestion window.
            self._vegas_restart = False
        elif self._vegas_round_limited and valid:
            target = self._vegas_round_cwnd * self.base_rtt_s / self._vegas_min_rtt
            # Original expected-minus-actual rate times BaseRTT (packets).
            queued = max(0.0, self._vegas_round_cwnd - target)
            if was_slow_start:
                if queued > 1.0:  # gamma
                    self.cwnd = max(2.0, min(self.cwnd, target + 1.0))
                    self.ssthresh = self.cwnd
                    self.mode = "Vegas avoidance"
                else:
                    self._vegas_grow_round = not self._vegas_grow_round
                    self.mode = "Vegas slow start"
            else:
                if queued < 2.0:  # alpha
                    self.cwnd += 1.0
                elif queued > 4.0:  # beta
                    self.cwnd = max(2.0, self.cwnd - 1.0)
                self.mode = "Vegas avoidance"
        elif cwnd_limited and not was_slow_start:
            # With too few RTT observations, use conservative Reno feedback.
            self._reno_increase(acked_packets)
            self.mode = "Vegas Reno fallback"
        self._vegas_round_cwnd = self.cwnd
        self._vegas_round_limited = False
        self._vegas_min_rtt = math.inf
        self._vegas_samples = 0

    def _bbr_ack(self, now: float, acked_packets: float,
                 rtt_s: Optional[float], rate_sample_pps: Optional[float],
                 inflight_packets: float, network_limited: bool,
                 round_end: bool) -> None:
        self._bbr_delivered += acked_packets
        self._bbr_round_limited |= network_limited
        expired = (self._bbr_min_rtt_stamp is not None and
                   now - self._bbr_min_rtt_stamp > 10.0)
        # Compute expiry BEFORE replacing the old sample, otherwise ProbeRTT
        # would never start when the min-RTT timestamp is refreshed.
        if rtt_s is not None and rtt_s > 0:
            if rtt_s < self.min_rtt_s or expired:
                self.min_rtt_s = rtt_s
                self._bbr_min_rtt_stamp = now
        if expired and self.mode != "ProbeRTT":
            self._bbr_prior_cwnd = self.cwnd
            self.mode = "ProbeRTT"
            self._bbr_probe_done_at = None
            self._bbr_probe_round = None
        # Keep at most one maximum per packet-timed round. ProbeRTT's
        # deliberately low samples must not lower the bandwidth estimate.
        usable_rate = (rate_sample_pps is not None and rate_sample_pps > 0
                       and math.isfinite(rate_sample_pps))
        if usable_rate and self.mode != "ProbeRTT" and (
                network_limited or rate_sample_pps >= self.btlbw_pps):
            while self._bbr_bw_samples and self._bbr_bw_samples[0][0] <= self._bbr_round - 10:
                self._bbr_bw_samples.popleft()
            if self._bbr_bw_samples and self._bbr_bw_samples[-1][0] == self._bbr_round:
                old_round, old_rate = self._bbr_bw_samples.pop()
                self._bbr_bw_samples.append((old_round, max(old_rate, rate_sample_pps)))
            else:
                self._bbr_bw_samples.append((self._bbr_round, rate_sample_pps))
            self.btlbw_pps = max(sample[1] for sample in self._bbr_bw_samples)
        if round_end:
            self._bbr_round += 1
            if self.mode == "Startup" and self._bbr_round_limited and self.btlbw_pps > 0:
                if self.btlbw_pps >= 1.25 * self._bbr_full_bw:
                    self._bbr_full_bw = self.btlbw_pps
                    self._bbr_full_bw_rounds = 0
                else:
                    self._bbr_full_bw_rounds += 1
                if self._bbr_full_bw_rounds >= 3:
                    self._bbr_full_pipe = True
                    self.mode = "Drain"
            self._bbr_round_limited = False
        if not math.isfinite(self.min_rtt_s):
            return
        # Initial bandwidth fallback uses ONLY measured RTT and initial cwnd.
        bw = self.btlbw_pps or self.initial_cwnd / self.min_rtt_s
        bdp = max(1.0, bw * self.min_rtt_s)
        if self.mode == "Drain" and inflight_packets <= max(4.0, bdp):
            self.mode = "ProbeBW"
            self._bbr_cycle_index = 0
            self._bbr_cycle_stamp = now
            self._bbr_loss_in_cycle = False
        if self.mode == "ProbeRTT":
            self.cwnd = 4.0
            self.pacing_rate_pps = bw
            if self._bbr_probe_done_at is None and inflight_packets <= 4.0:
                self._bbr_probe_done_at = now + 0.2
                self._bbr_probe_round = self._bbr_round
                return
            elif (self._bbr_probe_done_at is not None and
                  now >= self._bbr_probe_done_at and
                  self._bbr_round > self._bbr_probe_round):
                self._bbr_min_rtt_stamp = now
                self.cwnd = max(self.cwnd, self._bbr_prior_cwnd)
                self.mode = "ProbeBW" if self._bbr_full_pipe else "Startup"
                self._bbr_cycle_index = 0
                self._bbr_cycle_stamp = now
            else:
                return
        if self.mode == "ProbeBW":
            gain = self.BBR_GAINS[self._bbr_cycle_index]
            elapsed = now - self._bbr_cycle_stamp >= self.min_rtt_s
            advance = elapsed
            if gain > 1.0:
                advance = elapsed and (inflight_packets >= gain * bdp or self._bbr_loss_in_cycle)
            elif gain < 1.0:
                advance = elapsed or inflight_packets <= bdp
            if advance:
                self._bbr_cycle_index = (self._bbr_cycle_index + 1) % len(self.BBR_GAINS)
                self._bbr_cycle_stamp = now
                self._bbr_loss_in_cycle = False
            pacing_gain = self.BBR_GAINS[self._bbr_cycle_index]
            cwnd_gain = 2.0
        elif self.mode == "Drain":
            pacing_gain, cwnd_gain = 1.0 / self.BBR_HIGH_GAIN, self.BBR_HIGH_GAIN
        else:
            pacing_gain = cwnd_gain = self.BBR_HIGH_GAIN
        desired_rate = max(1e-9, pacing_gain * bw)
        # Replace the unmeasured bootstrap on the first ACK, then retain the
        # largest Startup pacing rate. Drain/ProbeBW may reduce it.
        if not self._bbr_has_measured_rate:
            self._bbr_has_measured_rate = True
            self.pacing_rate_pps = desired_rate
        elif self.mode == "Startup":
            self.pacing_rate_pps = max(self.pacing_rate_pps, desired_rate)
        else:
            self.pacing_rate_pps = desired_rate
        target_cwnd = max(4.0, cwnd_gain * bdp)
        if self._bbr_full_pipe:
            growth = acked_packets if network_limited else 0.0
            self.cwnd = max(4.0, min(self.cwnd + growth, target_cwnd))
        elif network_limited and (self.cwnd < target_cwnd or
                                  self._bbr_delivered < self.initial_cwnd):
            self.cwnd = max(4.0, self.cwnd + acked_packets)

    def on_ack(self, now: float, acked_packets: float,
               rtt_s: Optional[float], rate_sample_pps: Optional[float],
               inflight_packets: float, cwnd_limited: bool,
               round_end: bool, smoothed_rtt_s: Optional[float] = None) -> None:
        """Consume a cumulative ACK for new data; duplicate ACKs stay outside."""
        if acked_packets <= 0:
            return
        if rtt_s is not None and (not math.isfinite(rtt_s) or rtt_s <= 0):
            rtt_s = None
        if smoothed_rtt_s is not None:
            self.srtt_s = smoothed_rtt_s
        elif rtt_s is not None:
            self.srtt_s = rtt_s if self.srtt_s is None else (
                0.875 * self.srtt_s + 0.125 * rtt_s
            )
        if self.algo == "BBR":
            self._bbr_ack(now, acked_packets, rtt_s, rate_sample_pps,
                          inflight_packets, cwnd_limited, round_end)
        elif self.algo == "Vegas":
            self._vegas_ack(acked_packets, rtt_s, cwnd_limited, round_end)
        elif self.algo == "CUBIC":
            if not cwnd_limited:
                if self._cubic_pause_at is None:
                    self._cubic_pause_at = now
                return
            if self._cubic_pause_at is not None:
                if self._cubic_epoch is not None:
                    self._cubic_epoch += now - self._cubic_pause_at
                self._cubic_pause_at = None
            self._cubic_increase(now, acked_packets)
        elif cwnd_limited:
            self._reno_increase(acked_packets)

    def on_loss(self, now: float, flight_packets: float, timeout: bool,
                repeated_timeout: bool = False) -> None:
        """Respond once to a detected loss event, not once per dropped packet."""
        self._ca_credit = 0.0
        if self.algo == "BBR":
            # This demonstrator omits BBRv1's packet-conservation recovery,
            # ACK aggregation compensation and long-term policer detection.
            self._bbr_loss_in_cycle = True
            if timeout:
                self.cwnd = 1.0
            return
        beta = self.CUBIC_BETA if self.algo == "CUBIC" else 0.5
        if not (timeout and repeated_timeout):
            self.ssthresh = max(2.0, beta * flight_packets)
        if self.algo == "CUBIC":
            previous = self._cubic_wmax
            self._cubic_prior = self.cwnd
            self._cubic_wmax = self.cwnd
            if self.fast_convergence and previous is not None and self.cwnd < previous:
                self._cubic_wmax *= (1.0 + self.CUBIC_BETA) / 2.0
            self._cubic_epoch = None
            self._cubic_pause_at = None
            self._cubic_after_timeout = timeout
        self.cwnd = 1.0 if timeout or self.algo == "Tahoe" else self.ssthresh
        self.mode = "Slow start" if self.cwnd < self.ssthresh else "Congestion avoidance"
        if self.algo == "Vegas":
            self._vegas_restart = True
            self._vegas_samples = 0
            self._vegas_min_rtt = math.inf
            self._vegas_round_limited = False
            self._vegas_round_cwnd = self.cwnd
            self._vegas_grow_round = True

    def on_idle(self, now: float) -> None:
        """Pause CUBIC's epoch; discard Vegas's incomplete measurement round."""
        if self.algo == "CUBIC" and self._cubic_pause_at is None:
            self._cubic_pause_at = now
        if self.algo == "Vegas":
            self._vegas_restart = True
            self._vegas_samples = 0
            self._vegas_min_rtt = math.inf
            self._vegas_round_limited = False


class _PacketBudgetReached(Exception):
    pass


class PacketSimulator:
    """Saturated sender, FIFO drop-tail bottleneck and cumulative ACK receiver.

    Loss decisions belong to the network, never to a congestion controller.
    Data and ACK events preserve sequence numbers, so three duplicate ACKs
    and RTOs arise from actual receiver feedback. No ACK loss, headers, SACK,
    ECN, limited transmit, persist probes, handshake or competing flows are
    modeled. Receiver window is static; the application reads immediately.
    """
    def __init__(self, algo: str, path: Path, duration_s: float, seed: int = 42,
                 initial_window: str = "rfc5681", sample_interval_s: float = 0.1,
                 max_packets: int = 250000, ack_every: int = 1):
        if algo not in ALGORITHMS:
            raise ValueError("Unknown algorithm: " + algo)
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not math.isfinite(sample_interval_s) or sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be finite and positive")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(max_packets, int) or isinstance(max_packets, bool) or max_packets <= 0:
            raise ValueError("max_packets must be a positive integer")
        if ack_every not in (1, 2):
            raise ValueError("ack_every must be 1 or 2")
        self.algo, self.path, self.duration_s = algo, path, duration_s
        self.sample_interval_s = max(sample_interval_s, duration_s / 5000)
        self.max_packets, self.ack_every = max_packets, ack_every
        self.cc = CongestionController(algo, path.mss_bytes,
                                       initial_window_packets(path.mss_bytes, initial_window))
        self.trace = PathTrace(path, seed)
        self.loss_rng = random.Random("tcpsim:loss:%d" % seed)
        self.timer = RTOEstimator()
        self.now = 0.0
        self._heap = []  # type: List[Tuple[float, int, int, str, Any]]
        self._event_id = 0
        self._timer_generation = 0
        self._timer_active = False
        self._pace_pending = False
        self._next_pace = 0.0
        self._queue = deque()  # type: Deque[float]
        self.outstanding = OrderedDict()  # type: OrderedDict[int, Segment]
        self.snd_una = self.snd_nxt = self.rcv_next = 0
        self._receiver_buffer = {}  # type: Dict[int, int]
        self._pending_ack_count = 0
        self._ack_generation = 0
        self._ack_delay_s = 0.040
        self._last_reverse_delay = path.rtt_ms / 2000
        self._dupacks = 0
        self._in_fast_recovery = False
        self._fast_retransmit_seq = None  # type: Optional[int]
        self._rto_replay_end = None  # type: Optional[int]
        self._rto_cursor = 0
        self._round_end_seq = 0
        self._timed_seq = None  # type: Optional[int]
        self._karn_until = 0
        self._last_rtt = None  # type: Optional[float]
        self._rtt_sum = 0.0
        self._rtt_count = 0
        self._sender_delivered = 0
        self._last_delivery_time = 0.0
        self._first_flight_send = 0.0
        self._cwnd_integral = self._effective_integral = 0.0
        self._last_sample_time = 0.0
        self._sample_counts = (0, 0, 0, 0)
        self._counts = dict(sent_bytes=0, sent_packets=0, delivered_bytes=0,
                            unique_received_bytes=0, retransmissions=0,
                            random_drops=0, queue_drops=0, fast_retransmits=0,
                            timeouts=0)
        self.samples = []  # type: List[Dict[str, Any]]
        self.events = []  # type: List[Dict[str, Any]]
        self._omitted_events = 0
        self._used = False

    @property
    def flight_bytes(self) -> int:
        return self.snd_nxt - self.snd_una

    def _schedule(self, time_s: float, kind: str, payload: Any = None) -> None:
        self._event_id += 1
        # At equal timestamps, process ACKs before timer expiration and take
        # samples last, including any packet received exactly at the endpoint.
        priority = 2 if kind == "sample" else (1 if kind == "timeout" else 0)
        heapq.heappush(self._heap, (time_s, priority, self._event_id, kind, payload))

    def _advance(self, time_s: float) -> None:
        dt = time_s - self.now
        self._cwnd_integral += dt * self.cc.cwnd
        self._effective_integral += dt * effective_cwnd_pkts(self.cc.cwnd, self.path)
        self.now = time_s
        while self._queue and self._queue[0] <= self.now:
            self._queue.popleft()

    def _log(self, kind: str, before: Optional[float] = None,
             seq: Optional[int] = None) -> None:
        # The counters remain exact even when the detailed log is bounded.
        if len(self.events) >= 10000:
            self._omitted_events += 1
            return
        self.events.append(dict(time_s=self.now, event_type=kind, seq=seq,
                                algo=self.algo,
                                cwnd_before=self.cc.cwnd if before is None else before,
                                cwnd_pkts=self.cc.cwnd,
                                ssthresh_pkts=self.cc.ssthresh,
                                flight_pkts=self.flight_bytes / self.path.mss_bytes,
                                rto_s=self.timer.rto))

    def _arm_timer(self) -> None:
        self._timer_generation += 1
        self._timer_active = bool(self.outstanding)
        if self._timer_active:
            self._schedule(self.now + self.timer.rto, "timeout", self._timer_generation)

    def _random_drop(self, seq: int, transmissions: int) -> bool:
        """Override in deterministic experiments to inject a particular loss."""
        return self.loss_rng.random() < self.path.loss_prob

    def _transmit(self, segment: Segment, retransmission: bool = False) -> None:
        if self._counts["sent_packets"] >= self.max_packets:
            raise _PacketBudgetReached()
        if retransmission:
            segment.transmissions += 1
            segment.last_sent = self.now
            self._counts["retransmissions"] += 1
            # Without timestamps, avoid also timing later cumulative ACKs
            # that cover this recovery flight (a conservative Karn guard).
            self._karn_until = max(self._karn_until, self.snd_nxt)
        segment.delivered_at_send = self._sender_delivered
        segment.delivered_time_at_send = self._last_delivery_time
        segment.first_flight_send = self._first_flight_send
        self._counts["sent_packets"] += 1
        self._counts["sent_bytes"] += segment.size
        if not self._timer_active:
            self._arm_timer()
        if len(self._queue) >= self.path.queue_packets + 1:
            self._counts["queue_drops"] += 1
            self._log("queue_drop", seq=segment.seq)
            return
        service_start = max(self.now, self._queue[-1] if self._queue else self.now)
        capacity, _ = self.trace.at(service_start)
        _, propagation_rtt = self.trace.at(self.now)
        finish = service_start + 8 * segment.size / (capacity * 1e6)
        self._queue.append(finish)
        # Independent corruption occurs AFTER using bottleneck capacity.
        if self._random_drop(segment.seq, segment.transmissions):
            self._counts["random_drops"] += 1
            self._log("random_drop", seq=segment.seq)
        else:
            self._schedule(finish + propagation_rtt / 2, "receive",
                           (segment.seq, segment.size, propagation_rtt / 2))

    def _pump(self) -> None:
        mss = self.path.mss_bytes
        size = mss if self.path.rwnd_bytes is None else min(mss, self.path.rwnd_bytes)
        if size == 0:
            return
        limit = effective_cwnd_pkts(self.cc.cwnd, self.path) * mss
        if self._rto_replay_end is not None:
            if self.snd_una >= self._rto_replay_end:
                self._rto_replay_end = None
            else:
                # An RTO abandons the old ACK clock. Replay the outstanding
                # flight under the new slow-start window; counting the entire
                # pre-timeout FlightSize against cwnd=1 would deadlock this
                # clock and incorrectly require a timeout for every hole.
                # No SACK scoreboard: some buffered data may be retransmitted.
                self._rto_cursor = max(self._rto_cursor, self.snd_una)
                while self._rto_cursor < self._rto_replay_end:
                    segment = self.outstanding[self._rto_cursor]
                    pipe_bytes = max(0, self._rto_cursor - self.snd_una)
                    if pipe_bytes + segment.size > limit + 1e-8:
                        break
                    self._transmit(segment, True)
                    self._rto_cursor += segment.size
                return
        while self.flight_bytes + size <= limit + 1e-8:
            pacing = self.cc.pacing_rate_pps
            if pacing is not None and self.now + 1e-12 < self._next_pace:
                if not self._pace_pending:
                    self._schedule(self._next_pace, "pace")
                    self._pace_pending = True
                return
            if self._counts["sent_packets"] >= self.max_packets:
                raise _PacketBudgetReached()
            if not self.outstanding:
                self._first_flight_send = self.now
                self._last_delivery_time = self.now
            segment = Segment(self.snd_nxt, size, self.now, self.now)
            self.outstanding[segment.seq] = segment
            self.snd_nxt += size
            if self._timed_seq is None:
                self._timed_seq = segment.seq
            self._transmit(segment)
            if pacing is not None:
                self._next_pace = self.now + (size / mss) / max(pacing, 1e-9)

    def _send_ack(self, reverse_delay: float) -> None:
        self._pending_ack_count = 0
        self._ack_generation += 1
        self._schedule(self.now + reverse_delay, "ack", self.rcv_next)

    def _receive(self, seq: int, size: int, reverse_delay: float) -> None:
        previous_next = self.rcv_next
        gap_or_duplicate = seq != previous_next
        already_buffered = seq in self._receiver_buffer
        had_gap = bool(self._receiver_buffer)
        if seq >= self.rcv_next and not already_buffered:
            self._receiver_buffer[seq] = size
            self._counts["unique_received_bytes"] += size
        while self.rcv_next in self._receiver_buffer:
            delivered = self._receiver_buffer.pop(self.rcv_next)
            self.rcv_next += delivered
            self._counts["delivered_bytes"] += delivered
        self._last_reverse_delay = reverse_delay
        # Out-of-order, duplicate and gap-filling packets ACK immediately.
        if gap_or_duplicate or had_gap or already_buffered:
            self._send_ack(reverse_delay)
            return
        self._pending_ack_count += 1
        if self._pending_ack_count >= self.ack_every:
            self._send_ack(reverse_delay)
        elif self._pending_ack_count == 1:
            self._ack_generation += 1
            self._schedule(self.now + self._ack_delay_s, "delayed_ack", self._ack_generation)

    def _ack(self, ack: int) -> None:
        if ack < self.snd_una or ack > self.snd_nxt or not self.outstanding:
            return
        if ack == self.snd_una:
            self._dupacks += 1
            if (self._dupacks == 3 and self._fast_retransmit_seq != ack
                    and self._rto_replay_end is None):
                before = self.cc.cwnd
                self.cc.on_loss(self.now, self.flight_bytes / self.path.mss_bytes, False)
                self._fast_retransmit_seq = ack
                self._in_fast_recovery = self.algo in ("Reno", "CUBIC", "Vegas")
                if self._in_fast_recovery:
                    self.cc.cwnd = self.cc.ssthresh + 3.0
                self._counts["fast_retransmits"] += 1
                self._log("dup3ack", before, ack)
                self._transmit(next(iter(self.outstanding.values())), True)
                # Restarting here is conservative: no RTO retransmission can
                # occur less than an RTO after this fast retransmission.
                self._arm_timer()
            elif self._dupacks > 3 and self._in_fast_recovery:
                self.cc.cwnd += 1.0
            self._pump()
            return
        before_flight = self.flight_bytes
        previous_una = self.snd_una
        acked_segments = []
        while self.outstanding:
            first = next(iter(self.outstanding.values()))
            if first.seq + first.size > ack:
                break
            acked_segments.append(self.outstanding.popitem(last=False)[1])
        if not acked_segments:
            return
        self.snd_una = ack
        newly_acked = ack - previous_una
        self._sender_delivered += newly_acked
        ambiguous = previous_una < self._karn_until or any(s.transmissions > 1 for s in acked_segments)
        rtt_sample = None
        rate_sample = None
        if not ambiguous:
            reference = acked_segments[-1]
            rtt_sample = self.now - reference.first_sent
            # Delivery sampling limits ACK compression using both intervals.
            interval = max(self.now - reference.delivered_time_at_send,
                           reference.last_sent - reference.first_flight_send)
            if interval > 0:
                rate_sample = (self._sender_delivered - reference.delivered_at_send) / (
                    self.path.mss_bytes * interval)
            # Advance the send-interval baseline with delivered packets.
            # Leaving it at connection start makes a long-lived BBR flow
            # incorrectly divide a flight of delivered bytes by its entire age.
            self._first_flight_send = reference.last_sent
            self._last_rtt = rtt_sample
            self._rtt_sum += rtt_sample
            self._rtt_count += 1
        if self._timed_seq is not None and self._timed_seq < ack:
            timed = next((s for s in acked_segments if s.seq == self._timed_seq), None)
            if timed is not None and not ambiguous:
                self.timer.observe(self.now - timed.first_sent)
            self._timed_seq = None
        was_recovery = self._in_fast_recovery
        if was_recovery:
            # Classic Reno exits on the FIRST advancing ACK, even a partial
            # ACK. This is intentionally not RFC 6582 NewReno recovery.
            self.cc.cwnd = self.cc.ssthresh
            self._in_fast_recovery = False
        self._dupacks = 0
        self._fast_retransmit_seq = None
        round_end = ack >= self._round_end_seq
        if round_end:
            self._round_end_seq = self.snd_nxt
        mss = self.path.mss_bytes
        # Before the ACK freed space, was the sender using its available cwnd?
        size = mss if self.path.rwnd_bytes is None else min(mss, self.path.rwnd_bytes)
        limited = before_flight + size > self.cc.cwnd * mss - 1e-8
        receiver_limited = self.path.rwnd_bytes is not None and self.path.rwnd_bytes < self.cc.cwnd * mss
        limited = not receiver_limited if self.algo == "BBR" else limited and not receiver_limited
        self.cc.on_ack(self.now, newly_acked / mss, rtt_sample, rate_sample,
                       self.flight_bytes / mss, limited and not was_recovery, round_end,
                       smoothed_rtt_s=self.timer.srtt)
        self._last_delivery_time = self.now
        self._arm_timer()  # New ACK: restart, or stop if all data is ACKed.
        self._pump()

    def _timeout(self, generation: int) -> None:
        if generation != self._timer_generation or not self.outstanding:
            return
        segment = next(iter(self.outstanding.values()))
        before = self.cc.cwnd
        self.cc.on_loss(self.now, self.flight_bytes / self.path.mss_bytes,
                        True, repeated_timeout=segment.timer_transmissions > 0)
        segment.timer_transmissions += 1
        self._in_fast_recovery = False
        # Old-flight duplicate ACKs can still arrive after an RTO. Do not
        # immediately undo the loss window by fast-retransmitting this same
        # sequence again; an advancing cumulative ACK releases the guard.
        self._fast_retransmit_seq = segment.seq
        self._dupacks = 0
        self._counts["timeouts"] += 1
        self.timer.backoff()
        self._log("timeout", before, segment.seq)
        self._transmit(segment, True)
        self._rto_replay_end = self.snd_nxt
        self._rto_cursor = segment.seq + segment.size
        self._arm_timer()

    def _sample(self) -> None:
        dt = self.now - self._last_sample_time
        keys = ("sent_bytes", "delivered_bytes", "unique_received_bytes", "retransmissions")
        current = tuple(self._counts[k] for k in keys)
        rates = [(current[i] - self._sample_counts[i]) * 8 / (dt * 1e6)
                 if dt > 0 else 0.0 for i in range(3)]
        capacity, propagation = self.trace.at(self.now)
        self.samples.append(dict(
            time_s=self.now, interval_s=dt, algo=self.algo, algo_base=self.algo,
            cwnd_pkts=self.cc.cwnd, eff_cwnd_pkts=effective_cwnd_pkts(self.cc.cwnd, self.path),
            ssthresh_pkts=self.cc.ssthresh if math.isfinite(self.cc.ssthresh) else None,
            flight_pkts=self.flight_bytes / self.path.mss_bytes,
            send_mbps=rates[0], goodput_mbps=rates[1], received_mbps=rates[2],
            queue_pkts=max(0, len(self._queue) - 1),
            queue_delay_ms=max(0, self._queue[-1] - self.now) * 1000 if self._queue else 0.0,
            rtt_ms=self._last_rtt * 1000 if self._last_rtt is not None else None,
            srtt_ms=self.timer.srtt * 1000 if self.timer.srtt is not None else None,
            rto_s=self.timer.rto, capacity_mbps=capacity, base_rtt_ms=propagation * 1000,
            retransmissions=current[3] - self._sample_counts[3],
            phase="Fast recovery" if self._in_fast_recovery else self.cc.mode,
            delivered_bytes=current[1], sent_bytes=current[0]))
        self._last_sample_time = self.now
        self._sample_counts = current

    def run(self) -> SimulationResult:
        if self._used:
            raise RuntimeError("Create a fresh PacketSimulator for each run")
        self._used = True
        complete, stop_reason = True, "completed"
        self._sample()
        self._schedule(min(self.sample_interval_s, self.duration_s), "sample", 1)
        try:
            self._pump()
            self._round_end_seq = self.snd_nxt
            while self._heap:
                event_time, _, _, kind, payload = heapq.heappop(self._heap)
                if event_time > self.duration_s:
                    break
                self._advance(event_time)
                if kind == "receive":
                    self._receive(*payload)
                elif kind == "ack":
                    self._ack(payload)
                elif kind == "timeout":
                    self._timeout(payload)
                elif kind == "pace":
                    self._pace_pending = False
                    self._pump()
                elif kind == "delayed_ack":
                    if payload == self._ack_generation and self._pending_ack_count:
                        self._send_ack(self._last_reverse_delay)
                elif kind == "sample":
                    self._sample()
                    next_time = min((payload + 1) * self.sample_interval_s, self.duration_s)
                    if next_time > self.now + 1e-12:
                        self._schedule(next_time, "sample", payload + 1)
            self._advance(self.duration_s)
        except _PacketBudgetReached:
            complete = False
            stop_reason = "packet_budget_exhausted"
        # Use exactly the simulated interval, including the fractional tail.
        if self.now > self._last_sample_time:
            self._sample()
        elapsed = self.now
        summary = dict(algo=self.algo, duration_s=elapsed, requested_duration_s=self.duration_s,
                       **self._counts,
                       avg_goodput_mbps=self._counts["delivered_bytes"] * 8 / (elapsed * 1e6) if elapsed else 0.0,
                       avg_send_mbps=self._counts["sent_bytes"] * 8 / (elapsed * 1e6) if elapsed else 0.0,
                       avg_cwnd_pkts=self._cwnd_integral / elapsed if elapsed else self.cc.cwnd,
                       avg_eff_cwnd_pkts=self._effective_integral / elapsed if elapsed else effective_cwnd_pkts(self.cc.cwnd, self.path),
                       avg_rtt_ms=1000 * self._rtt_sum / self._rtt_count if self._rtt_count else None,
                       outstanding_bytes=self.flight_bytes,
                       receiver_buffer_bytes=sum(self._receiver_buffer.values()),
                       omitted_log_events=self._omitted_events,
                       complete=complete)
        return SimulationResult(self.samples, self.events, summary, complete, stop_reason)


def run_simulation(algo: str, base_path: Path, duration_s: float, seed: int = 42,
                   **kwargs) -> SimulationResult:
    return PacketSimulator(algo, base_path, duration_s, seed, **kwargs).run()


def simulate(algo: str, base_path: Path, duration_s: float, seed: int = 42, **kwargs):
    """DataFrame adapter; summary, completion and event log are in df.attrs."""
    return run_simulation(algo, base_path, duration_s, seed, **kwargs).dataframe()

REFERENCES = {
    "RFC 5681 — slow start, avoidance e fast recovery": "https://www.rfc-editor.org/rfc/rfc5681.html",
    "RFC 6298 — RTO, backoff e Karn": "https://www.rfc-editor.org/rfc/rfc6298.html",
    "RFC 9438 — CUBIC (sostituisce RFC 8312)": "https://www.rfc-editor.org/rfc/rfc9438.html",
    "RFC 6928 — finestra iniziale maggiore (Experimental)": "https://www.rfc-editor.org/rfc/rfc6928.html",
    "RFC 9293 — TCP e controllo di flusso": "https://www.rfc-editor.org/rfc/rfc9293.html",
    "Vegas — articolo originale": "https://www.cs.princeton.edu/courses/archive/fall06/cos561/papers/vegas.pdf",
    "Vegas — implementazione Linux": "https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_vegas.c",
    "BBRv1 — implementazione Linux di riferimento": "https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_bbr.c",
    "BBRv3 — documento IETF (Internet-Draft, versione da verificare)": "https://datatracker.ietf.org/doc/draft-ietf-ccwg-bbr/",
}

MODEL_NOTES = """
**Ambito.** Un trasferimento persistente per esperimento, con connessione già
stabilita. Gli algoritmi sono eseguiti separatamente sullo stesso percorso;
non competono fra loro. Per questo non viene calcolato l'indice di Jain.

**Rete.** Pacchetti di payload MSS, un bottleneck FIFO con drop-tail, ritardo
di propagazione e serializzazione espliciti. La capacità è **payload equivalente**:
non comprende header TCP/IP, Ethernet o il costo degli ACK. Il buffer indicato
conta i pacchetti in attesa, oltre a quello in servizio. Le perdite i.i.d. sono
applicate dopo il servizio e quindi consumano capacità; l'overflow è distinto.
Le variazioni sono costanti a tratti di un secondo, uguali per tutti gli
algoritmi. Un pacchetto mantiene la velocità scelta all'inizio del suo servizio
e il ritardo di propagazione scelto all'invio; variazioni di RTT possono
riordinare i pacchetti. Le sequenze casuali sono separate e riproducibili;
non implicano che algoritmi con invii differenti perdano gli stessi byte.

**TCP.** ACK cumulativi; a scelta uno ogni segmento o ogni due segmenti
in ordine (attesa massima 40 ms). I pacchetti fuori ordine e il riempimento
di un buco generano ACK immediati. Fast retransmit dopo tre ACK duplicati
reali; recupero **Reno classico**, anche per CUBIC e Vegas. Un ACK parziale
termina tale recupero: non viene simulato NewReno. RTO minimo e iniziale di
1 s, massimo 60 s, SRTT/RTTVAR con una misura alla volta e regola di Karn.
La soglia usa FlightSize; timeout ripetuti dello stesso segmento non la
riducono di nuovo. La finestra ricevitore è statica: zero blocca i dati,
senza simulare persist probes; valori inferiori a MSS usano segmenti piccoli.

**Controller.** Tahoe è storico. Reno segue crescita su ACK e recupero
RFC 5681. CUBIC usa le equazioni RFC 9438 con tempo in secondi, regione
Reno-friendly, beta 0,7 e slow start classico. Fast convergence è disattivata
per questi esperimenti con un solo flusso; HyStart++ non è implementato.
Vegas è una variante didattica: formula originale Expected−Actual × BaseRTT,
alpha=2, beta=4 e gamma=1, slow start alternato, recupero Reno.
**BBRv1-inspired** usa misure di consegna, filtro di banda su 10 round,
RTT minimo su 10 s e stati Startup, Drain, ProbeBW, ProbeRTT. Il pacing iniziale
usa un'ipotesi di RTT di 1 ms, sostituita dalle prime misure. Non implementa
packet conservation, compensazione di ACK aggregation o policer detection;
non rappresenta BBRv3. La proposta IETF BBR consultata è un Internet-Draft,
non una RFC pubblicata.

**Metriche.** Il goodput conta soltanto byte unici consegnati **in ordine
all'applicazione**. La consegna di dati precedentemente bloccati può produrre
picchi sopra la capacità nei singoli intervalli. «Ricezione unica» conta i
byte nuovi anche fuori ordine; «Invio» include le ritrasmissioni e può avere
burst sopra la capacità. Le medie sono byte/tempo; le finestre medie sono
integrate nel tempo. I marker degli eventi indicano la finestra **dopo** la
reazione; gli scarti di rete sono distinti dalla loro successiva rilevazione.
I grafici campionano lo stato, mentre il registro conserva i tempi degli eventi.

**Limiti.** Nessun SACK, ECN, RACK/TLP, limited transmit, ACK persi,
handshake, traffico concorrente o pila TCP completa. Le traiettorie servono
allo studio dei meccanismi: non certificano conformità RFC e non sono
benchmark di Linux o di altri sistemi operativi. Il limite di lavoro interrompe
esplicitamente simulazioni troppo grandi, senza inventare il tempo restante.
"""

AUDIT_ROWS = [
    ("Coda assente e banda troncata", "FIFO finita, serializzazione, accodamento e overflow"),
    ("Eventi di perdita classificati casualmente", "ACK duplicati e timeout prodotti dai pacchetti"),
    ("Goodput atteso = invio × (1 − p)", "Byte unici consegnati; ritrasmissioni e attese contabilizzate"),
    ("Soglia dalla cwnd già aumentata", "FlightSize al rilevamento della perdita"),
    ("CUBIC cresce per numero di RTT", "Tempo in secondi e crescita per ACK secondo RFC 9438"),
    ("Vegas senza ritardo di coda", "RTT misurato e crescita/uscita dallo slow start basate sul ritardo"),
    ("BBR conosce la capacità e oscilla artificialmente", "Stima dalle consegne e macchina a stati BBRv1-inspired"),
    ("Seed dipendente da hash(algo)", "RNG deterministici separati e percorso indicizzato nel tempo"),
    ("rwnd zero o sub-MSS arrotondata", "Limite in byte, zero distinto da illimitato"),
    ("Medie dei campioni e Jain su prove isolate", "Medie pesate nel tempo; nessuna pretesa di equità condivisa"),
    ("Interfaccia eseguita all'import", "Motore importabile, main() separato e dipendenze UI caricate al bisogno"),
]


def main() -> None:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="TCP · laboratorio a eventi", layout="wide")
    st.title("Controllo della congestione TCP")
    st.caption("Laboratorio a eventi: pacchetti, ACK, coda e ritrasmissioni. "
               "Confronto fra trasferimenti indipendenti; modelli didattici con riferimenti RFC.")
    with st.sidebar:
        with st.form("simulation_settings"):
            st.header("Collegamento")
            bw = st.number_input("Capacità payload equivalente (Mbps)", 0.01, 10000.0, 20.0, 1.0)
            rtt = st.number_input("RTT di propagazione (ms)", 1.0, 2000.0, 100.0, 1.0)
            queue = st.number_input("Buffer: pacchetti in attesa", 0, 10000, 64, 1)
            loss_pct = st.number_input("Perdita casuale per pacchetto (%)", 0.0, 100.0, 0.1, 0.01, format="%.4f")
            mss = st.selectbox("MSS: payload in byte", [536, 1200, 1360, 1460, 1500], index=3)
            use_rwnd = st.checkbox("Limita finestra ricevitore", False)
            rwnd_kib = st.number_input("rwnd (KiB; 0 blocca i dati)", 0.0, 1048576.0, 64.0, 1.0,
                                       help="1 KiB = 1024 byte; la finestra resta costante durante la prova.")
            bw_var = st.slider("Variazione capacità ± (%)", 0, 50, 0)
            rtt_var = st.slider("Variazione propagazione ± (%)", 0, 50, 0)
            st.header("Trasporto e simulazione")
            iw_label = st.selectbox("Finestra iniziale", ["RFC 5681 (2–4 MSS)",
                "RFC 6928 (fino a 10 MSS, Experimental)", "Storica (1 MSS)"])
            ack_choice = st.selectbox("ACK del ricevitore", ["Ogni segmento", "Ogni 2 segmenti (max 40 ms)"])
            duration = st.number_input("Durata richiesta (s)", 0.1, 600.0, 15.0, 0.5)
            seed = st.number_input("Seed", 0, 10000000, 42, 1)
            budget = st.selectbox("Limite trasmissioni per algoritmo", [50000, 250000, 1000000], index=1,
                                  help="Include le ritrasmissioni; l'eventuale interruzione viene segnalata.")
            selected = st.multiselect("Algoritmi", list(ALGORITHMS), default=["Reno", "CUBIC", "Vegas", "BBR"],
                                      format_func=lambda name: ALGORITHM_LABELS[name])
            submitted = st.form_submit_button("Esegui simulazione", type="primary")
        st.caption("Le modifiche si applicano premendo Esegui. "
                   "Il modello può richiedere diversi secondi per prove grandi.")

    if submitted or "tcp_results" not in st.session_state:
        if not selected:
            st.info("Seleziona almeno un algoritmo e premi Esegui simulazione.")
            return
        path = Path(bandwidth_mbps=bw, rtt_ms=rtt, loss_prob=loss_pct / 100,
                    mss_bytes=mss, queue_packets=queue,
                    bw_variation_frac=bw_var / 100, rtt_variation_frac=rtt_var / 100,
                    rwnd_bytes=int(round(rwnd_kib * 1024)) if use_rwnd else None)
        iw = {"RFC 5681 (2–4 MSS)": "rfc5681",
              "RFC 6928 (fino a 10 MSS, Experimental)": "rfc6928",
              "Storica (1 MSS)": "legacy1"}[iw_label]
        with st.spinner("Simulazione dei pacchetti e degli ACK…"):
            results = [run_simulation(name, path, duration, int(seed), initial_window=iw,
                         sample_interval_s=max(0.01, duration / 1000), max_packets=budget,
                         ack_every=1 if ack_choice == "Ogni segmento" else 2)
                       for name in selected]
        st.session_state["tcp_results"] = results
        st.session_state["tcp_path"] = path
        st.session_state["tcp_settings"] = {"seed": int(seed), "initial_window": iw,
             "duration_s": duration, "ack_every": 1 if ack_choice == "Ogni segmento" else 2,
             "max_packets": budget}
    results = st.session_state["tcp_results"]
    path = st.session_state["tcp_path"]
    settings = st.session_state["tcp_settings"]
    for result in results:
        if not result.complete:
            st.warning("%s: limite di trasmissioni raggiunto a %.3f s su %.3f s. "
                       "Statistiche parziali: riduci durata/banda o aumenta il limite." % (
                           ALGORITHM_LABELS[result.summary["algo"]], result.summary["duration_s"],
                           result.summary["requested_duration_s"]))
    columns = st.columns(4)
    columns[0].metric("BDP di propagazione (MSS)", "%.1f" % path.bdp_packets)
    columns[1].metric("Buffer in attesa", "%d pacchetti" % path.queue_packets)
    columns[2].metric("Perdita casuale", "%.4f%%" % (100 * path.loss_prob))
    columns[3].metric("RTT propagazione", "%g ms" % path.rtt_ms)
    st.caption("Risultati correnti: %.3g Mbps, %g s richiesti, seed %d, %s, ACK ogni %d segmento/i. "
               "La coda e la serializzazione si aggiungono al RTT di propagazione." % (
                   path.bandwidth_mbps, settings["duration_s"], settings["seed"],
                   settings["initial_window"], settings["ack_every"]))

    chart_tab, data_tab, method_tab = st.tabs(["Grafici e statistiche", "Dati ed eventi", "Modello, audit e fonti"])
    all_df = pd.DataFrame([row for result in results for row in result.samples])
    event_df = pd.DataFrame([event for result in results for event in result.events])
    summary = pd.DataFrame([result.summary for result in results])
    with chart_tab:
        metric_map = {"Finestra di congestione (MSS)": ("cwnd_pkts", "MSS"),
                      "Goodput applicativo (Mbps)": ("goodput_mbps", "Mbps"),
                      "Ricezione unica (Mbps)": ("received_mbps", "Mbps"),
                      "Invio, incluse ritrasmissioni (Mbps)": ("send_mbps", "Mbps"),
                      "RTT misurato (ms)": ("rtt_ms", "ms"),
                      "Coda: pacchetti in attesa": ("queue_pkts", "pacchetti"),
                      "RTO (s)": ("rto_s", "s")}
        choice = st.selectbox("Serie temporale", list(metric_map))
        metric, ylabel = metric_map[choice]
        show_events = st.checkbox("Segna fast retransmit e timeout sul grafico cwnd", True)
        rates = metric in ("goodput_mbps", "received_mbps", "send_mbps")
        fig = px.line(all_df, x="time_s", y=metric, color="algo", color_discrete_map=COLOR_MAP,
                      line_shape="vh" if rates else "hv",
                      hover_data=["phase", "flight_pkts", "rto_s"], render_mode="webgl")
        for trace in fig.data:
            trace.name = ALGORITHM_LABELS.get(trace.name, trace.name)
        if show_events and metric == "cwnd_pkts" and not event_df.empty:
            for event, label, symbol in [("dup3ack", "Fast retransmit (3 dupACK)", "x"),
                                         ("timeout", "Timeout", "diamond")]:
                subset = event_df[event_df["event_type"] == event]
                if not subset.empty:
                    fig.add_trace(go.Scatter(x=subset.time_s, y=subset.cwnd_pkts, mode="markers",
                        marker=dict(symbol=symbol, size=9, color="#333333"), name=label,
                        text=subset.algo,
                        hovertemplate="%{text}<br>t=%{x:.6f} s<br>cwnd dopo evento=%{y:.3f}<extra>" + label + "</extra>"))
        fig.update_layout(xaxis_title="Tempo (s)", yaxis_title=ylabel, legend_title_text="Algoritmo",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        if rates:
            st.caption("Ogni valore è la media nell'intervallo che termina al tempo indicato. "
                       "Invio e goodput applicativo possono avere picchi dovuti a burst e recupero di dati fuori ordine.")
        st.subheader("Statistiche sull'intervallo effettivamente simulato")
        visible = ["algo", "duration_s", "avg_goodput_mbps", "avg_send_mbps", "avg_cwnd_pkts",
                   "avg_eff_cwnd_pkts", "retransmissions", "random_drops", "queue_drops",
                   "fast_retransmits", "timeouts", "complete"]
        st.dataframe(summary[visible], use_container_width=True, hide_index=True)
        st.caption("Confronto di flussi isolati: nessuna misura di fairness. "
                   "I dati ancora in volo alla fine non contano come consegnati.")
    with data_tab:
        st.subheader("Campioni")
        st.dataframe(all_df, use_container_width=True, hide_index=True)
        st.download_button("Scarica campioni CSV", all_df.to_csv(index=False), "tcp_samples.csv", "text/csv")
        st.download_button("Scarica riepilogo CSV", summary.to_csv(index=False), "tcp_summary.csv", "text/csv")
        import json
        st.download_button("Scarica configurazione JSON", json.dumps(
            {"path": path.__dict__, "simulation": settings,
             "algorithms": [r.summary["algo"] for r in results]}, indent=2),
             "tcp_configuration.json", "application/json")
        st.subheader("Registro eventi")
        if event_df.empty:
            st.info("Nessuna perdita, fast retransmit o scadenza del timer registrata.")
        else:
            st.dataframe(event_df, use_container_width=True, hide_index=True)
            st.download_button("Scarica eventi CSV", event_df.to_csv(index=False), "tcp_events.csv", "text/csv")
        omitted = sum(r.summary["omitted_log_events"] for r in results)
        if omitted:
            st.warning("Registro limitato a 10.000 eventi per algoritmo: %d dettagli omessi. "
                       "I contatori riepilogativi includono tutti gli eventi." % omitted)
    with method_tab:
        st.subheader("Correzioni rispetto ad app.py")
        st.dataframe(pd.DataFrame(AUDIT_ROWS, columns=["Problema originale", "Nuova versione"]),
                     use_container_width=True, hide_index=True)
        st.markdown(MODEL_NOTES)
        st.subheader("Fonti primarie")
        for label, url in REFERENCES.items():
            st.markdown("- [%s](%s)" % (label, url))


if __name__ == "__main__":
    main()
