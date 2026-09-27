"""Quadruple-tank data generator.

Port of QTank_Dynamics.m + model_paramters.m, plus the simulation driver those
two files require but do not contain (initial levels, input design, disturbance
design, integration, noise application, sampling). The ODE and every physical
constant are taken verbatim from the .m files; see run_simulation.m for the
MATLAB equivalent of this driver.

Two departures from the .m files, both requested and both recorded here:

* The .m parameters encode only the paper's P+ (nonminimum-phase) operating
  point. P- is added as an override using the paper's published values
  (Johansson 2000, Table in Sec. II), NOT from the .m files.
* U_k and D_k arrive as function arguments in the .m file, so their design is
  chosen here: PRBS for both, matching the paper's identification experiments.

Usage:  python generate_qtank_data.py [--out DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

G = 981.0  # cm/s^2, gravitational constant (QTank_Dynamics.m line 5)

# Column order of the generated CSV. The two pump voltages and the disturbance
# are recorded, so the system is causally sufficient: v1 is a common cause of
# (h1, h4), v2 of (h2, h3), and d of (h3, h4). Omitting any of them would
# leave a latent confounder.
COLUMNS = ["v1", "v2", "d", "h1", "h2", "h3", "h4"]

#: The true cross-edges implied by the ODE, for scoring later. Self-loops
#: (every tank drains itself) are excluded.
GROUND_TRUTH_EDGES = [
    ("v1", "h1"),
    ("v1", "h4"),
    ("v2", "h2"),
    ("v2", "h3"),
    ("d", "h3"),
    ("d", "h4"),
    ("h3", "h1"),
    ("h4", "h2"),
]


@dataclass(frozen=True)
class QTankParams:
    """Physical constants. Defaults are model_paramters.m verbatim."""

    a1: float = 0.071
    A1: float = 28.0
    a3: float = 0.071
    A3: float = 28.0
    a2: float = 0.057
    A2: float = 32.0
    a4: float = 0.057
    A4: float = 32.0
    gam1: float = 0.43
    gam2: float = 0.34
    gam3: float = 0.40  # disturbance split between tanks 3 and 4
    k1: float = 3.14
    k2: float = 3.29
    meas_noise_std: float = 0.05  # sqrt of Q_mat
    input_noise_std: float = 0.05  # sqrt of R_mat


# P+ : gam1 + gam2 = 0.77 < 1, nonminimum phase. Straight from the .m files.
P_PLUS = QTankParams()

# P- : gam1 + gam2 = 1.30 > 1, minimum phase. From the paper, not the .m files.
P_MINUS = replace(P_PLUS, gam1=0.70, gam2=0.60, k1=3.33, k2=3.35)

# Steady-state levels and pump voltages at each operating point (paper, Sec. II).
OPERATING_POINTS = {
    "P_plus": {"params": P_PLUS, "h0": (12.6, 13.0, 4.8, 4.9), "v0": (3.15, 3.15)},
    "P_minus": {"params": P_MINUS, "h0": (12.4, 12.7, 1.8, 1.4), "v0": (3.00, 3.00)},
}


def qtank_dynamics(h: np.ndarray, u: np.ndarray, d: float, p: QTankParams) -> np.ndarray:
    """Port of QTank_Dynamics.m, line for line.

    Levels are clipped at zero before the square root: the model is only defined
    for non-negative levels, and a solver probe can otherwise step slightly
    negative and produce a NaN that silently poisons the whole trajectory.
    """
    h = np.maximum(h, 0.0)
    r = np.sqrt(2.0 * G * h)

    hdot = np.empty(4)
    hdot[0] = -(p.a1 / p.A1) * r[0] + (p.a3 / p.A1) * r[2] + (p.gam1 * p.k1 / p.A1) * u[0]
    hdot[1] = -(p.a2 / p.A2) * r[1] + (p.a4 / p.A2) * r[3] + (p.gam2 * p.k2 / p.A2) * u[1]
    hdot[2] = -(p.a3 / p.A3) * r[2] + ((1 - p.gam2) * p.k2 / p.A3) * u[1] + (p.gam3 / p.A3) * d
    hdot[3] = -(p.a4 / p.A4) * r[3] + ((1 - p.gam1) * p.k1 / p.A4) * u[0] + ((1 - p.gam3) / p.A4) * d
    return hdot


def prbs(n: int, hold: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Pseudorandom binary sequence, +/-amplitude, switching every `hold` samples.

    Binary rather than Gaussian because that is what the paper's identification
    experiments used, and because VAR-LiNGAM's identifiability rests on the
    excitation being non-Gaussian. A PRBS has excess kurtosis ~ -2, about as far
    from Gaussian as a bounded signal gets.
    """
    n_holds = int(np.ceil(n / hold))
    levels = rng.choice([-amplitude, amplitude], size=n_holds)
    return np.repeat(levels, hold)[:n]


def simulate(
    name: str,
    ts: float = 5.0,
    hours: float = 4.0,
    burn_in_s: float = 600.0,
    u_amplitude: float = 0.30,
    d_amplitude: float = 0.50,
    u_hold_s: float = 30.0,
    d_hold_s: float = 50.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Runs one operating point and returns (samples, metadata).

    The inputs are held constant across each sample interval (zero-order hold),
    which is how the real rig drives its pumps and what makes the discrete-time
    lag structure well defined.

    `u_hold_s` (30s) is a third of the dominant time constant (~90s): long
    enough for the tanks to respond to each switch, short enough to excite the
    system repeatedly within the run.  `d_hold_s` is deliberately different
    (50s) so the disturbance is not accidentally synchronised with the pumps,
    which would make their effects hard to separate.
    """
    spec = OPERATING_POINTS[name]
    p: QTankParams = spec["params"]
    rng = np.random.default_rng(seed)

    n_burn = int(round(burn_in_s / ts))
    n_keep = int(round(hours * 3600 / ts))
    n = n_burn + n_keep

    v0 = np.asarray(spec["v0"], dtype=float)
    # Commanded pump voltages: steady state plus an independent PRBS per pump.
    u_cmd = np.column_stack(
        [
            v0[0] + prbs(n, int(u_hold_s / ts), u_amplitude, rng),
            v0[1] + prbs(n, int(u_hold_s / ts), u_amplitude, rng),
        ]
    )
    d_cmd = prbs(n, int(d_hold_s / ts), d_amplitude, rng)

    # R_mat: "Unmeasured white noise disturbance in inputs". Unmeasured means the
    # plant sees command + noise while the CSV records the command, so the
    # recorded v1/v2 stay exactly binary and the noise acts as process noise.
    u_actual = u_cmd + rng.normal(0.0, p.input_noise_std, size=u_cmd.shape)

    h = np.asarray(spec["h0"], dtype=float)
    levels = np.empty((n, 4))
    for i in range(n):
        sol = solve_ivp(
            lambda _t, y, u=u_actual[i], d=d_cmd[i]: qtank_dynamics(y, u, d, p),
            (0.0, ts),
            h,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        h = sol.y[:, -1]
        levels[i] = h

    # Q_mat: measurement noise on the recorded levels.
    measured = levels + rng.normal(0.0, p.meas_noise_std, size=levels.shape)

    samples = np.column_stack([u_cmd, d_cmd, measured])[n_burn:]
    meta = {
        "operating_point": name,
        "gam1": p.gam1,
        "gam2": p.gam2,
        "gam1_plus_gam2": round(p.gam1 + p.gam2, 4),
        "phase": "minimum" if p.gam1 + p.gam2 > 1 else "nonminimum",
        "k1": p.k1,
        "k2": p.k2,
        "ts_seconds": ts,
        "hours": hours,
        "rows": int(samples.shape[0]),
        "burn_in_discarded": n_burn,
        "u_prbs_amplitude_V": u_amplitude,
        "u_prbs_hold_s": u_hold_s,
        "d_prbs_amplitude": d_amplitude,
        "d_prbs_hold_s": d_hold_s,
        "meas_noise_std": p.meas_noise_std,
        "input_noise_std": p.input_noise_std,
        "seed": seed,
        "time_constants_s": [
            round((p.A1 / p.a1) * np.sqrt(2 * spec["h0"][0] / G), 1),
            round((p.A2 / p.a2) * np.sqrt(2 * spec["h0"][1] / G), 1),
            round((p.A3 / p.a3) * np.sqrt(2 * spec["h0"][2] / G), 1),
            round((p.A4 / p.a4) * np.sqrt(2 * spec["h0"][3] / G), 1),
        ],
    }
    return samples, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import json

    for name in OPERATING_POINTS:
        samples, meta = simulate(name)
        csv_path = args.out / f"qtank_{name}.csv"
        np.savetxt(
            csv_path,
            samples,
            delimiter=",",
            header=",".join(COLUMNS),
            comments="",
            fmt="%.6f",
        )
        (args.out / f"qtank_{name}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{csv_path.name}: {samples.shape[0]} rows x {samples.shape[1]} cols  "
              f"({meta['phase']} phase, T={meta['time_constants_s']}s)")

    (args.out / "ground_truth_edges.json").write_text(
        json.dumps({"columns": COLUMNS, "edges": GROUND_TRUTH_EDGES}, indent=2)
    )
    print(f"ground truth: {len(GROUND_TRUTH_EDGES)} cross-edges -> ground_truth_edges.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
