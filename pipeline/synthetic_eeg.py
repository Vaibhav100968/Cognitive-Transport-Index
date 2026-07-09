"""
Publish synthetic 8-d EEG-like features over MQTT (no headset, no CSV).

Phases: calibration → easy_test → hard_test, matching the live session protocol
so StreamingSBP can initialize, normalize CTI, and emit energies end-to-end.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

FEATURES = [
    "Theta",
    "Alpha",
    "BetaL",
    "BetaH",
    "Gamma",
    "Arousal",
    "Valence",
    "Engagement",
]


def _make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def _sample_row(rng: np.random.Generator, loc: float, scale: float) -> dict:
    """Draw one feature vector; derived indices follow the same cloud for simplicity."""
    bands = rng.normal(loc=loc, scale=scale, size=5).astype(np.float64)
    # Keep derived indices in a plausible positive range for MQTT consumers.
    arousal = float(abs(bands[2] + bands[3]) / (abs(bands[0] + bands[1]) + 1e-3))
    valence = float(rng.normal(0.0, 0.3))
    engagement = float(abs(bands[2]) / (abs(bands[0] + bands[1]) + 1e-3))
    vals = list(bands) + [arousal, valence, engagement]
    return {name: float(v) for name, v in zip(FEATURES, vals)}


def main():
    p = argparse.ArgumentParser(description="Synthetic EEG MQTT publisher (demo)")
    p.add_argument("--participant", default="demo_player")
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--rate", type=float, default=10.0, help="Samples per second")
    p.add_argument(
        "--cal-samples",
        type=int,
        default=60,
        help="Calibration samples before easy/hard phases",
    )
    p.add_argument("--easy-samples", type=int, default=80)
    p.add_argument("--hard-samples", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    client = _make_client()
    try:
        client.connect(args.broker, args.port, 60)
    except ConnectionRefusedError:
        print(
            f"[synthetic] Cannot connect to MQTT at {args.broker}:{args.port}. "
            "Start mosquitto (or docker compose) first."
        )
        sys.exit(1)

    client.loop_start()
    stop = {"flag": False}

    def _on_sigint(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    dt = 1.0 / max(args.rate, 0.001)
    pid = args.participant
    feat_topic = f"eeg/features/{pid}"
    game_topic = f"game/events/{pid}"

    phases = [
        ("calibration", "easy", args.cal_samples, 0.0, 0.35),
        ("easy_test", "easy", args.easy_samples, 0.35, 0.45),
        ("hard_test", "hard", args.hard_samples, 1.1, 0.75),
    ]

    print(
        f"[synthetic] Publishing to {args.broker}:{args.port} as {pid!r} "
        f"({args.rate} Hz)"
    )

    try:
        # Reset subscriber state for a clean demo session.
        client.publish(
            game_topic,
            json.dumps(
                {
                    "event": "session_start",
                    "experiment_phase": "calibration",
                    "timestamp": time.time(),
                }
            ),
        )
        time.sleep(0.2)

        row_i = 0
        for phase, difficulty, n_samples, loc, scale in phases:
            if stop["flag"]:
                break
            client.publish(
                game_topic,
                json.dumps(
                    {
                        "event": "scenario_start",
                        "experiment_phase": phase,
                        "difficulty": difficulty,
                        "scenario_id": f"synthetic_{phase}",
                        "timestamp": time.time(),
                    }
                ),
            )
            print(f"[synthetic] phase={phase} n={n_samples}")

            for _ in range(n_samples):
                if stop["flag"]:
                    break
                payload = _sample_row(rng, loc, scale)
                payload["participant_id"] = pid
                payload["timestamp"] = time.time()
                client.publish(feat_topic, json.dumps(payload))
                row_i += 1
                if row_i % 25 == 0:
                    print(f"[synthetic] published {row_i} feature rows")
                time.sleep(dt)

        print(f"[synthetic] Done ({row_i} rows).")
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
