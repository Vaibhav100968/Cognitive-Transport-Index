"""
Replay baseline-z EEG CSV over MQTT for pipeline testing (no hardware).
"""
import argparse
import json
import os
import signal
import sys
import time

import pandas as pd
import paho.mqtt.client as mqtt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def portion_to_phase_difficulty(portion):
    if portion == 0:
        return "calibration", "easy"
    if portion == 1:
        return "easy_test", "easy"
    return "hard_test", "hard"


def main():
    p = argparse.ArgumentParser(description="Mock EEG MQTT publisher (CSV replay)")
    p.add_argument("--participant", default="player_1")
    p.add_argument("--rate", type=float, default=10.0, help="Rows per second")
    p.add_argument("--broker", default="localhost")
    args = p.parse_args()

    csv_path = os.path.join(_ROOT, "cleaned_vegs_data_baseline_z.csv")
    if not os.path.isfile(csv_path):
        print(f"[mock] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "Participant" not in df.columns or "portion" not in df.columns:
        print("[mock] CSV must include Participant and portion columns")
        sys.exit(1)

    col = df["Participant"].astype(str)
    uniq = col.unique()
    wanted = str(args.participant)
    if wanted in set(uniq):
        participant = wanted
    else:
        participant = None
        wlow = wanted.lower()
        for u in uniq:
            if str(u).lower() == wlow:
                participant = str(u)
                break
        if participant is None:
            participant = str(uniq[0])
            print(
                f"[mock] Participant {wanted!r} not in CSV; using {participant!r}"
            )

    sub = df[df["Participant"].astype(str) == str(participant)].copy()
    if sub.empty:
        print("[mock] No rows for selected participant")
        sys.exit(1)

    client = _make_client()
    client.connect(args.broker, 1883, 60)
    client.loop_start()

    stop = {"flag": False}

    def _on_sigint(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    dt = 1.0 / max(args.rate, 0.001)
    scenario_i = 0
    row_i = 0

    try:
        for portion in [0, 1, 2, 3]:
            if stop["flag"]:
                break
            part_df = sub[sub["portion"] == portion]
            if part_df.empty:
                continue

            phase, difficulty = portion_to_phase_difficulty(int(portion))
            game_topic = f"game/events/{participant}"
            game_payload = {
                "event": "scenario_start",
                "experiment_phase": phase,
                "difficulty": difficulty,
                "scenario_id": f"mock_{scenario_i}",
                "scenario_index": scenario_i,
                "timestamp": time.time(),
            }
            client.publish(game_topic, json.dumps(game_payload))
            scenario_i += 1

            for _, row in part_df.iterrows():
                if stop["flag"]:
                    break
                payload = {f: float(row[f]) for f in FEATURES}
                payload["participant_id"] = participant
                payload["timestamp"] = time.time()
                client.publish(
                    f"eeg/features/{participant}",
                    json.dumps(payload),
                )
                row_i += 1
                if row_i % 20 == 0:
                    print(
                        f"[mock] {participant} row {row_i} portion={portion}"
                    )
                time.sleep(dt)

        print("[mock] Finished all portions (or interrupted)")
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
