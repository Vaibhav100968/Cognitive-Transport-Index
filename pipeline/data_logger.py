"""
Subscribe to MQTT energy + game events and append synchronized rows to session_data.csv.
"""
import collections
import json
import os
import sys
import time

import pandas as pd
import paho.mqtt.client as mqtt

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
CSV_PATH = os.path.join(DATA_DIR, "session_data.csv")
COLUMNS = [
    "timestamp",
    "participant_id",
    "experiment_phase",
    "difficulty",
    "event_type",
    "scenario_id",
    "scenario_index",
    "reaction_time_ms",
    "choice",
    "raw_energy",
    "CTI",
    "window_id",
    "latency_ms",
]

game_buffers = collections.defaultdict(lambda: collections.deque(maxlen=200))
row_count = 0


def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_PATH, index=False)
        print(f"[Logger] Created {CSV_PATH}")
    else:
        print(f"[Logger] Appending to {CSV_PATH}")


def write_row(row_dict):
    global row_count
    row = {col: row_dict.get(col, None) for col in COLUMNS}
    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
    row_count += 1


def find_nearest_game_event(pid, timestamp, max_diff=2.0):
    best = None
    best_diff = max_diff
    for ev in game_buffers[pid]:
        diff = abs(ev.get("timestamp", 0) - timestamp)
        if diff < best_diff:
            best_diff = diff
            best = ev
    return best


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Logger] Connected to MQTT broker")
        client.subscribe("eeg/energy/+")
        client.subscribe("game/events/+")
        print(f"[Logger] Writing to {CSV_PATH}")
        print("[Logger] Subscribed to eeg/energy/+ and game/events/+")
    else:
        print(f"[Logger] Connection failed rc={rc}")


def on_message(client, userdata, msg):
    try:
        parts = msg.topic.split("/")
        payload = json.loads(msg.payload.decode())

        if parts[0] == "eeg" and parts[1] == "energy":
            pid = parts[2]
            ts = payload.get("timestamp", time.time())
            nearest = find_nearest_game_event(pid, ts)

            row = {
                "timestamp": ts,
                "participant_id": pid,
                "experiment_phase": payload.get("phase"),
                "difficulty": nearest.get("difficulty") if nearest else None,
                "event_type": "eeg_energy",
                "scenario_id": nearest.get("scenario_id") if nearest else None,
                "scenario_index": nearest.get("scenario_index") if nearest else None,
                "reaction_time_ms": None,
                "choice": None,
                "raw_energy": payload.get("raw_energy"),
                "CTI": payload.get("CTI"),
                "window_id": payload.get("window_id"),
                "latency_ms": payload.get("latency_ms"),
            }
            write_row(row)

            cti = payload.get("CTI")
            cti_str = f"{cti:.3f}" if cti is not None else "N/A"
            scenario_str = nearest.get("scenario_id", "?") if nearest else "?"
            print(f"[LOG] {pid} | CTI={cti_str} | scenario={scenario_str} | rows={row_count}")

        elif parts[0] == "game" and parts[1] == "events":
            pid = parts[2]
            payload["timestamp"] = payload.get("timestamp", time.time())
            game_buffers[pid].append(payload)

            event = payload.get("event", "")
            if event == "choice_made":
                row = {
                    "timestamp": payload["timestamp"],
                    "participant_id": pid,
                    "experiment_phase": payload.get("experiment_phase"),
                    "difficulty": payload.get("difficulty"),
                    "event_type": "game_choice",
                    "scenario_id": payload.get("scenario_id"),
                    "scenario_index": payload.get("scenario_index"),
                    "reaction_time_ms": payload.get("reaction_time_ms"),
                    "choice": payload.get("choice"),
                    "raw_energy": None,
                    "CTI": None,
                    "window_id": None,
                    "latency_ms": None,
                }
                write_row(row)
                print(
                    f"[GAME] {pid} | {payload.get('difficulty')} | "
                    f"RT={payload.get('reaction_time_ms')}ms | "
                    f"choice={payload.get('choice')} | scenario={payload.get('scenario_id')}"
                )

    except json.JSONDecodeError:
        print(f"[Logger] Bad JSON on {msg.topic}")
    except Exception as e:
        print(f"[Logger] Error: {e}")


def _make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


if __name__ == "__main__":
    ensure_csv()
    client = _make_client()
    client.on_connect = on_connect
    client.on_message = on_message
    print("[Logger] Connecting to MQTT broker localhost:1883...")
    try:
        client.connect("localhost", 1883, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n[Logger] Shutting down. Total rows written: {row_count}")
        client.disconnect()
    except ConnectionRefusedError:
        print("[Logger] Cannot connect. Run: brew services start mosquitto")
