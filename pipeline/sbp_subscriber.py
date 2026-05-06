import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import paho.mqtt.client as mqtt

from core.streaming_sbp import StreamingSBP

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
BROKER = "localhost"
PORT = 1883

participants = {}
# Each entry: {
#   'sbp': StreamingSBP or None,
#   'phase': 'calibration',
#   'baseline_rows': [],
#   'window_count': 0,
#   'easy_energies': [],
# }


def get_or_init_participant(pid):
    if pid not in participants:
        participants[pid] = {
            "sbp": None,
            "phase": "calibration",
            "baseline_rows": [],
            "window_count": 0,
            "easy_energies": [],
        }
    return participants[pid]


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[SBP Subscriber] Connected to MQTT broker {BROKER}:{PORT}")
        client.subscribe("eeg/features/+")
        client.subscribe("game/events/+")
        print("[SBP Subscriber] Subscribed to eeg/features/+ and game/events/+")
        print("[SBP Subscriber] Waiting for participants...")
    else:
        print(f"[SBP Subscriber] Connection failed rc={rc}")


def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        parts = topic.split("/")

        # ---- EEG FEATURES ----
        if parts[0] == "eeg" and parts[1] == "features":
            pid = parts[2]
            state = get_or_init_participant(pid)

            try:
                fvec = np.array([float(payload[f]) for f in FEATURES])
            except KeyError as e:
                print(f"[{pid}] Missing feature {e}, skipping")
                return

            if state["phase"] == "calibration":
                state["baseline_rows"].append(fvec)
                n = len(state["baseline_rows"])
                if n == 50 and state["sbp"] is None:
                    baseline = np.array(state["baseline_rows"])
                    state["sbp"] = StreamingSBP(baseline, FEATURES)
                    print(f"[{pid}] StreamingSBP initialized with {n} baseline samples")
                elif n % 50 == 0:
                    print(f"[{pid}] Calibration: {n} samples collected")
                return

            if state["sbp"] is None:
                if len(state["baseline_rows"]) >= 20:
                    baseline = np.array(state["baseline_rows"])
                    state["sbp"] = StreamingSBP(baseline, FEATURES)
                    state["sbp"].set_phase(state["phase"])
                    print(
                        f"[{pid}] StreamingSBP created late with {len(baseline)} baseline rows"
                    )
                else:
                    print(
                        f"[{pid}] Not enough baseline data yet ({len(state['baseline_rows'])} rows)"
                    )
                    return

            state["sbp"].add_sample(fvec)
            win_sz = state["sbp"].window_size

            if len(state["sbp"].buffer) >= win_sz:
                t_start = time.time()
                result = state["sbp"].compute_energy()
                latency_ms = (time.time() - t_start) * 1000

                if result is not None:
                    state["window_count"] += 1
                    if state["phase"] == "easy_test":
                        state["easy_energies"].append(result["raw_energy"])
                        if (
                            len(state["easy_energies"]) >= 3
                            and state["sbp"].mu_easy is None
                        ):
                            state["sbp"].mu_easy = float(
                                np.mean(state["easy_energies"])
                            )
                            state["sbp"].sigma_easy = (
                                float(np.std(state["easy_energies"])) + 1e-8
                            )
                            print(
                                f"[{pid}] CTI normalization set from "
                                f"{len(state['easy_energies'])} easy windows: "
                                f"mu={state['sbp'].mu_easy:.6f} "
                                f"sigma={state['sbp'].sigma_easy:.6f}"
                            )
                    out = {
                        "participant_id": pid,
                        "raw_energy": result["raw_energy"],
                        "CTI": result["CTI"],
                        "phase": result["phase"],
                        "window_id": state["window_count"],
                        "latency_ms": round(latency_ms, 1),
                        "timestamp": time.time(),
                    }
                    client.publish(f"eeg/energy/{pid}", json.dumps(out))

                    cti_str = (
                        f"{result['CTI']:.3f}"
                        if result["CTI"] is not None
                        else "calibrating"
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {pid} | "
                        f"Phase: {result['phase']:12s} | "
                        f"CTI: {cti_str:>8s} | "
                        f"Energy: {result['raw_energy']:.6f} | "
                        f"Latency: {latency_ms:.0f}ms | "
                        f"Window: #{state['window_count']}"
                    )

        # ---- GAME EVENTS ----
        elif parts[0] == "game" and parts[1] == "events":
            pid = parts[2]
            state = get_or_init_participant(pid)
            event = payload.get("event", "")
            new_phase = payload.get("experiment_phase", "")

            if event == "scenario_start" and new_phase:
                old_phase = state["phase"]
                if new_phase != old_phase:
                    state["phase"] = new_phase
                    if state["sbp"] is not None:
                        state["sbp"].set_phase(new_phase)
                    elif new_phase != "calibration" and len(state["baseline_rows"]) >= 20:
                        baseline = np.array(state["baseline_rows"])
                        state["sbp"] = StreamingSBP(baseline, FEATURES)
                        state["sbp"].set_phase(new_phase)
                    print(f"[{pid}] Phase: {old_phase} → {new_phase}")

            elif event == "session_start":
                state["phase"] = "calibration"
                state["baseline_rows"] = []
                state["sbp"] = None
                state["window_count"] = 0
                state["easy_energies"] = []
                print(f"[{pid}] New session started — reset state")

            elif event == "choice_made":
                print(
                    f"[{pid}] Choice: {payload.get('choice')} | "
                    f"RT: {payload.get('reaction_time_ms')}ms | "
                    f"Difficulty: {payload.get('difficulty')}"
                )

    except json.JSONDecodeError:
        print(f"[SBP Subscriber] Bad JSON on topic {msg.topic}")
    except Exception as e:
        print(f"[SBP Subscriber] Error: {e}")


def _make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


if __name__ == "__main__":
    client = _make_client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[SBP Subscriber] Connecting to {BROKER}:{PORT}...")
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SBP Subscriber] Shutting down.")
        client.disconnect()
    except ConnectionRefusedError:
        print(f"[SBP Subscriber] Cannot connect to {BROKER}:{PORT}")
        print("Make sure mosquitto is running: brew services start mosquitto")
