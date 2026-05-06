"""
Real-time matplotlib dashboard: CTI from MQTT eeg/energy/+ (up to 4 participants).
"""
import collections
import json
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
from matplotlib.animation import FuncAnimation

BROKER = "localhost"
PORT = 1883
TOPIC = "eeg/energy/+"

MAX_PARTICIPANTS = 4
CTI_HISTORY = 150

PHASE_FACE = {
    "calibration": "#f0f0f0",
    "easy_test": "#e8f4f8",
    "hard_test": "#fce8e8",
}

participants_lock = threading.Lock()
participants = {}
# pid -> {
#   'ctis': deque(maxlen=150),
#   'phases': deque(maxlen=150),
#   'current_phase': str,
#   'latest_cti': float | None,
# }


def _make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC)
    else:
        print(f"[Dashboard] MQTT connect failed rc={rc}")


def on_mqtt_message(client, userdata, msg):
    try:
        parts = msg.topic.split("/")
        if len(parts) < 3 or parts[0] != "eeg" or parts[1] != "energy":
            return
        pid = parts[2]
        payload = json.loads(msg.payload.decode())
        cti = payload.get("CTI")
        phase = payload.get("phase") or "unknown"

        with participants_lock:
            if pid not in participants:
                participants[pid] = {
                    "ctis": collections.deque(maxlen=CTI_HISTORY),
                    "phases": collections.deque(maxlen=CTI_HISTORY),
                    "current_phase": phase,
                    "latest_cti": cti,
                }
            p = participants[pid]
            p["current_phase"] = phase
            p["latest_cti"] = cti
            if cti is not None:
                try:
                    p["ctis"].append(float(cti))
                    p["phases"].append(phase)
                except (TypeError, ValueError):
                    pass
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"[Dashboard] MQTT message error: {e}")


def mqtt_loop():
    client = _make_client()
    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"[Dashboard] MQTT loop error: {e}")


def _snapshot_participants():
    with participants_lock:
        return {
            pid: {
                "ctis": list(data["ctis"]),
                "phases": list(data["phases"]),
                "current_phase": data["current_phase"],
                "latest_cti": data["latest_cti"],
            }
            for pid, data in participants.items()
        }


def animate(frame, fig):
    snap = _snapshot_participants()
    fig.clf()

    if not snap:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "Waiting for EEG data...",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        fig.tight_layout()
        return

    pids = sorted(snap.keys())[:MAX_PARTICIPANTS]
    n = len(pids)
    fig.set_size_inches(10, max(4, 3 * n))

    for i, pid in enumerate(pids):
        ax = fig.add_subplot(n, 1, i + 1)
        row = snap[pid]
        phase = row["current_phase"]
        face = PHASE_FACE.get(phase, "#ffffff")
        ax.set_facecolor(face)

        ys = np.asarray(row["ctis"], dtype=float)
        xs = np.arange(len(ys))
        if ys.size > 0:
            ax.plot(xs, ys, color="C0", linewidth=1.5)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(2.0, color="red", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_ylabel("Cognitive Transport Index (CTI)")
        ax.set_xlabel("Window index (rolling)")
        ax.set_title(f"{pid}  |  Phase: {phase}")

        latest = row["latest_cti"]
        if latest is not None:
            try:
                ax.text(
                    0.98,
                    0.98,
                    f"CTI: {float(latest):.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            except (TypeError, ValueError):
                pass

    fig.tight_layout()


def main():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    print("[Dashboard] Live CTI Dashboard running")
    print("[Dashboard] Subscribed to eeg/energy/+")
    print("[Dashboard] Waiting for participants...")

    t = threading.Thread(target=mqtt_loop, daemon=True)
    t.start()
    time.sleep(0.3)

    fig = plt.figure(figsize=(10, 8))
    ani = FuncAnimation(
        fig,
        animate,
        fargs=(fig,),
        interval=2000,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
