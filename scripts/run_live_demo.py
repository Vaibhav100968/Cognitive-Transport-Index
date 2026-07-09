#!/usr/bin/env python3
"""
Headless live demo: MQTT broker must already be reachable.

Starts SBP subscriber + data logger, publishes synthetic EEG, prints CTI
samples from the energy topic, then exits. No headset, no VEGS CSV, no GUI.

Usage:
  # Local mosquitto
  python scripts/run_live_demo.py

  # Docker Compose
  docker compose up --build
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_mqtt_client():
    import paho.mqtt.client as mqtt

    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def wait_for_broker(host: str, port: int, timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        c = _make_mqtt_client()
        try:
            c.connect(host, port, 5)
            c.disconnect()
            print(f"[demo] MQTT broker OK at {host}:{port}")
            return
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise SystemExit(f"[demo] Broker unreachable at {host}:{port} ({last_err})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    parser.add_argument("--participant", default="demo_player")
    parser.add_argument("--seconds", type=int, default=75, help="Max demo wall time")
    parser.add_argument("--rate", type=float, default=8.0)
    args = parser.parse_args()

    os.chdir(_ROOT)
    wait_for_broker(args.broker, args.port)

    py = sys.executable
    # Propagate broker settings so subscriber/logger match the publisher.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MQTT_BROKER"] = args.broker
    env["MQTT_PORT"] = str(args.port)

    procs: list[subprocess.Popen] = []
    stop = {"flag": False}
    samples: list[dict] = []

    def _terminate_all(*_):
        stop["flag"] = True
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    def spawn(rel: str, extra: list[str] | None = None):
        cmd = [py, rel] + (extra or [])
        proc = subprocess.Popen(
            cmd,
            cwd=_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        procs.append(proc)
        return proc

    print("[demo] Starting SBP subscriber + data logger…")
    spawn(os.path.join("pipeline", "sbp_subscriber.py"))
    time.sleep(1.5)
    spawn(os.path.join("pipeline", "data_logger.py"))
    time.sleep(1.0)

    listener = _make_mqtt_client()

    def on_message(_c, _u, msg):
        try:
            payload = json.loads(msg.payload.decode())
            samples.append(payload)
            cti = payload.get("CTI")
            cti_s = f"{cti:.3f}" if isinstance(cti, (int, float)) else "None"
            print(
                f"[demo] CTI sample  phase={payload.get('phase')}  "
                f"CTI={cti_s}  energy={payload.get('raw_energy')}  "
                f"latency_ms={payload.get('latency_ms')}"
            )
        except Exception as e:
            print(f"[demo] bad energy payload: {e}")

    listener.on_message = on_message
    listener.connect(args.broker, args.port, 60)
    listener.subscribe(f"eeg/energy/{args.participant}")
    listener.loop_start()

    print("[demo] Starting synthetic EEG publisher…")
    # Shorter windows than a full session so the demo finishes in ~1–2 minutes.
    pub = spawn(
        os.path.join("pipeline", "synthetic_eeg.py"),
        [
            "--participant",
            args.participant,
            "--broker",
            args.broker,
            "--port",
            str(args.port),
            "--rate",
            str(args.rate),
            "--cal-samples",
            "55",
            "--easy-samples",
            "70",
            "--hard-samples",
            "70",
        ],
    )

    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline and not stop["flag"]:
            if pub.poll() is not None:
                # Publisher finished; allow in-flight SBP windows to publish.
                time.sleep(5.0)
                break
            time.sleep(0.5)
    finally:
        _terminate_all()
        for proc in procs:
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
        listener.loop_stop()
        try:
            listener.disconnect()
        except Exception:
            pass

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[demo] Collected {len(samples)} CTI/energy messages")
    if samples:
        with_cti = [s for s in samples if s.get("CTI") is not None]
        print(f"[demo] Of which {len(with_cti)} have numeric CTI")
        if with_cti:
            vals = [float(s["CTI"]) for s in with_cti]
            print(
                f"[demo] CTI range: {min(vals):.3f} … {max(vals):.3f}  "
                f"mean={sum(vals)/len(vals):.3f}"
            )
        print("[demo] Session log: data/session_data.csv")
        print("[demo] SUCCESS — live MQTT pipeline demonstrated.")
    else:
        print(
            "[demo] No energy messages received. Check broker connectivity and "
            "that pipeline/sbp_subscriber.py can reach the same broker."
        )
        sys.exit(1)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
