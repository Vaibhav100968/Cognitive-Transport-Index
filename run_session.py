"""
Orchestrate SBP subscriber, data logger, live dashboard, and mock or live EEG.
"""
import os
import subprocess
import sys
import time

import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _make_mqtt_client():
    import paho.mqtt.client as mqtt

    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def preflight(broker):
    failures = []

    c = _make_mqtt_client()
    try:
        c.connect(broker, 1883, 5)
        c.loop_start()
        time.sleep(0.25)
        c.loop_stop()
        c.disconnect()
    except Exception as e:
        failures.append(f"MQTT broker unreachable at {broker}:1883 ({e})")
        failures.append("Run: brew services start mosquitto")

    checks = [
        (os.path.join(_ROOT, "core", "streaming_sbp.py"), "core/streaming_sbp.py"),
        (os.path.join(_ROOT, "pipeline", "sbp_subscriber.py"), "pipeline/sbp_subscriber.py"),
        (os.path.join(_ROOT, "pipeline", "data_logger.py"), "pipeline/data_logger.py"),
    ]
    for path, label in checks:
        if not os.path.isfile(path):
            failures.append(f"Missing {label}")

    data_dir = os.path.join(_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    if failures:
        print("Preflight failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("✓ All preflight checks passed")


def main():
    p = argparse.ArgumentParser(description="Run full EEG→SBP session stack")
    p.add_argument(
        "--participants",
        default="player_1",
        help="Comma-separated participant ids (e.g. player_1,player_2)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Start mock CSV EEG publishers instead of Cortex",
    )
    p.add_argument("--broker", default="localhost")
    args = p.parse_args()

    os.chdir(_ROOT)
    preflight(args.broker)

    procs = []
    py = sys.executable

    def spawn(script_rel, extra_args=None):
        cmd = [py, script_rel]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.Popen(
            cmd,
            cwd=_ROOT,
            stdin=subprocess.DEVNULL,
        )

    start_time = time.time()
    try:
        procs.append(spawn(os.path.join("pipeline", "sbp_subscriber.py")))
        print("✓ SBP Subscriber started")
        time.sleep(3)

        procs.append(spawn(os.path.join("pipeline", "data_logger.py")))
        print("✓ Data Logger started")
        time.sleep(2)

        procs.append(spawn(os.path.join("pipeline", "live_dashboard.py")))
        print("✓ Live Dashboard started")
        time.sleep(2)

        plist = [x.strip() for x in args.participants.split(",") if x.strip()]
        for part in plist:
            if args.mock:
                procs.append(
                    spawn(
                        os.path.join("pipeline", "mock_eeg.py"),
                        [
                            "--participant",
                            part,
                            "--broker",
                            args.broker,
                        ],
                    )
                )
                print(f"✓ Mock EEG started for {part}")
            else:
                print(f"⚠️  Start cortex_client manually for {part}:")
                print(
                    f"   python3 pipeline/cortex_client.py --client-id ID "
                    f"--client-secret SECRET --participant {part}"
                )

        participants_str = ", ".join(plist)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("SESSION RUNNING")
        print(f"Participants: {participants_str}")
        print(f"Mode: {'MOCK (CSV replay)' if args.mock else 'LIVE (Emotiv)'}")
        print("Data: data/session_data.csv")
        print("Press Ctrl+C to end session")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        csv_path = os.path.join(_ROOT, "data", "session_data.csv")

        while True:
            time.sleep(30)
            elapsed = (time.time() - start_time) / 60.0
            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    n = max(0, sum(1 for _ in f) - 1)
            except OSError:
                n = 0
            running = sum(1 for p in procs if p.poll() is None)
            print(
                f"[{elapsed:.1f} min] Rows logged: {n} | "
                f"Processes: {running} running"
            )

    except KeyboardInterrupt:
        duration_min = (time.time() - start_time) / 60.0
        print("Ending session...")
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("SESSION COMPLETE")
        print(f"Duration: {duration_min:.1f} minutes")
        print("Run analysis: python3 analysis/experiments.py")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
