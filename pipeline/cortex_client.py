"""
Emotiv Cortex → MQTT bridge: band powers (pow stream) + derived indices as normalized features.
Requires Emotiv Launcher (Cortex service on wss://localhost:6868).
"""
import argparse
import json
import os
import ssl
import time

import numpy as np
import paho.mqtt.client as mqtt
import websocket

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

MQTT_PORT = 1883
CORTEX_WS = "wss://localhost:6868"


def send_recv(ws, msg_dict):
    ws.send(json.dumps(msg_dict))
    return json.loads(ws.recv())


def _make_mqtt_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        return mqtt.Client()


def main():
    p = argparse.ArgumentParser(description="Emotiv Cortex EEG → MQTT feature publisher")
    p.add_argument("--client-id", required=True, help="Emotiv app Client ID")
    p.add_argument("--client-secret", required=True, help="Emotiv app Client Secret")
    p.add_argument("--participant", required=True, help="Participant id, e.g. player_1")
    p.add_argument("--broker", default="localhost", help="MQTT broker host")
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Run 60s calibration and save mu/sigma to data/",
    )
    args = p.parse_args()

    mode = "CALIBRATION (60s)" if args.calibrate else "STREAM"
    print("=" * 60)
    print("Emotiv Cortex → MQTT")
    print(f"  Participant: {args.participant}")
    print(f"  Mode:        {mode}")
    print(f"  MQTT:        {args.broker}:{MQTT_PORT}")
    print(f"  Cortex:      {CORTEX_WS}")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    cal_path = os.path.join(data_dir, f"calibration_{args.participant}.json")

    mqtt_client = _make_mqtt_client()
    try:
        mqtt_client.connect(args.broker, MQTT_PORT, 60)
    except ConnectionRefusedError:
        print(
            f"[cortex_client] Cannot connect to MQTT at {args.broker}:{MQTT_PORT}\n"
            "  Start the broker (e.g. mosquitto) and retry."
        )
        return

    mqtt_client.loop_start()

    ws = None
    try:
        ws = websocket.create_connection(
            CORTEX_WS,
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )

        resp = send_recv(
            ws,
            {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "requestAccess",
                "params": {
                    "clientId": args.client_id,
                    "clientSecret": args.client_secret,
                },
            },
        )
        print("requestAccess:", resp.get("result", resp.get("error")))
        if resp.get("error"):
            print("[cortex_client] requestAccess failed:", resp["error"])
            return

        res = resp.get("result")
        if isinstance(res, dict) and res.get("accessGranted") is False:
            print(
                ">>> Access not granted yet — click Allow in Emotiv Launcher for this app, "
                "then press Enter here."
            )
            input()
            resp = send_recv(
                ws,
                {
                    "id": 11,
                    "jsonrpc": "2.0",
                    "method": "requestAccess",
                    "params": {
                        "clientId": args.client_id,
                        "clientSecret": args.client_secret,
                    },
                },
            )
            print("requestAccess (retry):", resp.get("result", resp.get("error")))
            if resp.get("error"):
                print("[cortex_client] requestAccess retry failed:", resp["error"])
                return
            res = resp.get("result")
            if isinstance(res, dict) and res.get("accessGranted") is False:
                print(
                    "[cortex_client] Access still denied (accessGranted: False). "
                    "Approve the app in Emotiv Launcher and try again."
                )
                return

        resp = send_recv(
            ws,
            {
                "id": 2,
                "jsonrpc": "2.0",
                "method": "authorize",
                "params": {
                    "clientId": args.client_id,
                    "clientSecret": args.client_secret,
                    "debit": 1,
                },
            },
        )
        if "result" not in resp:
            print(f"Authorize failed. Full response: {resp}")
            return
        if resp.get("error"):
            print("authorize error:", resp["error"])
            return
        cortex_token = resp["result"]["cortexToken"]
        print(f"Authorized. Token: {cortex_token[:20]}...")

        resp = send_recv(
            ws,
            {"id": 3, "jsonrpc": "2.0", "method": "queryHeadsets", "params": {}},
        )
        if "error" in resp and resp["error"]:
            print("queryHeadsets error:", resp["error"])
            return
        headsets = resp.get("result") or []
        if not headsets:
            print(
                "No headset found. Make sure EPOC X is connected in Emotiv Launcher"
            )
            return
        headset_id = headsets[0]["id"]
        print(f"Headset found: {headset_id}")

        resp = send_recv(
            ws,
            {
                "id": 4,
                "jsonrpc": "2.0",
                "method": "createSession",
                "params": {
                    "cortexToken": cortex_token,
                    "headset": headset_id,
                    "status": "active",
                },
            },
        )
        if "error" in resp and resp["error"]:
            print("createSession error:", resp["error"])
            return
        session_id = resp["result"]["id"]
        print(f"Session created: {session_id}")

        resp = send_recv(
            ws,
            {
                "id": 5,
                "jsonrpc": "2.0",
                "method": "subscribe",
                "params": {
                    "cortexToken": cortex_token,
                    "session": session_id,
                    "streams": ["pow"],
                },
            },
        )
        if "error" in resp and resp["error"]:
            print("subscribe error:", resp["error"])
            return
        print("Subscribed to pow stream. Data flowing...")

        latest_pow = None
        last_publish_time = 0
        publish_count = 0
        calibration_vectors = []

        cal_mu = None
        cal_sigma = None
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                cal = json.load(f)
            cal_mu = np.array(cal["mu"])
            cal_sigma = np.array(cal["sigma"])
            print(f"Loaded calibration from {cal_path}")
        else:
            print("No calibration file found — using raw features (run --calibrate first)")

        while True:
            try:
                raw = ws.recv()
                data = json.loads(raw)
            except Exception as e:
                print(f"WebSocket error: {e}. Reconnecting...", flush=True)
                break

            if "pow" in data:
                pow_data = data["pow"]
                try:
                    theta = float(np.mean([row[0] for row in pow_data]))
                    alpha = float(np.mean([row[1] for row in pow_data]))
                    beta_l = float(np.mean([row[2] for row in pow_data]))
                    beta_h = float(np.mean([row[3] for row in pow_data]))
                    gamma = float(np.mean([row[4] for row in pow_data]))
                    engagement = float(beta_l / (alpha + theta + 1e-8))
                    arousal = float((beta_l + beta_h) / (alpha + theta + 1e-8))
                    valence = float(
                        np.clip(alpha - beta_h, -1.0, 1.0)
                        / (alpha + beta_h + 1e-8)
                    )
                    raw_vec = np.array(
                        [
                            theta,
                            alpha,
                            beta_l,
                            beta_h,
                            gamma,
                            arousal,
                            valence,
                            engagement,
                        ]
                    )
                    latest_pow = raw_vec
                except (IndexError, TypeError):
                    pass

            now = time.time()
            if latest_pow is not None:
                if now - last_publish_time >= 0.1:
                    raw_vec = latest_pow

                    if args.calibrate:
                        calibration_vectors.append(raw_vec.tolist())
                        if len(calibration_vectors) % 10 == 0:
                            rem = max(0.0, 60 - len(calibration_vectors) * 0.1)
                            print(
                                f"Calibration: {len(calibration_vectors)} samples "
                                f"({rem:.0f}s remaining)",
                                flush=True,
                            )
                        if len(calibration_vectors) >= 600:
                            mu = np.mean(calibration_vectors, axis=0)
                            sigma = np.std(calibration_vectors, axis=0) + 1e-8
                            cal_data = {
                                "participant_id": args.participant,
                                "features": FEATURES,
                                "mu": mu.tolist(),
                                "sigma": sigma.tolist(),
                                "n_samples": len(calibration_vectors),
                                "timestamp": now,
                            }
                            os.makedirs(data_dir, exist_ok=True)
                            with open(cal_path, "w") as f:
                                json.dump(cal_data, f, indent=2)
                            print(f"Calibration saved to {cal_path}", flush=True)
                            print(f"mu={mu.round(4)}", flush=True)
                            print(f"sigma={sigma.round(4)}", flush=True)
                            break
                    else:
                        if cal_mu is not None:
                            norm_vec = np.clip(
                                (raw_vec - cal_mu) / (cal_sigma + 1e-8),
                                -5.0,
                                5.0,
                            )
                        else:
                            norm_vec = raw_vec

                        payload = {f: float(norm_vec[i]) for i, f in enumerate(FEATURES)}
                        payload["participant_id"] = args.participant
                        payload["timestamp"] = now

                        topic = f"eeg/features/{args.participant}"
                        mqtt_client.publish(topic, json.dumps(payload))
                        publish_count += 1

                        if publish_count % 10 == 0:
                            print(
                                f"[{args.participant}] Published #{publish_count} | "
                                f"Engagement={payload['Engagement']:.3f} | "
                                f"Arousal={payload['Arousal']:.3f}",
                                flush=True,
                            )

                    last_publish_time = now

    except KeyboardInterrupt:
        print("\n[cortex_client] Interrupted — shutting down.")
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        mqtt_client.loop_stop()
        try:
            mqtt_client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
