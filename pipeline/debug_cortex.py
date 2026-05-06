"""
Minimal Cortex debug: same auth as cortex_client.py, then print raw ws messages for 15s.
"""
import argparse
import json
import ssl
import time

import websocket

PARTICIPANT = "debug_user"
CORTEX_WS = "wss://localhost:6868"
CAPTURE_SEC = 15


def send_recv(ws, msg_dict):
    ws.send(json.dumps(msg_dict))
    return json.loads(ws.recv())


def main():
    p = argparse.ArgumentParser(description="Print raw Cortex WebSocket messages (pow) for 15s")
    p.add_argument("--client-id", required=True)
    p.add_argument("--client-secret", required=True)
    args = p.parse_args()

    print(f"[debug_cortex] participant={PARTICIPANT} (hardcoded)", flush=True)

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
        print("requestAccess:", resp.get("result", resp.get("error")), flush=True)
        if resp.get("error"):
            print("[cortex_client] requestAccess failed:", resp["error"], flush=True)
            return

        res = resp.get("result")
        if isinstance(res, dict) and res.get("accessGranted") is False:
            print(
                ">>> Access not granted yet — click Allow in Emotiv Launcher for this app, "
                "then press Enter here.",
                flush=True,
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
            print("requestAccess (retry):", resp.get("result", resp.get("error")), flush=True)
            if resp.get("error"):
                print("[cortex_client] requestAccess retry failed:", resp["error"], flush=True)
                return
            res = resp.get("result")
            if isinstance(res, dict) and res.get("accessGranted") is False:
                print(
                    "[cortex_client] Access still denied (accessGranted: False). "
                    "Approve the app in Emotiv Launcher and try again.",
                    flush=True,
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
            print(f"Authorize failed. Full response: {resp}", flush=True)
            return
        if resp.get("error"):
            print("authorize error:", resp["error"], flush=True)
            return
        cortex_token = resp["result"]["cortexToken"]
        print(f"Authorized. Token: {cortex_token[:20]}...", flush=True)

        resp = send_recv(
            ws,
            {"id": 3, "jsonrpc": "2.0", "method": "queryHeadsets", "params": {}},
        )
        if "error" in resp and resp["error"]:
            print("queryHeadsets error:", resp["error"], flush=True)
            return
        headsets = resp.get("result") or []
        if not headsets:
            print(
                "No headset found. Make sure EPOC X is connected in Emotiv Launcher",
                flush=True,
            )
            return
        headset_id = headsets[0]["id"]
        print(f"Headset found: {headset_id}", flush=True)

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
            print("createSession error:", resp["error"], flush=True)
            return
        session_id = resp["result"]["id"]
        print(f"Session created: {session_id}", flush=True)

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
            print("subscribe error:", resp["error"], flush=True)
            return
        print("Subscribed to pow stream. Data flowing...", flush=True)

        deadline = time.time() + CAPTURE_SEC
        print(f"[debug_cortex] Raw messages for {CAPTURE_SEC}s:", flush=True)
        while time.time() < deadline:
            raw = ws.recv()
            print(raw, flush=True)

    except KeyboardInterrupt:
        print("\n[debug_cortex] Interrupted.", flush=True)
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
