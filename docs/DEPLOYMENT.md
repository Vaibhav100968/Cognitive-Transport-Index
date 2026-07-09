# Live deployment & runnable demo

This project is a **local real-time pipeline** (MQTT + Python), not a hosted SaaS.
There is no public cloud URL for Emotiv sessions (headset + Cortex must run on the
operator machine). Use the options below to replicate the environment and see CTI
update live.

## Option A — One-command Docker demo (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose).

```bash
git clone https://github.com/Vaibhav100968/Cognitive-Transport-Index.git
cd Cognitive-Transport-Index
docker compose up --build
```

What runs:

| Service | Role |
|---|---|
| `mosquitto` | MQTT broker on host port **1883** |
| `cti-demo` | Synthetic EEG publisher + SBP subscriber + logger |

You should see log lines like:

```text
[demo] CTI sample  phase=easy_test  CTI=0.412  energy=...  latency_ms=...
[demo] SUCCESS — live MQTT pipeline demonstrated.
```

Session rows land in `./data/session_data.csv` (bind-mounted).

Stop with `Ctrl+C`, then `docker compose down`.

### Broker-only (use host Python)

```bash
docker compose up mosquitto
# other terminal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_live_demo.py --broker localhost
```

## Option B — Local live demo (no Docker)

```bash
brew install mosquitto && brew services start mosquitto   # macOS
# or: sudo apt install mosquitto && sudo systemctl start mosquitto

pip install -r requirements.txt
python scripts/run_live_demo.py
```

This starts:

1. `pipeline/sbp_subscriber.py` — features → CTI  
2. `pipeline/data_logger.py` — CSV sink  
3. `pipeline/synthetic_eeg.py` — fake rest / easy / hard feature stream  

No Emotiv headset and no VEGS CSV required.

## Option C — Full session stack (dashboard)

```bash
# Terminal 1
python run_session.py --participants demo_player

# Terminal 2 (synthetic publisher)
python pipeline/synthetic_eeg.py --participant demo_player
```

Or with VEGS CSV replay (data available upon request):

```bash
python run_session.py --mock --participants "Player 1"
```

## Option D — Live Emotiv EPOC X

1. Install **Emotiv Launcher** so Cortex listens on `wss://localhost:6868`.
2. Start MQTT (`brew services start mosquitto` or `docker compose up mosquitto`).
3. Start the session stack:

```bash
python run_session.py --participants player_1
```

4. In another terminal, connect the headset:

```bash
python pipeline/cortex_client.py \
  --client-id YOUR_CLIENT_ID \
  --client-secret YOUR_CLIENT_SECRET \
  --participant player_1
```

Approve the app in Emotiv Launcher when prompted. Features publish to
`eeg/features/player_1`; CTI appears on `eeg/energy/player_1` and in the live dashboard.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `MQTT_BROKER` | `localhost` | subscriber, logger, demo, Docker |
| `MQTT_PORT` | `1883` | same |
| `STREAMING_SBP_EPOCHS` | `20` | score-net training budget per window |
| `STREAMING_SBP_STEPS` | `100` | Euler–Maruyama steps per window |

Compose sets a smaller epoch/step budget so the demo finishes quickly. For paper-faithful latency, unset those overrides (or set `20` / `100`).

## Why there is no public demo URL

- Live EEG requires a physical Emotiv headset and the Cortex desktop service.
- Subject data is **not** published (available upon request).
- The “demo” is therefore a **reproducible local / Docker run** that proves the
  MQTT → SBP → CTI path with synthetic features.

Checked-in sample outputs (offline, no broker): [`samples/outputs/synthetic_benchmark.md`](../samples/outputs/synthetic_benchmark.md).

## Architecture pointer

See [ARCHITECTURE.md](ARCHITECTURE.md) for topic layout, CTI math, and process roles.
