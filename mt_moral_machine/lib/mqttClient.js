import mqtt from "mqtt";

let client = null;

function getClient() {
  if (typeof window === "undefined") return null;
  if (client) return client;
  try {
    client = mqtt.connect("ws://localhost:9001", {
      reconnectPeriod: 3000,
      connectTimeout: 5000,
    });
    client.on("connect", () => {
      console.log("[MQTT] Connected to broker ws://localhost:9001");
    });
    client.on("error", (err) => {
      console.warn("[MQTT] Connection error (game continues):", err.message);
    });
    client.on("offline", () => {
      console.warn("[MQTT] Broker offline — events will be dropped until reconnect");
    });
  } catch (e) {
    console.warn("[MQTT] Failed to connect:", e.message);
    client = null;
  }
  return client;
}

export function publishEvent(participantId, payload) {
  const c = getClient();
  if (!c) return;
  const topic = `game/events/${participantId}`;
  const message = JSON.stringify({
    ...payload,
    participant_id: participantId,
    timestamp: Date.now() / 1000,
  });
  try {
    c.publish(topic, message);
  } catch (e) {
    console.warn("[MQTT] Publish failed:", e.message);
  }
}
