"use client";

import { useEffect, useRef, useState } from "react";

const DEFAULT_DURATION_S = 15;

export default function Timer({ active, onExpire, scenarioKey, durationSec = DEFAULT_DURATION_S }) {
  const [remaining, setRemaining] = useState(durationSec);
  const expiredRef = useRef(false);
  const onExpireRef = useRef(onExpire);
  onExpireRef.current = onExpire;

  useEffect(() => {
    expiredRef.current = false;
    setRemaining(durationSec);
  }, [scenarioKey, active, durationSec]);

  useEffect(() => {
    if (!active) return;

    const start = performance.now();

    const id = setInterval(() => {
      const elapsed = (performance.now() - start) / 1000;
      const next = Math.max(0, durationSec - elapsed);
      setRemaining(next);
      if (next <= 0 && !expiredRef.current) {
        expiredRef.current = true;
        clearInterval(id);
        onExpireRef.current();
      }
    }, 32);

    return () => clearInterval(id);
  }, [active, scenarioKey, durationSec]);

  const pct = durationSec > 0 ? (remaining / durationSec) * 100 : 0;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative h-3 w-48 overflow-hidden rounded-full bg-zinc-800 ring-1 ring-zinc-600">
        <div
          className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-500 transition-[width] duration-75 ease-linear"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-sm tabular-nums text-amber-400">
        {remaining > 0 ? remaining.toFixed(1) : "0.0"}s
      </span>
    </div>
  );
}
