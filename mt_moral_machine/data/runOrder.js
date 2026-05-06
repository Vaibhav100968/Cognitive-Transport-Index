import { SCENARIOS } from "./scenarios";

/**
 * Every session uses this prefix in `runId` (plus a timestamp) so logs identify this
 * fixed design. Scenario order and tiering never shuffle.
 */
export const CANONICAL_RUN_PREFIX = "dms-canonical-v1";

/**
 * Play order: four blocks of 5. Each block is strictly easier → harder.
 *
 * Rubric (roughly): count skew and “protect the vulnerable / follow the signal”
 * alignment make choices easier; near-equal counts, both sides sympathetic, or
 * pure rule-vs-number conflicts without a clear hero side make them harder.
 *
 * Block 1 — very lopsided utilitarian / protect-kids clarity
 * Block 2 — still skewed counts, introducing pets, jaywalking, or 2v3 tension
 * Block 3 — rule vs numbers, sacred single figures (child, doctor), 2v3–3v2
 * Block 4 — tight counts (2v2, 3v3, 4v4) and mirrored life-stage tradeoffs
 */
export const SCENARIO_IDS_BY_ROUND = [
  // 1: 1v5 kids … 2: 3 toddlers v 1 … 3: 1v3 adults … 4: 2 kids+dog v 1 … 5: 4v1 + legal frame
  4, 17, 2, 1, 12,
  // 6: 1 illegal v 4 legal kids … 7: 1v2 kids … 8: 2 toddlers v 3 adults … 9: 6 illegal v 1 legal … 10: 4v2 jaywalk v legal
  14, 11, 9, 8, 3,
  // 11: 3 legal elders v 2 jaywalk … 12: 5 jaywalk v 2 legal … 13: 2v3 elders v teens … 14: 1 child v 4 elders … 15: 1 doctor v 7 jaywalk
  5, 15, 19, 7, 18,
  // 16: 2v2 legal adults … 17: 3v3 legal v illegal … 18: 2v2 same-age band … 19: 2 elderly v 2 young … 20: 4v4 mixed mirror
  6, 10, 16, 13, 20,
];

export const RUN_LENGTH = SCENARIO_IDS_BY_ROUND.length;

const byId = new Map(SCENARIOS.map((s) => [s.id, s]));

export const ORDERED_SCENARIOS = SCENARIO_IDS_BY_ROUND.map((id) => {
  const s = byId.get(id);
  if (!s) throw new Error(`runOrder: missing scenario id ${id}`);
  return s;
});

/** 0-based round index → decision timer in seconds (+3s each block of 5). */
export function decisionSecondsForRound(roundIndexZeroBased) {
  const block = Math.floor(roundIndexZeroBased / 5);
  const BASE_S = 15;
  const EXTRA_PER_BLOCK_S = 3;
  return BASE_S + block * EXTRA_PER_BLOCK_S;
}
