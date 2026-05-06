"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import GameScreen from "@/components/GameScreen";
import ResultScreen from "@/components/ResultScreen";
import LegendFigure from "@/components/LegendFigure";
import RasterSprite from "@/components/RasterSprite";
import {
  CANONICAL_RUN_PREFIX,
  ORDERED_SCENARIOS,
  RUN_LENGTH,
  decisionSecondsForRound,
} from "@/data/runOrder";
import { variantForAge } from "@/lib/ageIcons";
import { roleLegendCaption } from "@/lib/roles";
import { getHumanSpriteSrc, getPetSpriteSrc } from "@/lib/referenceSprites";
import { publishEvent } from "@/lib/mqttClient";

const RESULT_VISIBLE_MS = 2400;
const BETWEEN_SCENARIOS_MS = 1000;
const LEGEND_PEOPLE = [
  { age: 10, attire: "student", label: "Child" },
  { age: 32, attire: "professional", label: "Adult" },
  { age: 72, attire: "medical", label: "Elder" },
];
const LEGEND_ROLES = [
  "professional",
  "medical",
  "worker",
  "student",
  "athlete",
  "casual",
];
function buildExplanation(scenario, choice) {
  const leftN = scenario.left.count;
  const rightN = scenario.right.count;
  const swerveLeft = choice === "left";
  if (swerveLeft) {
    return `You saved ${rightN} on the right but sacrificed ${leftN} on the left.`;
  }
  return `You saved ${leftN} on the left but sacrificed ${rightN} on the right.`;
}

function choiceLabel(choice) {
  if (choice === "left") return "You swerved LEFT";
  if (choice === "right") return "You stayed RIGHT";
  return "Time up — defaulted to STAY RIGHT";
}

function getPhaseInfo(index) {
  if (index <= 4) return { phase: "calibration", difficulty: "easy" };
  if (index <= 9) return { phase: "easy_test", difficulty: "easy" };
  return { phase: "hard_test", difficulty: "hard" };
}

export default function Home() {
  const [phase, setPhase] = useState("consent");
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [decisions, setDecisions] = useState([]);
  const [runId, setRunId] = useState(null);
  const [runStartedAt, setRunStartedAt] = useState(null);
  const [lastOutcome, setLastOutcome] = useState(null);
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState(null);
  const [consentChecked, setConsentChecked] = useState(false);
  const [diskLogStatus, setDiskLogStatus] = useState("idle"); // idle|saving|saved|error
  const [diskLogError, setDiskLogError] = useState(null);

  const scenarioStartRef = useRef(0);
  const scenarioIndexRef = useRef(0);
  const decidedRef = useRef(false);
  const diskLoggedRef = useRef(false);
  scenarioIndexRef.current = scenarioIndex;

  const scenario = ORDERED_SCENARIOS[scenarioIndex];
  const timerDurationSec = decisionSecondsForRound(scenarioIndex);

  const recordDecision = useCallback((choice, reactionMs) => {
    if (decidedRef.current) return;
    decidedRef.current = true;

    const idx = scenarioIndexRef.current;
    const sc = ORDERED_SCENARIOS[idx];
    const entry = {
      scenarioId: sc.id,
      choice,
      reactionMs,
    };
    setDecisions((d) => [...d, entry]);
    setLastOutcome({
      scenario: sc,
      choice,
      reactionMs,
      explanation: buildExplanation(sc, choice),
      label: choiceLabel(choice),
    });
    setPlaying(false);
    setPhase("result");
    const { phase: expPhase, difficulty } = getPhaseInfo(idx);
    publishEvent("player_1", {
      event: "choice_made",
      scenario_id: sc.id,
      scenario_index: idx,
      experiment_phase: expPhase,
      difficulty: difficulty,
      choice: choice,
      reaction_time_ms: reactionMs,
    });
    setStats(null);
    setStatsError(null);
    setStatsLoading(true);
  }, []);

  const startRun = () => {
    const now = Date.now();
    setRunId(`${CANONICAL_RUN_PREFIX}_${now}`);
    setRunStartedAt(new Date(now).toISOString());
    diskLoggedRef.current = false;
    setDiskLogStatus("idle");
    setDiskLogError(null);
    decidedRef.current = false;
    setDecisions([]);
    setScenarioIndex(0);
    setLastOutcome(null);
    setPhase("playing");
    publishEvent("player_1", {
      event: "session_start",
      experiment_phase: "calibration",
      difficulty: "easy",
    });
  };

  useEffect(() => {
    if (phase !== "playing") return;
    decidedRef.current = false;
    scenarioStartRef.current = Date.now();
    setPlaying(true);
    if (scenario) {
      const { phase: expPhase, difficulty } = getPhaseInfo(scenarioIndex);
      publishEvent("player_1", {
        event: "scenario_start",
        scenario_id: scenario.id,
        scenario_index: scenarioIndex,
        experiment_phase: expPhase,
        difficulty: difficulty,
        timer_duration_sec: timerDurationSec,
      });
    }
  }, [phase, scenarioIndex]);

  const handleChooseLeft = useCallback(() => {
    if (!playing) return;
    const rt = Date.now() - scenarioStartRef.current;
    recordDecision("left", rt);
  }, [playing, recordDecision]);

  const handleChooseRight = useCallback(() => {
    if (!playing) return;
    const rt = Date.now() - scenarioStartRef.current;
    recordDecision("right", rt);
  }, [playing, recordDecision]);

  const handleTimerExpire = useCallback(() => {
    if (!playing) return;
    const rt = Date.now() - scenarioStartRef.current;
    recordDecision("timeout", rt);
  }, [playing, recordDecision]);

  useEffect(() => {
    if (phase !== "result") return;

    let cancelled = false;
    fetch("/api/stats")
      .then((r) => {
        if (!r.ok) throw new Error("Stats failed");
        return r.json();
      })
      .then((data) => {
        if (!cancelled) {
          setStats(data);
          setStatsLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStatsError("Could not load stats.");
          setStatsLoading(false);
        }
      });

    const t1 = setTimeout(() => {
      if (!cancelled) setPhase("gap");
    }, RESULT_VISIBLE_MS);

    return () => {
      cancelled = true;
      clearTimeout(t1);
    };
  }, [phase, lastOutcome?.scenario?.id]);

  useEffect(() => {
    if (phase !== "gap") return;

    const idx = scenarioIndexRef.current;
    const t = setTimeout(() => {
      if (idx >= RUN_LENGTH - 1) {
        publishEvent("player_1", {
          event: "session_complete",
          total_scenarios: RUN_LENGTH,
        });
        setPhase("summary");
      } else {
        setScenarioIndex(idx + 1);
        setPhase("playing");
      }
    }, BETWEEN_SCENARIOS_MS);

    return () => clearTimeout(t);
  }, [phase]);

  useEffect(() => {
    if (phase !== "playing" || !playing) return;

    const onKey = (e) => {
      if (e.repeat) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        handleChooseLeft();
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        handleChooseRight();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, playing, handleChooseLeft, handleChooseRight]);

  const replay = () => {
    setPhase("consent");
    setScenarioIndex(0);
    setLastOutcome(null);
    setDecisions([]);
    setPlaying(false);
    setConsentChecked(false);
    setRunId(null);
    setRunStartedAt(null);
    diskLoggedRef.current = false;
    setDiskLogStatus("idle");
    setDiskLogError(null);
  };

  const downloadAllRunsCSV = async () => {
    const res = await fetch("/api/decisions-log");
    if (!res.ok) throw new Error("No CSV data yet.");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "decisions_all_runs.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  useEffect(() => {
    if (phase !== "summary") return;
    if (!runId) return;
    if (decisions.length === 0) return;
    if (diskLoggedRef.current) return;

    diskLoggedRef.current = true;
    setDiskLogStatus("saving");
    setDiskLogError(null);

    fetch("/api/decisions-log", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        runId,
        startedAt: runStartedAt,
        decisions,
      }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error("Failed to write CSV.");
        return r.json();
      })
      .then(() => setDiskLogStatus("saved"))
      .catch((e) => {
        setDiskLogStatus("error");
        setDiskLogError(e?.message ?? "Failed to write CSV.");
      });
  }, [phase, runId, runStartedAt, decisions.length]);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800/80 py-4 text-center">
        <h1 className="text-xl font-black tracking-tight text-zinc-50 sm:text-2xl">
          Driver Moral Simulator
        </h1>
        <p className="mt-1 text-xs text-zinc-500">
          First-person ethics at speed — no wrong answers, only tradeoffs.
        </p>
      </header>

      <main className="flex flex-1 flex-col items-center justify-center px-4 py-8">
        {phase === "consent" && (
          <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 rounded-2xl border border-zinc-700 bg-zinc-900/90 p-6 shadow-2xl ring-1 ring-white/5 sm:p-8">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-zinc-500">
              Judge
            </p>
            <h2 className="text-2xl font-bold text-zinc-50">Rules and consent</h2>
            <div className="space-y-4 rounded-xl border border-zinc-800 bg-black/35 p-4 text-left text-sm leading-relaxed text-zinc-300">
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                  The scenario
                </p>
                <p className="mt-1.5">
                  You are deciding for a vehicle that cannot avoid everyone ahead: something is in the road, and
                  you must choose which side of the crosswalk bears the outcome.{" "}
                  <span className="font-semibold text-zinc-200">
                    Continue in your lane and affect one group, or swerve and affect the other
                  </span>
                  — there is no third path that saves everyone.
                </p>
              </div>
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                  What’s in the scene
                </p>
                <ul className="mt-1.5 list-inside list-disc space-y-1.5 text-zinc-300">
                  <li>
                    <span className="font-semibold text-zinc-200">Two lanes</span> with a crosswalk in view from
                    the driver’s perspective.
                  </li>
                  <li>
                    <span className="font-semibold text-zinc-200">Pedestrians</span> (and sometimes{" "}
                    <span className="font-semibold text-zinc-200">pets</span>) on each side.
                  </li>
                  <li>
                    <span className="font-semibold text-zinc-200">Crossing status</span> is shown for each zone:
                    legal crossing vs jaywalking, similar to who has the walk at a signal.
                  </li>
                </ul>
              </div>
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                  Your role
                </p>
                <p className="mt-1.5">
                  You act as the judge from <span className="font-semibold text-zinc-200">inside the car</span>.
                  Each scenario presents{" "}
                  <span className="font-semibold text-zinc-200">two possible outcomes</span> — impact on the left
                  group if you swerve, or on the right group if you stay. Pick one before time runs out using the
                  highlighted zones or buttons (or arrow keys).
                </p>
              </div>
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                  After the run
                </p>
                <p className="mt-1.5">
                  You&apos;ll complete <span className="font-semibold text-amber-400/95">20</span> scenarios with a
                  short outcome after each, then a <span className="font-semibold text-zinc-200">summary</span> of this
                  session. The first scenarios are clearer tradeoffs; later ones are closer calls. Progress shows round
                  number / 20 (each case keeps its research id as &quot;case #&quot;).
                </p>
              </div>
              <p className="border-t border-zinc-800 pt-3 text-zinc-400">
                Time per decision starts at <span className="font-semibold text-amber-400">15 seconds</span> and
                increases by <span className="font-semibold text-amber-400">3 seconds</span> every five rounds (up
                to 24s on the last block). Your choices and reaction times are recorded for this run.
              </p>
            </div>

            <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-zinc-700/80 bg-zinc-900/60 px-4 py-3 text-left">
              <input
                type="checkbox"
                checked={consentChecked}
                onChange={(e) => setConsentChecked(e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-amber-500"
              />
              <span className="text-sm leading-snug text-zinc-300">
                I understand the rules and consent to my decision data being recorded for this session.
              </span>
            </label>

            <div className="flex justify-end">
              <button
                type="button"
                disabled={!consentChecked}
                onClick={() => setPhase("legend")}
                className="rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 px-8 py-3 text-base font-bold text-black shadow-lg shadow-orange-900/30 transition hover:brightness-110 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
              >
                Continue
              </button>
            </div>
          </div>
        )}

        {phase === "legend" && (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 rounded-2xl border border-zinc-700 bg-zinc-900/90 p-6 shadow-2xl ring-1 ring-white/5 sm:p-8">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-zinc-500">
              Quick legend
            </p>
            <h2 className="text-2xl font-bold text-zinc-50">What the crossing figures mean</h2>

            <p className="text-[11px] leading-relaxed text-zinc-400">
              Crossing figures are transparent PNG sprites (flat vector style). They are shown larger here than in the
              windshield so you can see detail — in play they scale to the road scene.
            </p>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-zinc-800 bg-black/35 p-4">
                <p className="mb-2 text-xs font-bold uppercase tracking-wider text-zinc-500">
                  Age (same as gameplay)
                </p>
                <p className="mb-2 text-[10px] leading-snug text-zinc-500">
                  Child, adult, and elder each have their own generated sprite (HUD still shows the text label).
                </p>
                <div className="flex flex-wrap items-end justify-center gap-3 sm:gap-4">
                  {LEGEND_PEOPLE.map((p) => (
                    <div key={`${p.age}-${p.attire}`} className="flex flex-col items-center gap-2">
                      <LegendFigure
                        src={getHumanSpriteSrc({
                          role: p.attire,
                          ageVariant: variantForAge(p.age),
                        })}
                      />
                      <p className="text-[11px] font-semibold text-zinc-300">
                        {p.label}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-zinc-800 bg-black/35 p-4">
                <p className="mb-2 text-xs font-bold uppercase tracking-wider text-zinc-500">
                  Crossing status
                </p>
                <div className="space-y-2">
                  <div className="rounded-md border-2 border-solid border-emerald-600/60 bg-emerald-500/10 p-2">
                    <span className="rounded bg-emerald-800 px-2 py-0.5 text-[11px] font-black uppercase text-white">
                      ✓ Legal crossing
                    </span>
                  </div>
                  <div className="rounded-md border-2 border-dashed border-orange-500 bg-orange-400/12 p-2">
                    <span className="rounded bg-orange-600 px-2 py-0.5 text-[11px] font-black uppercase text-white">
                      ⚠ Jaywalking
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-black/35 p-4">
              <p className="mb-3 text-xs font-bold uppercase tracking-wider text-zinc-500">
                Role labels (same HUD text)
              </p>
              <div className="grid gap-3 sm:grid-cols-3">
                {LEGEND_ROLES.map((role) => (
                  <div
                    key={role}
                    className="flex flex-col items-center gap-2 rounded-lg border border-zinc-700/70 bg-zinc-900/70 px-2 py-3 text-center"
                  >
                    <LegendFigure src={getHumanSpriteSrc({ role, ageVariant: "adult" })} />
                    <p className="text-center text-[11px] font-semibold leading-snug text-zinc-200">
                      {roleLegendCaption(role)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-black/35 p-4">
              <p className="mb-3 text-xs font-bold uppercase tracking-wider text-zinc-500">
                Pets (same art in the windshield)
              </p>
              <p className="mb-3 text-[10px] leading-snug text-zinc-500">
                Dog and cat use the same generated flat-vector style as the people.
              </p>
              <div className="flex flex-wrap items-end justify-center gap-8 sm:gap-12">
                <div className="flex flex-col items-center gap-2">
                  <LegendFigure
                    wide
                    src={getPetSpriteSrc("dog")}
                    imgClassName="drop-shadow-[0_4px_8px_rgba(0,0,0,0.4)]"
                  />
                  <p className="text-[11px] font-semibold text-zinc-200">Dog</p>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <LegendFigure
                    wide
                    src={getPetSpriteSrc("cat")}
                    imgClassName="drop-shadow-[0_4px_8px_rgba(0,0,0,0.4)]"
                  />
                  <p className="text-[11px] font-semibold text-zinc-200">Cat</p>
                </div>
              </div>
            </div>

            <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
              <button
                type="button"
                onClick={() => setPhase("consent")}
                className="rounded-xl border border-zinc-600 px-6 py-3 font-semibold text-zinc-200 transition hover:border-zinc-400 hover:bg-zinc-800"
              >
                Back
              </button>
              <button
                type="button"
                onClick={startRun}
                className="rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 px-8 py-3 text-base font-bold text-black shadow-lg shadow-orange-900/30 transition hover:brightness-110 active:scale-[0.98]"
              >
                Start run
              </button>
            </div>
          </div>
        )}

        {phase === "playing" && scenario && (
          <GameScreen
            scenario={scenario}
            scenarioKey={`${scenarioIndex}-${scenario.id}-${timerDurationSec}`}
            roundIndex={scenarioIndex + 1}
            roundTotal={RUN_LENGTH}
            timerDurationSec={timerDurationSec}
            playing={playing}
            onChooseLeft={handleChooseLeft}
            onChooseRight={handleChooseRight}
            onTimerExpire={handleTimerExpire}
          />
        )}

        {phase === "gap" && (
          <div className="flex flex-col items-center gap-3 animate-pulse">
            <div className="h-1 w-32 rounded-full bg-zinc-700" />
            <p className="text-sm font-medium uppercase tracking-widest text-zinc-500">
              Next scenario
            </p>
          </div>
        )}

        {phase === "result" && lastOutcome && (
          <ResultScreen
            choiceLabel={lastOutcome.label}
            explanation={lastOutcome.explanation}
            reactionMs={lastOutcome.reactionMs}
            stats={stats}
            statsLoading={statsLoading}
            statsError={statsError}
          />
        )}

        {phase === "summary" && (
          <div className="mx-auto flex w-full max-w-lg flex-col items-center gap-6 rounded-2xl border border-zinc-700 bg-zinc-900/90 p-8 text-center">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-zinc-500">
              Run complete
            </p>
            <h2 className="text-2xl font-bold">20 scenarios cleared</h2>
            <p className="text-sm text-zinc-400">
              Logged {decisions.length} decisions for this run.
            </p>
            <p className="text-xs text-zinc-500">
              Saved to `data/exports/decisions_all_runs.csv` on the server.
            </p>
            {diskLogStatus === "saving" && (
              <p className="text-xs text-zinc-400">Saving CSV…</p>
            )}
            {diskLogStatus === "error" && (
              <p className="text-xs text-red-400">CSV save failed: {diskLogError}</p>
            )}
            <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
              <button
                type="button"
                disabled={decisions.length === 0}
                onClick={downloadAllRunsCSV}
                className="rounded-xl border border-zinc-600 px-6 py-3 font-semibold text-zinc-200 transition hover:border-zinc-400 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Download all runs CSV
              </button>
              <button
                type="button"
                onClick={replay}
                className="rounded-xl border border-zinc-600 px-6 py-3 font-semibold text-zinc-200 transition hover:border-zinc-400 hover:bg-zinc-800"
              >
                Play again
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
