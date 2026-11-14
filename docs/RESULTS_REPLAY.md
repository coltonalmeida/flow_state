# Results page: side-by-side replays on the downtown network

The first section of the Results tab replays recorded SUMO runs on the `district` network (the same map and renderer as the Simulation page): the left
map is main's **fixed-time plans**, the right map is the **same plans with a FlowState controller**. Both sides use the same seed, demand and events; only
the signal control differs. Nothing is simulated in the browser; the page plays back recorded positions and signal states.

## What the right side is

The Results tab shows two versions only: the signals **before the agent** (fixed-time) and **after the agent** (the FlowState ML agent).

The agent is the PPO policy in `artifacts/ml/rl/`: one shared network picks hold / end / extend for every signal every 5 s, restricted to safe timings,
with ambulance green priority beside it. It acts only on green *durations*: amber, all-red, pedestrian clearance and protected-left phases always run
their programmed length.

The page asks the backend for the agent by name (`POST /api/replay` with `controller: "rl_ppo"`). If the model file is missing or does not match its
manifest, the backend falls back to the hand-written queue rules and the page says so above the right-hand map.

Not shown on the page, but still in the repository: the hand-written rules (`backend/flowstate/replay/`), the forecast-assisted controller, and their
held-out results in `artifacts/ml/evaluation.json`. The agent is labelled verified because it passed the publication gate against those rules there;
`frontend/scripts/extract_controller_results.py` refuses to build the page's data file if the agent is not verified.

## How the numbers are made

`python -m flowstate.replay.record --scenarios ambulance,rush_hour,accident,construction --seeds 3100-3107 --controller rl_ppo --out frontend/public/flowstate-demo --eval-out artifacts/replay/district-eval.json`
(run from `backend/` with the paths adjusted). `index.json` records which controller the runs used.

For every scenario and seed it runs both arms headless (5 minutes each, main's engine, libsumo), records half-second frames for the display seed, and
computes paired statistics over all seeds: mean per arm, mean paired difference, a 95 % bootstrap interval of the difference, and better/worse run
counts. Everything shown in this section, including the worse-run counts, comes from that run.

"Wait" is the whole time a vehicle stood still, every standstill spell summed, over the cars that finished their trip. (SUMO's own accumulated waiting
time only covers the last 100 s of each vehicle; it is kept in the asset as `waitingWindow`.)

The emergency unit shown first is chosen by a stated rule (queued ≥ 10 s on the fixed-time side and held ≤ 3 s on the right; otherwise the unit that
gained most) and is therefore a best case; every unit in the run is listed with its own numbers.

## Limits

* 8 paired seeds per scenario is a small sample; read the intervals before the percentages. The held-out evaluation further down the page
  (80 seeds × 6 scenarios) is the evidence; this section is an illustration.
* The replay figures are not comparable with the held-out evaluation's: 5-minute runs and finished cars only here, against every departed vehicle there.
* The learned policy was trained on rush hour, construction, accident, weather and transit. The ambulance scenario was not a training scenario.
* Unit trip time is the call-to-scene leg only.
* Demand is a seeded model, not calibrated to Toronto counts.
