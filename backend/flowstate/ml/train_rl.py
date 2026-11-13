"""PPO for the shared per-junction signal policy, on real district episodes, with a behaviour-cloning warm start.

Loop (one process pool; every worker runs whole SUMO episodes, since libsumo hosts one simulation per process):

1. **Warm start** - episodes driven by the queue rule at 5 s decisions (``LearnedPolicy(mode="teacher")``, with a little random
   exploration) give (observation, valid actions, teacher action) rows; the actor is fitted to the teacher by cross-entropy and
   the critic to the observed discounted returns.
2. **PPO** - each iteration runs ``episodes`` stochastic rollouts with the current weights on random training scenarios / seeds,
   then updates on every junction's trajectory (generalised advantage estimation, clipped objective, masked actions).
3. **Selection** - every ``val_every`` iterations the greedy policy is run on fixed validation episodes; the saved policy is the
   checkpoint with the best validation score (waiting, time lost including insertion delay, and trips not served, each relative to
   the queue rule), not the last one. The manifest is rewritten with every selected checkpoint, so an interrupted run leaves a
   policy the trainer itself described.

Seed discipline: rollouts draw from the training seeds (1000-1199), selection uses validation seeds (2000-2003), and the test
seeds (3000-3079) are touched only by ``ml.cli evaluate``. The held-out scenario (``stadium``) is never trained or selected on.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import rl_env as R
from .dataset import SEEDS, TRAIN_SCENARIOS
from .episodes import model_dir, run_episode
from .features import feature_hash

METRIC_KEYS = ("mean_wait_s", "mean_time_loss_s", "mean_depart_delay_s", "unserved", "throughput_vph", "teleports")


@dataclass
class RLConfig:
    seed: int = 1
    iters: int = 30
    episodes: int = 14  # rollouts per iteration
    workers: int = 14
    bc_episodes: int = 28
    bc_epochs: int = 30
    bc_eps: float = 0.1
    hidden: int = 64
    gamma: float = 0.98  # per 5 s decision
    lam: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 4096
    clip: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    blend: float = 0.5
    reward_scale: float = 0.1
    val_every: int = 5
    val_seeds: tuple[int, ...] = (2000, 2001, 2002, 2003)
    out_dir: str = ""


def _rollout(job: dict[str, Any]) -> dict[str, Any]:
    from ..replay.priority import EmergencyPriority
    from .policies import LearnedPolicy

    if job.get("arm"):
        r = run_episode(job["arm"], job["scenario"], job["seed"])
        return {"metrics": {k: r[k] for k in METRIC_KEYS}, "scenario": job["scenario"], "seed": job["seed"], "arm": job["arm"]}
    pol = LearnedPolicy(job.get("weights"), mode=job["mode"], seed=job["seed"] * 7919 + job.get("iter", 0), eps=job.get("eps", 0.0), record=job.get("record", False))
    r = run_episode("rl_ppo", job["scenario"], job["seed"], policies=[EmergencyPriority(), pol])
    out: dict[str, Any] = {"metrics": {k: r[k] for k in METRIC_KEYS}, "scenario": job["scenario"], "seed": job["seed"], "counts": pol.counts,
                           "network_id": pol.topo.network_id if pol.topo else None}
    if job.get("record"):
        out["traj"] = pol.trajectory()
    return out


def _gae(traj: dict[str, np.ndarray], cfg: RLConfig, use_values: bool = True) -> dict[str, np.ndarray]:
    """Advantages and returns for every junction's trajectory in one episode (the last sample only bootstraps)."""
    r = R.rewards(traj["queue"], cfg.blend, cfg.reward_scale, traj.get("backlog"))  # (T-1, J)
    v = traj["val"] if use_values else np.zeros_like(traj["val"])
    lam = cfg.lam if use_values else 1.0
    adv = np.zeros_like(r)
    last = np.zeros(r.shape[1], dtype=np.float32)
    for k in range(len(r) - 1, -1, -1):
        delta = r[k] + cfg.gamma * v[k + 1] - v[k]
        last = delta + cfg.gamma * lam * last
        adv[k] = last
    flat = lambda a: a[:-1].reshape(-1, *a.shape[2:])  # noqa: E731
    return {"obs": flat(traj["obs"]), "mask": flat(traj["mask"]), "act": flat(traj["act"]), "logp": flat(traj["logp"]), "teacher": flat(traj["teacher"]),
            "adv": adv.reshape(-1), "ret": (adv + v[:-1]).reshape(-1)}


def _cat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {k: float(np.mean([r["metrics"][k] for r in rows])) for k in METRIC_KEYS}


def train(cfg: RLConfig) -> dict[str, Any]:
    import torch
    from torch import nn

    out = Path(cfg.out_dir) if cfg.out_dir else model_dir() / "rl"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(cfg.seed)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mk = lambda n_out: nn.Sequential(nn.Linear(R.OBS_DIM, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, n_out))  # noqa: E731
            self.pi, self.v = mk(R.N_ACTIONS), mk(1)

        def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
            return torch.distributions.Categorical(logits=self.pi(obs).masked_fill(~mask, -1e9))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    weights = lambda: {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}  # noqa: E731
    tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)  # noqa: E731
    lo, hi = SEEDS["train"]
    draw = lambda n: [(TRAIN_SCENARIOS[int(rng.integers(len(TRAIN_SCENARIOS)))], int(rng.integers(lo, hi + 1))) for _ in range(n)]  # noqa: E731
    val_jobs = [(sc, s) for sc in TRAIN_SCENARIOS for s in cfg.val_seeds]
    log_path = out / "training_log.jsonl"
    log_path.write_text("")
    t0 = time.time()

    def log(rec: dict[str, Any]) -> None:
        rec["wall_s"] = round(time.time() - t0)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    with mp.get_context("spawn").Pool(cfg.workers, maxtasksperchild=6) as pool:  # recycle workers: long-lived SUMO processes grow
        # validation baselines, once (the same episodes every checkpoint is scored on)
        base_rows = pool.map(_rollout, [{"arm": a, "scenario": sc, "seed": s} for a in ("fixed_time", "queue_responsive", "safe_heuristic") for sc, s in val_jobs], chunksize=1)
        baselines = {a: _mean_metrics([r for r in base_rows if r["arm"] == a]) for a in ("fixed_time", "queue_responsive", "safe_heuristic")}
        ref = baselines["queue_responsive"]
        log({"stage": "baselines", **{a: {k: round(v, 3) for k, v in m.items()} for a, m in baselines.items()}})

        def validate(tag: str, **job: Any) -> dict[str, float]:
            rows = pool.map(_rollout, [{"scenario": sc, "seed": s, **job} for sc, s in val_jobs], chunksize=1)
            m = _mean_metrics(rows)
            delay = lambda x: x["mean_time_loss_s"] + x["mean_depart_delay_s"]  # noqa: E731 - time lost on the road and waiting to get onto it
            # 2.5 = the shipped heuristic. Unserved trips count: a policy must not score well by keeping traffic out of the averages
            m["score"] = m["mean_wait_s"] / ref["mean_wait_s"] + delay(m) / delay(ref) + 0.5 * m["unserved"] / max(1.0, ref["unserved"])
            log({"stage": "validation", "tag": tag, **{k: round(v, 4) for k, v in m.items()}})
            return m

        teacher_val = validate("teacher", mode="teacher")

        # 1. behaviour cloning
        rows = pool.map(_rollout, [{"scenario": sc, "seed": s, "mode": "teacher", "eps": cfg.bc_eps, "record": True} for sc, s in draw(cfg.bc_episodes)], chunksize=1)
        network_id = rows[0]["network_id"]
        data = _cat([_gae(r["traj"], cfg, use_values=False) for r in rows])
        choice = data["mask"].sum(1) > 1
        obs, mask, lab, ret = tt(data["obs"]), tt(data["mask"], torch.bool), tt(data["teacher"], torch.long), tt(data["ret"])
        idx_c = torch.nonzero(tt(choice, torch.bool))[:, 0]
        counts = np.bincount(data["teacher"][choice], minlength=R.N_ACTIONS).astype(float)
        cls_w = tt(counts.sum() / (R.N_ACTIONS * np.maximum(counts, 1.0)) ** 0.5 / (counts.sum() ** 0.5 / R.N_ACTIONS ** 0.5))  # mild re-weighting of rare actions
        for ep in range(cfg.bc_epochs):
            perm = idx_c[torch.randperm(len(idx_c))]
            tot = 0.0
            for b in range(0, len(perm), 2048):
                i = perm[b:b + 2048]
                logits = net.pi(obs[i]).masked_fill(~mask[i], -1e9)
                loss = nn.functional.cross_entropy(logits, lab[i], weight=cls_w)
                j = torch.randint(0, len(obs), (2048,))
                loss = loss + 0.5 * nn.functional.mse_loss(net.v(obs[j])[:, 0], ret[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
        with torch.no_grad():
            agree = float((net.pi(obs[idx_c]).masked_fill(~mask[idx_c], -1e9).argmax(1) == lab[idx_c]).float().mean())
        log({"stage": "bc", "rows": int(len(obs)), "choice_rows": int(len(idx_c)), "teacher_actions": counts.tolist(), "agreement": round(agree, 4), "loss": round(tot, 3)})

        best = {"score": float("inf"), "iter": -1, "metrics": {}}

        def write_manifest(done_iters: int) -> dict[str, Any]:
            manifest = {"kind": "shared per-junction PPO policy (hold / end / extend, masked)", "sha256": hashlib.sha256((out / "policy.npz").read_bytes()).hexdigest(),
                        "obs_dim": R.OBS_DIM, "obs_version": R.OBS_VERSION, "feature_hash": feature_hash(), "network_id": network_id, "config": asdict(cfg),
                        "selected_iter": best["iter"], "completed_iters": done_iters, "complete": done_iters >= cfg.iters,
                        "validation": {"selected": best["metrics"], "teacher": teacher_val, "baselines": baselines,
                                       "episodes": len(val_jobs), "seeds": list(cfg.val_seeds), "scenarios": list(TRAIN_SCENARIOS)},
                        "seed_pools": {"train": list(SEEDS["train"]), "selection": list(cfg.val_seeds), "test": list(SEEDS["test"])},
                        "train_wall_s": round(time.time() - t0), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "note": "simulation only; verified status is decided by ml.cli evaluate on the test seeds, not here"}
            (out / "RL_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            return manifest

        def checkpoint(tag: str, it: int) -> None:
            w = weights()
            m = validate(tag, weights=w, mode="greedy")
            if m["score"] < best["score"]:
                best.update(score=m["score"], iter=it, metrics=m)
                np.savez(out / "policy.npz", **w)
            write_manifest(it)

        checkpoint("bc", 0)

        # 2. PPO
        for g in opt.param_groups:
            g["lr"] = cfg.lr
        for it in range(1, cfg.iters + 1):
            w = weights()
            rows = pool.map(_rollout, [{"scenario": sc, "seed": s, "weights": w, "mode": "sample", "record": True, "iter": it} for sc, s in draw(cfg.episodes)], chunksize=1)
            data = _cat([_gae(r["traj"], cfg) for r in rows])
            obs, mask, act = tt(data["obs"]), tt(data["mask"], torch.bool), tt(data["act"], torch.long)
            logp_old, ret = tt(data["logp"]), tt(data["ret"])
            choice = tt(data["mask"].sum(1) > 1)
            adv = tt(data["adv"])
            adv = (adv - adv[choice > 0].mean()) / (adv[choice > 0].std() + 1e-8)
            stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "n": 0}
            for _ in range(cfg.epochs):
                perm = torch.randperm(len(obs))
                for b in range(0, len(perm), cfg.minibatch):
                    i = perm[b:b + cfg.minibatch]
                    c = choice[i]
                    d = net.dist(obs[i], mask[i])
                    lp = d.log_prob(act[i])
                    ratio = torch.exp(lp - logp_old[i])
                    surr = torch.min(ratio * adv[i], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[i])
                    denom = c.sum().clamp(min=1.0)
                    pi_loss = -(surr * c).sum() / denom
                    ent = (d.entropy() * c).sum() / denom
                    v_loss = nn.functional.mse_loss(net.v(obs[i])[:, 0], ret[i])
                    loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    stats["pi"] += float(pi_loss); stats["v"] += float(v_loss); stats["ent"] += float(ent)
                    stats["kl"] += float(((logp_old[i] - lp) * c).sum() / denom); stats["n"] += 1
            n = max(1, stats["n"])
            acts = np.sum([r["counts"] for r in rows], axis=0)
            log({"stage": "ppo", "iter": it, "rows": int(len(obs)), "choice_rows": int(choice.sum()), "rollout": {k: round(v, 3) for k, v in _mean_metrics(rows).items()},
                 "actions": acts.tolist(), "pi_loss": round(stats["pi"] / n, 4), "v_loss": round(stats["v"] / n, 4), "entropy": round(stats["ent"] / n, 4), "kl": round(stats["kl"] / n, 5)})
            if it % cfg.val_every == 0 or it == cfg.iters:
                checkpoint(f"iter{it}", it)

    np.savez(out / "final.npz", **weights())
    return write_manifest(cfg.iters)
