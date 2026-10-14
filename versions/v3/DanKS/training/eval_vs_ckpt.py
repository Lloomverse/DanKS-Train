#!/usr/bin/env python3
"""Play two local online selectors against each other."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from DanKS.retrieval import StructuralCandidateRanker
from DanKS.training.accelerator import initialize_device
from DanKS.training.train_online import (
    _apply_decision,
    _apply_worker_env,
    _load_model,
    _observation_job,
    _rank_and_label,
    _start_table,
)


def _mark_lead(table, lead: bool) -> None:
    if not table.history:
        raise RuntimeError("applied a move but history is empty")
    table.history[-1]["lead"] = bool(lead)


def _local_slot(
    model: torch.nn.Module,
    ranked_result: dict[str, object],
    history: object,
    device: torch.device,
) -> int:
    state = torch.as_tensor(ranked_result["state"], dtype=torch.float32, device=device).unsqueeze(0)
    candidates = torch.as_tensor(
        ranked_result["candidates"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask = torch.as_tensor(ranked_result["mask"], dtype=torch.float32, device=device).unsqueeze(0)
    hist = torch.as_tensor(history, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        logits, _value = model(state, candidates, mask, hist)
        filled = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        slot = int(filled.argmax(dim=1)[0].item())
    if slot < 0 or slot >= len(ranked_result["ranked"]):
        raise ValueError(f"policy slot {slot} outside ranked {len(ranked_result['ranked'])}")
    if float(ranked_result["mask"][slot]) <= 0:
        raise RuntimeError(f"policy selected masked slot {slot}")
    return slot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def play_games(
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    ranker: StructuralCandidateRanker,
    games: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    if games <= 0:
        raise ValueError(f"games must be > 0, got {games}")
    if games % 2 != 0:
        raise ValueError(f"games must be even so seats can swap, got {games}")
    wins_a = 0
    score_a_sum = 0.0
    score_b_sum = 0.0
    for game_idx in range(games):
        team_a = game_idx % 2
        table = _start_table()
        rewards = None
        while rewards is None:
            job = _observation_job(table)
            seat = int(job["seat"])
            ranked_result = _rank_and_label(job, ranker)
            model = model_a if seat % 2 == team_a else model_b
            slot = _local_slot(model, ranked_result, job["history"], device)
            rewards = _apply_decision(table, job, ranked_result, slot)
            _mark_lead(table, bool(job["lead"]))
        if rewards[0] == rewards[1]:
            raise RuntimeError(f"tie rewards {rewards}")
        reward_a = float(rewards[team_a])
        reward_b = float(rewards[1 - team_a])
        if reward_a > 0:
            wins_a += 1
        score_a_sum += reward_a / 3.0
        score_b_sum += reward_b / 3.0
        print(
            f"Game {game_idx + 1}/{games} team_a={team_a} "
            f"wins_a={wins_a} scores_a={score_a_sum:.4f} scores_b={score_b_sum:.4f}",
            flush=True,
        )
    return {
        "games": games,
        "win_rate_a": wins_a / games,
        "win_rate_b": (games - wins_a) / games,
        "avg_score_a": score_a_sum / games,
        "avg_score_b": score_b_sum / games,
    }


def main() -> None:
    args = parse_args()
    ckpt_a = Path(args.checkpoint_a)
    ckpt_b = Path(args.checkpoint_b)
    if not ckpt_a.is_file():
        raise FileNotFoundError(f"checkpoint-a not found: {ckpt_a}")
    if not ckpt_b.is_file():
        raise FileNotFoundError(f"checkpoint-b not found: {ckpt_b}")
    _apply_worker_env()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = initialize_device(args.device)
    model_a = _load_model(str(ckpt_a), device)
    model_b = _load_model(str(ckpt_b), device)
    model_a.eval()
    model_b.eval()
    ranker = StructuralCandidateRanker()
    result = play_games(
        model_a=model_a,
        model_b=model_b,
        ranker=ranker,
        games=args.games,
        device=device,
    )
    result["checkpoint_a"] = str(ckpt_a)
    result["checkpoint_b"] = str(ckpt_b)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
