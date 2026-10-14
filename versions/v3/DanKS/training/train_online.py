#!/usr/bin/env python3
"""In-memory self-play loop for the existing V3 Top10 selector.

Optional Ray cluster jobs take RAY_ADDRESS from the environment;
the default path is single-machine colocated training.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import random
import select
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from collections import Counter
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.reduction import DupFd
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parent
V3_ROOT = TRAINING_DIR.parents[1]
REPO_ROOT = TRAINING_DIR.parents[2]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guandan import Environment
from guandan.engine.types import Move

from DanKS.retrieval import StructuralCandidateRanker, build_context
from DanKS.retrieval.cards import normalize_card, normalize_cards
from DanKS.retrieval.models import ActionCandidate
from DanKS.retrieval.rules import normalize_kind, normalize_rank
from DanKS.training.accelerator import cpu_state_dict, initialize_device, npu_available
from DanKS.training.featurizer import IncrementalHistoryFeaturizer, featurize_topk
from DanKS.training.model import (
    EfficientTeamBeliefTop10Selector,
    build_selector_from_checkpoint,
    selector_model_type,
)
from DanKS.training.ppo import (
    PPOConfig,
    Top10PPOAgent,
    compute_gae,
)
from DanKS.training.schema import (
    CANDIDATE_DIM,
    FEATURE_VERSION,
    HISTORY_EVENT_DIM,
    HISTORY_EVENT_SEMANTICS,
    HISTORY_LENGTH,
    HISTORY_PROTOCOL,
    STATE_DIM,
    STRUCTURED_TOPK,
    TEAM_BELIEF_PROTOCOL,
    TEAM_BELIEF_SEAT_COUNT,
    TEAM_BELIEF_TARGET_DIM,
    TOPK,
)
from DanKS.training.team_belief import build_team_belief_labels


_ENGINE_CARD = {"SB": "BJ", "HR": "RJ"}
_MAX_STEPS = 400
_OFFICIAL_MIN_STEPS = 2048
# A larger rollout amortizes fixed per-update H2D/PPO/sync overhead and is the
# default throughput setting.  The official lower bound remains 2048 so a
# caller can still request smaller validated windows explicitly.
_COLLECT_MIN_STEPS = 8192
_SAVE_EVERY_UPDATES = 100
_DEFAULT_NUM_TABLES = 256
_DEFAULT_INFER_BATCH = 256
_SLOW_RANK_SEC = 5.0
_INFER_COALESCE_MIN = 128
_INFER_COALESCE_SEC = 0.008
_ACTION_EMPTY = -2
_ACTION_STOP = -1


class _CollectStopped(Exception):
    """Sampler thread stopped while collect was waiting."""
_TABLE_TIME_KEYS = (
    "observe_sec",
    "rank_sec",
    "featurize_sec",
    "team_belief_sec",
    "apply_sec",
    "table_reset_sec",
)
_WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "DANKS_NATIVE_BATCH_INCLUDE_BEFORE_HAND": "1",
    "DANKS_NATIVE_BATCH_REUSE_COVER_INPUTS": "1",
    "DANKS_NATIVE_BEAM_BATCH": "1",
    "DANKS_NATIVE_BEAM_SUPER_BATCH": "1",
}
_TRIBUTE_SUIT = {"D": 0, "C": 1, "H": 2, "S": 3}
_TRIBUTE_FACE = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}
_TRIBUTE_KIND = {
    "PASS": 0,
    "Single": 1,
    "Pair": 2,
    "Triple": 3,
    "TriplePlus": 4,
    "StraightPair": 5,
    "StraightTriple": 6,
    "Straight": 7,
    "StraightFlush": 10,
    "FourKings": 16,
}
_TRIBUTE_BOMB = {4: 8, 5: 9, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15}
_TRIBUTE_PLAYTYPE_JD = 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tables", type=int, default=_DEFAULT_NUM_TABLES)
    parser.add_argument(
        "--infer-batch",
        type=int,
        default=_DEFAULT_INFER_BATCH,
        help="max tables per GPU call; extra ready tables stay queued",
    )
    parser.add_argument("--min-update-steps", type=int, default=_COLLECT_MIN_STEPS)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--output",
        default=str(TRAINING_DIR.parent / "checkpoints"),
        help="experiment directory; writes "
        "top10_selector_online_<update>-<batchsize>.pt every "
        f"{_SAVE_EVERY_UPDATES} updates",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--infer-device",
        required=True,
        help="sampling/inference GPU; must differ from --device",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _engine_card(card: object) -> str:
    label = str(card)
    return normalize_card(_ENGINE_CARD.get(label, label))


def _engine_cards(cards: object) -> list[str]:
    if cards in (None, Move.PASS, "PASS"):
        return []
    if not isinstance(cards, (list, tuple)):
        raise TypeError(f"engine cards must be a sequence, got {type(cards).__name__}")
    return [_engine_card(card) for card in cards]


def _hand(game: Environment, seat: int) -> list[str]:
    return [_engine_card(card) for card in game.players[seat].hand_cards]


def _engine_legal_actions(game: Environment) -> list[ActionCandidate]:
    actions: list[ActionCandidate] = []
    for index, raw in enumerate(game.legal_moves.action_list):
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(f"illegal engine action row: {raw!r}")
        kind_raw, rank_raw, cards_raw = raw
        if kind_raw == Move.PASS:
            actions.append(ActionCandidate(index=index, kind="PASS", cards=(), rank="PASS"))
            continue
        kind = normalize_kind(str(kind_raw))
        if kind == "PASS":
            raise ValueError(f"engine non-pass row normalized to PASS: {raw!r}")
        rank = normalize_rank(rank_raw)
        if rank is None:
            raise ValueError(f"engine action has no rank: {raw!r}")
        cards = tuple(normalize_cards(_engine_cards(cards_raw)))
        if not cards:
            raise ValueError(f"engine action has no cards: {raw!r}")
        actions.append(ActionCandidate(index=index, kind=kind, cards=cards, rank=rank))
    if not actions:
        raise ValueError("engine legal action list is empty during play")
    return actions


def _tribute_card_id(card: str) -> int:
    if card == "BJ":
        return 1037
    if card == "RJ":
        return 1294
    if len(card) != 2 or card[0] not in _TRIBUTE_SUIT or card[1] not in _TRIBUTE_FACE:
        raise ValueError(f"unmapped tribute card: {card!r}")
    return (_TRIBUTE_SUIT[card[0]] << 8) + _TRIBUTE_FACE[card[1]]


def _tribute_pattern(kind: str, cards: tuple[str, ...]) -> int:
    if kind == "Bomb":
        if len(cards) not in _TRIBUTE_BOMB:
            raise ValueError(f"bomb size {len(cards)} has no tribute pattern")
        return _TRIBUTE_BOMB[len(cards)]
    if kind not in _TRIBUTE_KIND:
        raise ValueError(f"unmapped tribute kind: {kind!r}")
    return _TRIBUTE_KIND[kind]


def _tribute_action_key(pattern: int, card_ids: list[int]) -> tuple[int, tuple[int, ...]]:
    return (int(pattern), tuple(sorted(int(card_id) for card_id in card_ids)))


def _engine_action_key(action: ActionCandidate) -> tuple[int, tuple[int, ...]]:
    if action.kind == "PASS":
        return _tribute_action_key(0, [])
    return _tribute_action_key(
        _tribute_pattern(action.kind, action.cards),
        [_tribute_card_id(card) for card in action.cards],
    )


def _filter_legal_v3(
    hand: list[str],
    legal: list[ActionCandidate],
    *,
    cur_rank: str | None,
    lead: bool,
    last_pattern: int,
    last_card_ids: list[int],
) -> list[ActionCandidate]:
    del hand, cur_rank, lead, last_pattern, last_card_ids
    if not legal:
        raise RuntimeError("engine legal action list is empty")
    return list(legal)


def _trick(game: Environment) -> tuple[str, str | None, int, int | None]:
    move = game.state.greater_action
    if game.action_first or move.type in (None, Move.PASS):
        return "Lead", None, 0, None
    kind = normalize_kind(str(move.type))
    if kind == "PASS":
        raise ValueError("follow target normalized to PASS")
    rank = normalize_rank(move.rank)
    cards = _engine_cards(move.cards)
    if not cards:
        raise ValueError("follow target has no cards")
    if game.state.greater_pos < 0 or game.state.greater_pos > 3:
        raise ValueError(f"invalid greater_pos={game.state.greater_pos}")
    return kind, rank, len(cards), int(game.state.greater_pos)


def _retrieval_context(
    game: Environment,
    seat: int,
    history: list[dict[str, object]],
    played_cards: list[str],
):
    current_kind, current_rank, current_size, last_player = _trick(game)
    hand = _hand(game, seat)
    return hand, build_context(
        {
            "my_seat": seat,
            "curRank": game.rank,
            "public_counts": [len(player.hand_cards) for player in game.players],
            "current_kind": current_kind,
            "current_rank": current_rank,
            "current_size": current_size,
            "last_player": last_player,
            "played_cards": list(played_cards),
            "history": history,
            "known_hand_cards": {str(seat): hand},
        }
    )


def _match_engine_index(selected: ActionCandidate, legal: list[ActionCandidate]) -> int:
    selected_cards = Counter(selected.cards)
    if 0 <= selected.index < len(legal):
        candidate = legal[selected.index]
        if candidate.kind == selected.kind and Counter(candidate.cards) == selected_cards:
            return selected.index
    for action in legal:
        if action.kind == selected.kind and Counter(action.cards) == selected_cards:
            if action.index < 0 or action.index >= len(legal):
                raise ValueError(f"ranked action index out of engine range: {action.index}")
            return action.index
    raise ValueError(
        f"ranked action not in engine legal list: kind={selected.kind} cards={selected.cards}"
    )


def _terminal_rewards(game: Environment) -> tuple[int, ...]:
    if game.belongs_to is None or game.rank_inc is None:
        raise ValueError("episode ended without settlement first_pos/inc")
    first = int(game.belongs_to)
    inc = int(game.rank_inc)
    if first not in range(4) or inc not in {1, 2, 3}:
        raise ValueError(f"invalid settlement first={first} inc={inc}")
    winners = {first, (first + 2) % 4}
    return tuple(inc if seat in winners else -inc for seat in range(4))


def _assign_gae(rows: list[dict[str, object]], rewards: tuple[int, ...]) -> list[dict[str, object]]:
    if not rows:
        raise RuntimeError("episode produced no decisions")
    by_seat: dict[int, list[int]] = {seat: [] for seat in range(4)}
    for index, row in enumerate(rows):
        by_seat[int(row["seat"])].append(index)
    for seat, indices in by_seat.items():
        if not indices:
            continue
        values = np.asarray([rows[index]["value"] for index in indices], dtype=np.float32)
        step_rewards = np.zeros(len(indices), dtype=np.float32)
        step_rewards[-1] = float(rewards[seat])
        dones = np.zeros(len(indices), dtype=np.float32)
        dones[-1] = 1.0
        advantage, returns = compute_gae(step_rewards, values, dones)
        for offset, index in enumerate(indices):
            rows[index]["advantage"] = float(advantage[offset])
            rows[index]["returns"] = float(returns[offset])
    missing = [index for index, row in enumerate(rows) if "advantage" not in row]
    if missing:
        raise RuntimeError(f"GAE missed rows {missing}")
    return rows


def _rank_and_label(
    job: dict[str, object],
    ranker: StructuralCandidateRanker,
) -> dict[str, object]:
    hand = job["hand"]
    legal = job["legal"]
    ctx = job["ctx"]
    if not isinstance(hand, list) or not isinstance(legal, list):
        raise TypeError("worker job hand/legal must be lists")
    if not isinstance(ctx.cur_rank, str):
        raise TypeError(f"ctx.cur_rank must be str, got {type(ctx.cur_rank).__name__}")
    last_card_ids = job["last_card_ids"]
    if not isinstance(last_card_ids, list):
        raise TypeError("job last_card_ids must be a list")
    lead = bool(job["lead"])
    filtered = _filter_legal_v3(
        hand,
        legal,
        cur_rank=ctx.cur_rank,
        lead=lead,
        last_pattern=int(job["last_pattern"]),
        last_card_ids=last_card_ids,
    )
    t_rank0 = time.perf_counter()
    ranked = ranker.rank(hand, filtered, ctx, top_k=TOPK)
    rank_sec = time.perf_counter() - t_rank0
    if not ranked:
        raise ValueError("ranker returned no candidates")
    if len(ranked) > TOPK:
        raise ValueError(f"ranker returned {len(ranked)} rows, expected <= {TOPK}")
    t_feat0 = time.perf_counter()
    state, candidates, mask = featurize_topk(hand, ctx, ranked, top_k=TOPK)
    featurize_sec = time.perf_counter() - t_feat0
    t_belief0 = time.perf_counter()
    labels, label_mask = build_team_belief_labels(
        job["all_hands"],
        actor_seat=int(job["seat"]),
        ctx=ctx,
        candidates=ranked,
        capacity=TOPK,
        partitioner=ranker.partitioner,
    )
    team_belief_sec = time.perf_counter() - t_belief0
    if rank_sec < 0.0 or featurize_sec < 0.0 or team_belief_sec < 0.0:
        raise RuntimeError(
            f"rank/featurize/belief timings are negative: "
            f"rank={rank_sec} featurize={featurize_sec} belief={team_belief_sec}"
        )
    return {
        "ranked": ranked,
        "state": state,
        "candidates": candidates,
        "mask": mask,
        "team_belief_labels": labels,
        "team_belief_mask": label_mask,
        "rank_sec": rank_sec,
        "featurize_sec": featurize_sec,
        "team_belief_sec": team_belief_sec,
        "lead": lead,
        "n_hand": len(hand),
        "n_legal": len(legal),
        "n_v3": len(filtered),
    }


@dataclass
class Table:
    game: Environment
    history: list[dict[str, object]]
    played_cards: list[str]
    encoder: IncrementalHistoryFeaturizer
    steps: int = 0


def _start_table() -> Table:
    game = Environment(first_player=random.randrange(4))
    for seat in range(4):
        game.add_player(f"p{seat}", seat)
    messages = game.start()
    if not messages or game.loop.__func__ is not Environment.play:
        raise RuntimeError("engine start did not enter play")
    if [len(player.hand_cards) for player in game.players] != [27, 27, 27, 27]:
        raise RuntimeError("engine start did not deal 27 cards to every seat")
    encoder = IncrementalHistoryFeaturizer()
    encoder.reset()
    return Table(game=game, history=[], played_cards=[], encoder=encoder)


def _observation_job(table: Table) -> dict[str, object]:
    game = table.game
    if game.loop.__func__ is not Environment.play:
        raise RuntimeError("asked for an observation after play ended")
    seat = int(game.state.current_pos)
    if seat not in range(4):
        raise ValueError(f"invalid current_pos={game.state.current_pos}")
    legal = _engine_legal_actions(game)
    hand, ctx = _retrieval_context(game, seat, table.history, table.played_cards)
    history = table.encoder.features(table.history, my_seat=seat)
    lead = bool(game.action_first or game.state.greater_action.type in (None, Move.PASS))
    last_pattern = -1
    last_card_ids: list[int] = []
    if not lead:
        move = game.state.greater_action
        last_cards = tuple(normalize_cards(_engine_cards(move.cards)))
        last_pattern = _tribute_pattern(normalize_kind(str(move.type)), last_cards)
        last_card_ids = [_tribute_card_id(card) for card in last_cards]
    return {
        "seat": seat,
        "legal": legal,
        "hand": hand,
        "ctx": ctx,
        "all_hands": [_hand(game, other) for other in range(4)],
        "history": history,
        "lead": lead,
        "last_pattern": last_pattern,
        "last_card_ids": last_card_ids,
    }


def _apply_decision(
    table: Table,
    job: dict[str, object],
    ranked_result: dict[str, object],
    slot: int,
) -> tuple[int, ...] | None:
    ranked = ranked_result["ranked"]
    if not isinstance(ranked, list):
        raise TypeError("ranked result must be a list")
    if slot < 0 or slot >= len(ranked):
        raise ValueError(f"policy selected slot {slot} outside ranked length {len(ranked)}")
    selected = ranked[slot].action
    legal = job["legal"]
    if not isinstance(legal, list):
        raise TypeError("job legal must be a list")
    act_index = _match_engine_index(selected, legal)
    table.game.loop({"actIndex": act_index})
    table.history.append(
        {
            "pos": int(job["seat"]),
            "kind": selected.kind,
            "rank": selected.rank,
            "cards": list(selected.cards),
        }
    )
    table.played_cards.extend(selected.cards)
    table.steps += 1
    if table.steps > _MAX_STEPS:
        raise RuntimeError(f"episode exceeded {_MAX_STEPS} steps")
    finished = table.game.loop.__func__ is not Environment.play
    if table.game.settlement.order and table.game.settlement.episode_over():
        finished = True
    if not finished:
        return None
    if not table.game.settlement.order or not table.game.settlement.episode_over():
        raise RuntimeError("episode left play without finishing")
    if table.steps <= 0:
        raise RuntimeError("episode produced no decisions")
    return _terminal_rewards(table.game)


def _rss_bytes(pid: int) -> int:
    if pid <= 0:
        raise ValueError(f"pid must be > 0, got {pid}")
    path = Path(f"/proc/{pid}/status")
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path} VmRSS line incomplete: {line!r}")
            if parts[-1] != "kB":
                raise ValueError(f"{path} VmRSS unit must be kB, got {parts[-1]!r}")
            kb = int(parts[1])
            if kb < 0:
                raise ValueError(f"{path} negative VmRSS: {kb}")
            return kb * 1024
    raise ValueError(f"{path} has no VmRSS")


def _device_memory_bytes(device: torch.device) -> dict[str, int] | None:
    if device.type == "cpu":
        return None
    if device.type == "cuda":
        return {
            "alloc": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
            "peak_alloc": int(torch.cuda.max_memory_allocated(device)),
        }
    if device.type == "npu":
        return {
            "alloc": int(torch.npu.memory_allocated(device)),
            "reserved": int(torch.npu.memory_reserved(device)),
            "peak_alloc": int(torch.npu.max_memory_allocated(device)),
        }
    raise ValueError(f"unsupported device type {device.type}")


def _bytes_to_mb(value: int) -> float:
    if value < 0:
        raise ValueError(f"byte count must be >= 0, got {value}")
    return value / (1024.0 * 1024.0)


def _cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"cuda index requested for device {device}")
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _min_mean_max(values: list[float], *, name: str) -> tuple[float, float, float]:
    if not values:
        raise RuntimeError(f"{name} samples is empty")
    total = 0.0
    for value in values:
        total += value
    return min(values), total / len(values), max(values)


def _percentile(values: list[float], q: float, *, name: str) -> float:
    if not values:
        raise RuntimeError(f"{name} percentile on empty")
    if q < 0.0 or q > 100.0:
        raise ValueError(f"{name} q={q} not in [0, 100]")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"explained variance shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size < 2:
        raise RuntimeError(f"explained variance needs >= 2 samples, got {y_true.size}")
    var_y = float(np.var(y_true))
    if var_y <= 0.0:
        raise RuntimeError(f"returns variance must be > 0, got {var_y}")
    residual = y_true - y_pred
    var_res = float(np.var(residual))
    if var_res < 0.0:
        raise RuntimeError(f"residual variance is negative: {var_res}")
    return 1.0 - var_res / var_y


def _mean_or_raise(total: float, n: int, *, name: str) -> float:
    if n <= 0:
        raise RuntimeError(f"{name} count must be > 0, got {n}")
    if total < 0.0:
        raise RuntimeError(f"{name} sum is negative: {total}")
    return total / float(n)


def _smi_float(parsed: dict[str, str], key: str) -> float:
    raw = parsed[key]
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"nvidia-smi {key}={raw!r} is not a number") from exc
    if value < 0.0:
        raise RuntimeError(f"nvidia-smi {key}={value} is negative")
    return value


def _query_nvidia_smi(device: torch.device) -> dict[str, str]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("CUDA perf report requires nvidia-smi in PATH")
    index = _cuda_index(device)
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,nounits,noheader",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"nvidia-smi failed exit={exc.returncode} stderr={exc.stderr.strip()!r}"
        ) from exc
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"nvidia-smi returned {len(lines)} GPU rows, expected 1: {lines!r}")
    parts = [part.strip() for part in lines[0].split(",")]
    names = (
        "name",
        "sm_util_pct",
        "mem_util_pct",
        "mem_used_mb",
        "mem_total_mb",
        "temp_c",
        "power_w",
        "power_limit_w",
    )
    if len(parts) != len(names):
        raise RuntimeError(f"nvidia-smi field count {len(parts)} != {len(names)}: {parts!r}")
    parsed = dict(zip(names, parts, strict=True))
    for key in ("sm_util_pct", "mem_util_pct", "mem_used_mb", "mem_total_mb"):
        if parsed[key] in {"", "[N/A]", "N/A"}:
            raise RuntimeError(f"nvidia-smi {key} is unavailable: {parsed[key]!r}")
    return parsed


def _empty_table_times() -> dict[str, float]:
    return {key: 0.0 for key in _TABLE_TIME_KEYS}


def _require_table_perf(payload: object, *, table_id: int) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError(f"table {table_id} perf must be a dict, got {type(payload).__name__}")
    for key in _TABLE_TIME_KEYS:
        if key not in payload:
            raise KeyError(f"table {table_id} perf missing {key}")
        seconds = float(payload[key])
        if seconds < 0.0:
            raise ValueError(f"table {table_id} {key}={seconds} is negative")
    for key in ("observe_calls", "finished_episodes", "steps", "rss_bytes"):
        if key not in payload:
            raise KeyError(f"table {table_id} perf missing {key}")
        count = int(payload[key])
        if count < 0:
            raise ValueError(f"table {table_id} {key}={count} is negative")
    if "cpu_util_pct" not in payload:
        raise KeyError(f"table {table_id} perf missing cpu_util_pct")
    if float(payload["cpu_util_pct"]) < 0.0:
        raise ValueError(f"table {table_id} cpu_util_pct is negative")
    return payload


def _print_perf_report(
    *,
    update: int,
    updates: int,
    steps: int,
    num_tables: int,
    infer_batch: int,
    infer_calls: int,
    device: torch.device,
    collect_wall: float,
    total_wall: float,
    parent_times: dict[str, float],
    parent_rss: int,
    parent_gpu: dict[str, int] | None,
    table_perfs: list[dict[str, object]],
    train_info: dict[str, float],
    pipeline: dict[str, float | int],
) -> None:
    if collect_wall <= 0.0:
        raise RuntimeError(f"collect_wall must be > 0, got {collect_wall}")
    if total_wall <= 0.0:
        raise RuntimeError(f"total_wall must be > 0, got {total_wall}")
    if len(table_perfs) != num_tables:
        raise ValueError(f"table perf count {len(table_perfs)} != num_tables {num_tables}")
    if infer_batch <= 0:
        raise ValueError(f"infer_batch must be > 0, got {infer_batch}")
    sums = _empty_table_times()
    maxes = _empty_table_times()
    observe_calls = 0
    finished_episodes = 0
    for table_id, perf in enumerate(table_perfs):
        checked = _require_table_perf(perf, table_id=table_id)
        for key in _TABLE_TIME_KEYS:
            value = float(checked[key])
            sums[key] += value
            if value > maxes[key]:
                maxes[key] = value
        observe_calls += int(checked["observe_calls"])
        finished_episodes += int(checked["finished_episodes"])
    for key in (
        "batch_wait_sec",
        "infer_sec",
        "stack_sec",
        "update_sec",
        "save_sec",
        "cpu_util_pct",
    ):
        if key not in parent_times:
            raise KeyError(f"parent_times missing {key}")
        if float(parent_times[key]) < 0.0:
            raise ValueError(f"{key}={parent_times[key]} is negative")
    if infer_calls < 0 or observe_calls < 0:
        raise RuntimeError(
            f"perf counts must be >= 0, infer_calls={infer_calls} observe_calls={observe_calls}"
        )
    if infer_calls == 0 and observe_calls != 0:
        raise RuntimeError(
            f"leftover collect has observe_calls={observe_calls} with infer_calls=0"
        )
    if infer_calls > 0 and observe_calls <= 0:
        raise RuntimeError(
            f"perf counts must be > 0, infer_calls={infer_calls} observe_calls={observe_calls}"
        )
    missing_train = [
        key
        for key in ("loss", "kl", "entropy", "explained_variance")
        if key not in train_info
    ]
    if missing_train:
        raise KeyError(f"train info missing {missing_train}")
    if finished_episodes <= 0:
        raise RuntimeError("no finished episodes in collect window")
    traj_len = float(steps) / float(finished_episodes)
    worker_sec = sum(
        sums[key]
        for key in (
            "observe_sec",
            "rank_sec",
            "featurize_sec",
            "team_belief_sec",
            "apply_sec",
            "table_reset_sec",
        )
    )
    if worker_sec <= 0.0:
        raise RuntimeError(f"worker timing collapsed to {worker_sec}")
    infer_sec = float(parent_times["infer_sec"])
    update_sec = float(parent_times["update_sec"])
    rank_mean = 0.0 if observe_calls == 0 else sums["rank_sec"] / float(observe_calls)
    for key in (
        "rank_call_max",
        "batch_size_mean",
        "batch_size_max",
        "ready_nowait_mean",
        "q_left_mean",
        "q_left_max",
    ):
        if key not in pipeline:
            raise KeyError(f"pipeline missing {key}")
        if float(pipeline[key]) < 0.0:
            raise ValueError(f"{key}={pipeline[key]} is negative")
    rank_call_max = float(pipeline["rank_call_max"])
    batch_mean = float(pipeline["batch_size_mean"])
    batch_max = float(pipeline["batch_size_max"])
    nowait_mean = float(pipeline["ready_nowait_mean"])
    q_left_mean = float(pipeline["q_left_mean"])
    q_left_max = float(pipeline["q_left_max"])
    for key in ("feature_copy_sec", "h2d_sec", "d2h_sec", "dispatch_sec"):
        if key not in parent_times:
            raise KeyError(f"parent_times missing {key}")
        if float(parent_times[key]) < 0.0:
            raise ValueError(f"{key}={parent_times[key]} is negative")
    worker_cpu = 0.0
    for perf in table_perfs:
        worker_cpu += float(perf["cpu_util_pct"])
    worker_cpu /= float(num_tables)
    print(
        f"update={update}/{updates} steps={steps} episodes={finished_episodes} "
        f"collect={collect_wall:.2f}s infer={infer_sec:.2f}s "
        f"stack={float(parent_times['stack_sec']):.3f}s ppo={update_sec:.2f}s "
        f"total={total_wall:.2f}s step/s={steps / total_wall:.1f} "
        f"rank={100.0 * sums['rank_sec'] / worker_sec:.0f}% "
        f"rank_mean={rank_mean:.3f}s rank_max={rank_call_max:.2f}s "
        f"infer_n={infer_calls} batch_mean={batch_mean:.1f} batch_max={batch_max:.0f} "
        f"nowait={nowait_mean:.1f} q_left={q_left_mean:.1f}(max={q_left_max:.0f}) "
        f"copy={float(parent_times['feature_copy_sec']):.3f}s "
        f"h2d={float(parent_times['h2d_sec']):.3f}s "
        f"d2h={float(parent_times['d2h_sec']):.3f}s "
        f"dispatch={float(parent_times['dispatch_sec']):.3f}s "
        f"wcpu={worker_cpu:.0f}% "
        f"loss={float(train_info['loss']):.4f} kl={float(train_info['kl']):.4f} "
        f"entropy={float(train_info['entropy']):.3f} "
        f"ef={float(train_info['explained_variance']):.3f} "
        f"traj={traj_len:.1f}",
        flush=True,
    )


def _apply_worker_env() -> None:
    for key, value in _WORKER_ENV.items():
        os.environ[key] = value


def _pin_worker_threads() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    intra = torch.get_num_threads()
    inter = torch.get_num_interop_threads()
    if intra != 1 or inter != 1:
        raise RuntimeError(f"worker threads must be 1, got intra={intra} interop={inter}")


def _policy_device_tensors(
    batch: list[dict[str, object]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(
            np.stack([row["state"] for row in batch]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack([row["candidates"] for row in batch]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack([row["mask"] for row in batch]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack([row["history"] for row in batch]),
            dtype=torch.float32,
            device=device,
        ),
    )


def _policy_device_tensors_from_slots(
    slots: "_FeatureSlots",
    table_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not table_ids:
        raise ValueError("table_ids is empty")
    tensors: list[torch.Tensor] = []
    for name in ("state", "candidates", "mask", "history"):
        first = slots.field_view(table_ids[0], name)
        stacked = np.empty((len(table_ids),) + first.shape, dtype=np.float32)
        stacked[0] = first
        for index, table_id in enumerate(table_ids[1:], start=1):
            stacked[index] = slots.field_view(table_id, name)
        tensors.append(torch.as_tensor(stacked, dtype=torch.float32, device=device))
    return tensors[0], tensors[1], tensors[2], tensors[3]


def _infer_prefix_floats() -> int:
    nbytes = _infer_prefix_bytes()
    if nbytes % 4 != 0:
        raise RuntimeError(f"infer prefix bytes {nbytes} not aligned to float32")
    prefix_n = nbytes // 4
    expected_n = (
        STATE_DIM + TOPK * CANDIDATE_DIM + TOPK + HISTORY_LENGTH * HISTORY_EVENT_DIM
    )
    if prefix_n != expected_n:
        raise RuntimeError(f"infer prefix floats {prefix_n} != {expected_n}")
    return prefix_n


def _graph_bucket_sizes(max_batch: int) -> tuple[int, ...]:
    if max_batch <= 0:
        raise ValueError(f"max_batch must be > 0, got {max_batch}")
    sizes: list[int] = []
    size = 64
    while size < max_batch:
        sizes.append(size)
        size *= 2
    sizes.append(max_batch)
    return tuple(sizes)


def _split_infer_pack(
    pack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pack.ndim != 2:
        raise RuntimeError(f"infer pack ndim {pack.ndim} != 2")
    batch, prefix_n = pack.shape
    if prefix_n != _infer_prefix_floats():
        raise RuntimeError(f"infer pack width {prefix_n} != {_infer_prefix_floats()}")
    off = 0
    state = pack[:, off : off + STATE_DIM]
    off += STATE_DIM
    cand = TOPK * CANDIDATE_DIM
    candidates = pack[:, off : off + cand].reshape(batch, TOPK, CANDIDATE_DIM)
    off += cand
    mask = pack[:, off : off + TOPK]
    off += TOPK
    hist = HISTORY_LENGTH * HISTORY_EVENT_DIM
    history = pack[:, off : off + hist].reshape(
        batch, HISTORY_LENGTH, HISTORY_EVENT_DIM
    )
    off += hist
    if off != prefix_n:
        raise RuntimeError(f"infer pack offset {off} != prefix {prefix_n}")
    return state, candidates, mask, history


class _CompiledInfer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_batch: int,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError(f"compiled infer requires cuda, got {device}")
        if max_batch <= 0:
            raise ValueError(f"max_batch must be > 0, got {max_batch}")
        torch.cuda.set_device(device)
        self._max_batch = max_batch
        self._device = device
        self._model = model
        self._stream = torch.cuda.Stream(device=device)
        prefix_n = _infer_prefix_floats()
        self._h_pack = torch.empty((max_batch, prefix_n), pin_memory=True)
        self._np_pack = self._h_pack.numpy()
        h_state, h_candidates, h_mask, h_history = _split_infer_pack(self._h_pack)
        self._np_state = h_state.numpy()
        self._np_candidates = h_candidates.numpy()
        self._np_mask = h_mask.numpy()
        self._np_history = h_history.numpy()
        self._h_action = torch.empty((max_batch,), dtype=torch.long, pin_memory=True)
        self._h_logp = torch.empty((max_batch,), dtype=torch.float32, pin_memory=True)
        self._h_value = torch.empty((max_batch,), dtype=torch.float32, pin_memory=True)
        self._np_action = self._h_action.numpy()
        self._np_logp = self._h_logp.numpy()
        self._np_value = self._h_value.numpy()
        self._buckets: list[dict[str, object]] = []
        for size in _graph_bucket_sizes(max_batch):
            self._buckets.append(self._capture_bucket(size, prefix_n))
        if not self._buckets:
            raise RuntimeError("infer captured no CUDA graph buckets")
        self._action: torch.Tensor | None = None
        self._logp: torch.Tensor | None = None
        self._value: torch.Tensor | None = None
        self._batch = 0

    def _policy_sample(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.shape != mask.shape:
            raise RuntimeError(
                f"sample logits {tuple(logits.shape)} != mask {tuple(mask.shape)}"
            )
        masked = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        action = torch.multinomial(torch.softmax(masked, dim=-1), 1).squeeze(-1)
        if action.shape != (logits.shape[0],):
            raise RuntimeError(
                f"sample action width {action.shape[0]} != {logits.shape[0]}"
            )
        logp = torch.log_softmax(masked, dim=-1).gather(
            -1, action.unsqueeze(-1)
        ).squeeze(-1)
        if logp.shape != action.shape:
            raise RuntimeError(
                f"sample logp {tuple(logp.shape)} != action {tuple(action.shape)}"
            )
        return action, logp

    def _capture_bucket(self, size: int, prefix_n: int) -> dict[str, object]:
        if size <= 0 or size > self._max_batch:
            raise ValueError(f"graph bucket {size} outside 1..{self._max_batch}")
        d_pack = torch.empty((size, prefix_n), device=self._device)
        d_state, d_candidates, d_mask, d_history = _split_infer_pack(d_pack)
        graph = torch.cuda.CUDAGraph()
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            d_mask.zero_()
            d_mask[:, 0] = 1.0
            for _ in range(3):
                logits, value = self._model(
                    d_state,
                    d_candidates,
                    d_mask,
                    d_history,
                )
                if logits.shape[0] != size or value.shape[0] != size:
                    raise RuntimeError(
                        f"graph warmup width {logits.shape[0]}/{value.shape[0]} "
                        f"!= bucket {size}"
                    )
                action, logp = self._policy_sample(logits, d_mask)
                if action.shape[0] != size or logp.shape[0] != size:
                    raise RuntimeError(
                        f"graph warmup sample width {action.shape[0]}/{logp.shape[0]} "
                        f"!= bucket {size}"
                    )
            stream.synchronize()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream), torch.inference_mode():
            logits, value = self._model(
                d_state,
                d_candidates,
                d_mask,
                d_history,
            )
            action, logp = self._policy_sample(logits, d_mask)
            g_action = action
            g_logp = logp
            g_value = value
        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        if (
            g_action.shape[0] != size
            or g_logp.shape[0] != size
            or g_value.shape[0] != size
        ):
            raise RuntimeError(
                f"graph capture width {g_action.shape[0]}/{g_logp.shape[0]}/"
                f"{g_value.shape[0]} != bucket {size}"
            )
        d_mask[:, :2] = 1.0
        graph.replay()
        stream.synchronize()
        first = g_action.detach().clone()
        graph.replay()
        stream.synchronize()
        if torch.equal(first, g_action):
            raise RuntimeError(f"graph sample RNG is frozen across replays bucket={size}")
        d_mask.zero_()
        d_mask[:, 0] = 1.0
        return {
            "size": size,
            "graph": graph,
            "d_pack": d_pack,
            "g_action": g_action,
            "g_logp": g_logp,
            "g_value": g_value,
        }

    def _pick_bucket(self, batch: int) -> dict[str, object]:
        for bucket in self._buckets:
            size = int(bucket["size"])
            if size >= batch:
                return bucket
        raise RuntimeError(
            f"no infer bucket >= {batch}, max={self._max_batch}"
        )

    def submit(
        self,
        slots: "_FeatureSlots",
        table_ids: list[int],
    ) -> None:
        batch = len(table_ids)
        if batch <= 0 or batch > self._max_batch:
            raise RuntimeError(f"infer batch {batch} outside 1..{self._max_batch}")
        bucket = self._pick_bucket(batch)
        d_pack = bucket["d_pack"]
        graph = bucket["graph"]
        g_action = bucket["g_action"]
        g_logp = bucket["g_logp"]
        g_value = bucket["g_value"]
        if not isinstance(d_pack, torch.Tensor) or not isinstance(graph, torch.cuda.CUDAGraph):
            raise TypeError("infer bucket tensors/graph have the wrong type")
        if (
            not isinstance(g_action, torch.Tensor)
            or not isinstance(g_logp, torch.Tensor)
            or not isinstance(g_value, torch.Tensor)
        ):
            raise TypeError("infer bucket outputs have the wrong type")
        slots.copy_infer_prefixes(table_ids, self._np_pack[:batch])
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            d_pack[:batch].copy_(self._h_pack[:batch], non_blocking=True)
            graph.replay()
            self._action = g_action[:batch]
            self._logp = g_logp[:batch]
            self._value = g_value[:batch]
        self._batch = batch

    def run_numpy(
        self,
        states: list[np.ndarray],
        candidates: list[np.ndarray],
        masks: list[np.ndarray],
        histories: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = len(states)
        if (
            batch <= 0
            or batch > self._max_batch
            or len(candidates) != batch
            or len(masks) != batch
            or len(histories) != batch
        ):
            raise RuntimeError(
                f"infer run_numpy width {batch}/{len(candidates)}/{len(masks)}/"
                f"{len(histories)} outside 1..{self._max_batch}"
            )
        bucket = self._pick_bucket(batch)
        d_pack = bucket["d_pack"]
        graph = bucket["graph"]
        g_action = bucket["g_action"]
        g_logp = bucket["g_logp"]
        g_value = bucket["g_value"]
        if not isinstance(d_pack, torch.Tensor) or not isinstance(graph, torch.cuda.CUDAGraph):
            raise TypeError("infer bucket tensors/graph have the wrong type")
        if (
            not isinstance(g_action, torch.Tensor)
            or not isinstance(g_logp, torch.Tensor)
            or not isinstance(g_value, torch.Tensor)
        ):
            raise TypeError("infer bucket outputs have the wrong type")
        for index in range(batch):
            np.copyto(self._np_state[index], states[index], casting="no")
            np.copyto(self._np_candidates[index], candidates[index], casting="no")
            np.copyto(self._np_mask[index], masks[index], casting="no")
            np.copyto(self._np_history[index], histories[index], casting="no")
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            d_pack[:batch].copy_(self._h_pack[:batch], non_blocking=True)
            graph.replay()
            self._action = g_action[:batch]
            self._logp = g_logp[:batch]
            self._value = g_value[:batch]
        self._batch = batch
        return self.take_numpy()

    def take_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._action is None or self._logp is None or self._value is None:
            raise RuntimeError("infer take without submit")
        if self._batch <= 0:
            raise RuntimeError("infer batch is empty")
        stream = self._stream
        with torch.cuda.stream(stream):
            self._h_action[: self._batch].copy_(self._action, non_blocking=True)
            self._h_logp[: self._batch].copy_(self._logp, non_blocking=True)
            self._h_value[: self._batch].copy_(self._value, non_blocking=True)
        stream.synchronize()
        action = self._np_action[: self._batch].copy()
        logp = self._np_logp[: self._batch].copy()
        value = self._np_value[: self._batch].copy()
        if (
            action.shape[0] != self._batch
            or logp.shape[0] != self._batch
            or value.shape[0] != self._batch
        ):
            raise RuntimeError(
                f"infer output width {action.shape[0]}/{logp.shape[0]}/{value.shape[0]} "
                f"!= batch {self._batch}"
            )
        self._action = None
        self._logp = None
        self._value = None
        self._batch = 0
        return action, logp, value


def _feature_fields() -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (
        ("state", (STATE_DIM,)),
        ("candidates", (TOPK, CANDIDATE_DIM)),
        ("mask", (TOPK,)),
        ("history", (HISTORY_LENGTH, HISTORY_EVENT_DIM)),
        ("team_belief_labels", (TOPK, TEAM_BELIEF_SEAT_COUNT, TEAM_BELIEF_TARGET_DIM)),
        ("team_belief_mask", (TOPK, TEAM_BELIEF_SEAT_COUNT)),
    )


def _feature_slot_bytes() -> int:
    size = 0
    for _, shape in _feature_fields():
        n = 1
        for dim in shape:
            n *= dim
        size += n * 4
    if size <= 0:
        raise RuntimeError(f"feature slot size must be > 0, got {size}")
    return size


def _infer_prefix_bytes() -> int:
    size = 0
    for name, shape in _feature_fields():
        if name not in ("state", "candidates", "mask", "history"):
            break
        n = 1
        for dim in shape:
            n *= dim
        size += n * 4
    expected = (
        STATE_DIM + TOPK * CANDIDATE_DIM + TOPK + HISTORY_LENGTH * HISTORY_EVENT_DIM
    ) * 4
    if size != expected:
        raise RuntimeError(f"infer prefix bytes {size} != {expected}")
    return size


def _feature_slot_stride() -> int:
    size = _feature_slot_bytes()
    stride = (size + 63) // 64 * 64
    if stride < size:
        raise RuntimeError(f"feature slot stride {stride} < {size}")
    return stride


class _FeatureSlots:
    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        num_tables: int,
        *,
        owner: bool,
    ) -> None:
        if num_tables <= 0:
            raise ValueError(f"num_tables must be > 0, got {num_tables}")
        stride = _feature_slot_stride()
        expected = stride * num_tables
        if shm.size < expected:
            raise RuntimeError(f"feature shm size {shm.size} < {expected}")
        self._shm = shm
        self._owner = owner
        self._closed = False
        self._num_tables = num_tables
        buf = np.ndarray((expected,), dtype=np.uint8, buffer=shm.buf)
        self._views = [_map_feature_slot(buf, table_id, stride) for table_id in range(num_tables)]
        prefix = _infer_prefix_bytes()
        prefix_n = prefix // 4
        self._prefix_2d = np.ndarray(
            (num_tables, prefix_n),
            dtype=np.float32,
            buffer=shm.buf,
            strides=(stride, 4),
        )
        if self._prefix_2d.shape != (num_tables, prefix_n):
            raise RuntimeError(
                f"prefix_2d {self._prefix_2d.shape} != ({num_tables}, {prefix_n})"
            )
        self._infer_prefix = [self._prefix_2d[table_id] for table_id in range(num_tables)]
        for table_id, view in enumerate(self._infer_prefix):
            if view.shape != (prefix_n,):
                raise RuntimeError(
                    f"table {table_id} infer prefix shape {view.shape} != ({prefix_n},)"
                )
            if not view.flags.c_contiguous:
                raise RuntimeError(f"table {table_id} infer prefix is not contiguous")

    @classmethod
    def create(cls, num_tables: int) -> "_FeatureSlots":
        stride = _feature_slot_stride()
        shm = shared_memory.SharedMemory(create=True, size=stride * num_tables)
        return cls(shm, num_tables, owner=True)

    @classmethod
    def attach(cls, name: str, num_tables: int) -> "_FeatureSlots":
        if not name:
            raise ValueError("feature shm name is empty")
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, num_tables, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def fill(self, table_id: int, arrays: dict[str, object]) -> None:
        views = self._table_views(table_id)
        for name, shape in _feature_fields():
            src = arrays[name]
            if not isinstance(src, np.ndarray):
                raise TypeError(f"{name} must be ndarray, got {type(src).__name__}")
            if src.shape != shape or src.dtype != np.float32:
                raise ValueError(
                    f"{name} has shape/dtype {src.shape}/{src.dtype}, expected {shape}/float32"
                )
            if not src.flags.c_contiguous:
                src = np.ascontiguousarray(src, dtype=np.float32)
            np.copyto(views[name], src, casting="no")

    def field_view(self, table_id: int, name: str) -> np.ndarray:
        views = self._table_views(table_id)
        if name not in views:
            raise KeyError(f"feature field {name} is missing")
        return views[name]

    def infer_prefix_view(self, table_id: int) -> np.ndarray:
        if table_id < 0 or table_id >= self._num_tables:
            raise ValueError(f"table_id out of range: {table_id}")
        return self._infer_prefix[table_id]

    def copy_infer_prefixes(self, table_ids: list[int], dst: np.ndarray) -> None:
        n_ids = len(table_ids)
        prefix_n = int(self._prefix_2d.shape[1])
        if dst.shape != (n_ids, prefix_n):
            raise RuntimeError(
                f"prefix dest {tuple(dst.shape)} != ({n_ids}, {prefix_n})"
            )
        if dst.dtype != np.float32:
            raise TypeError(f"prefix dest dtype {dst.dtype} != float32")
        ids = np.asarray(table_ids, dtype=np.intp)
        if ids.shape != (n_ids,):
            raise RuntimeError(f"table id index shape {ids.shape} != ({n_ids},)")
        if np.any(ids < 0) or np.any(ids >= self._num_tables):
            raise RuntimeError(f"table ids out of range: {ids.tolist()}")
        np.copyto(dst, self._prefix_2d[ids], casting="no")

    def copy_table(self, table_id: int) -> dict[str, np.ndarray]:
        views = self._table_views(table_id)
        copied: dict[str, np.ndarray] = {}
        for name, shape in _feature_fields():
            value = np.array(views[name], dtype=np.float32, copy=True)
            if value.shape != shape:
                raise RuntimeError(f"copied {name} shape {value.shape} != {shape}")
            copied[name] = value
        return copied

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        if self._owner:
            self._shm.unlink()

    def _table_views(self, table_id: int) -> dict[str, np.ndarray]:
        if table_id < 0 or table_id >= self._num_tables:
            raise ValueError(f"table_id out of range: {table_id}")
        return self._views[table_id]


def _map_feature_slot(
    buf: np.ndarray,
    table_id: int,
    stride: int,
) -> dict[str, np.ndarray]:
    start = table_id * stride
    chunk = buf[start : start + stride]
    offset = 0
    views: dict[str, np.ndarray] = {}
    for name, shape in _feature_fields():
        n = 1
        for dim in shape:
            n *= dim
        nbytes = n * 4
        raw = chunk[offset : offset + nbytes]
        if raw.size != nbytes:
            raise RuntimeError(f"{name} slot bytes {raw.size} != {nbytes}")
        views[name] = raw.view(np.float32).reshape(shape)
        offset += nbytes
    return views


def _ready_dtype() -> np.dtype:
    dtype = np.dtype(
        [
            ("ready", np.int32),
            ("seat", np.int32),
            ("n_hand", np.int32),
            ("n_legal", np.int32),
            ("n_v3", np.int32),
            ("lead", np.int32),
            ("observe_sec", np.float64),
            ("rank_sec", np.float64),
            ("featurize_sec", np.float64),
            ("team_belief_sec", np.float64),
            ("cpu_sec", np.float64),
        ]
    )
    if dtype.itemsize <= 0:
        raise RuntimeError(f"ready record size must be > 0, got {dtype.itemsize}")
    return dtype


def _ready_layout(num_tables: int) -> tuple[int, int, int]:
    if num_tables <= 0:
        raise ValueError(f"num_tables must be > 0, got {num_tables}")
    rows = _ready_dtype().itemsize * num_tables
    ring_off = (rows + 7) // 8 * 8
    ring_bytes = num_tables * 4
    counters_off = (ring_off + ring_bytes + 7) // 8 * 8
    total = counters_off + 16
    if ring_off < rows or counters_off < ring_off + ring_bytes or total < counters_off + 16:
        raise RuntimeError(
            f"ready layout overlap rows={rows} ring_off={ring_off} "
            f"counters_off={counters_off} total={total}"
        )
    return total, ring_off, counters_off


class _ReadySlots:
    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        num_tables: int,
        lock: object,
        *,
        owner: bool,
    ) -> None:
        if num_tables <= 0:
            raise ValueError(f"num_tables must be > 0, got {num_tables}")
        if lock is None:
            raise ValueError("ready lock is required")
        total, ring_off, counters_off = _ready_layout(num_tables)
        if shm.size < total:
            raise RuntimeError(f"ready shm size {shm.size} < {total}")
        dtype = _ready_dtype()
        self._shm = shm
        self._owner = owner
        self._closed = False
        self._num_tables = num_tables
        self._lock = lock
        self._rows = np.ndarray((num_tables,), dtype=dtype, buffer=shm.buf)
        self._ring = np.ndarray(
            (num_tables,), dtype=np.int32, buffer=shm.buf, offset=ring_off
        )
        self._head = np.ndarray((1,), dtype=np.int64, buffer=shm.buf, offset=counters_off)
        self._tail = np.ndarray(
            (1,), dtype=np.int64, buffer=shm.buf, offset=counters_off + 8
        )
        if owner:
            self._rows[:] = 0
            self._ring[:] = -1
            self._head[0] = 0
            self._tail[0] = 0

    @classmethod
    def create(cls, num_tables: int, lock: object) -> "_ReadySlots":
        total, _, _ = _ready_layout(num_tables)
        shm = shared_memory.SharedMemory(create=True, size=total)
        return cls(shm, num_tables, lock, owner=True)

    @classmethod
    def attach(cls, name: str, num_tables: int, lock: object) -> "_ReadySlots":
        if not name:
            raise ValueError("ready shm name is empty")
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, num_tables, lock, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def publish(
        self,
        table_id: int,
        *,
        seat: int,
        n_hand: int,
        n_legal: int,
        n_v3: int,
        lead: bool,
        observe_sec: float,
        rank_sec: float,
        featurize_sec: float,
        team_belief_sec: float,
        cpu_sec: float,
    ) -> None:
        rec = self._row(table_id)
        if int(rec["ready"]) != 0:
            raise RuntimeError(f"table {table_id} published while still marked ready")
        rec["seat"] = seat
        rec["n_hand"] = n_hand
        rec["n_legal"] = n_legal
        rec["n_v3"] = n_v3
        rec["lead"] = int(lead)
        rec["observe_sec"] = observe_sec
        rec["rank_sec"] = rank_sec
        rec["featurize_sec"] = featurize_sec
        rec["team_belief_sec"] = team_belief_sec
        rec["cpu_sec"] = cpu_sec
        rec["ready"] = 1
        with self._lock:
            tail = int(self._tail[0])
            used = tail - int(self._head[0])
            if used < 0 or used >= self._num_tables:
                raise RuntimeError(
                    f"ready ring used {used} before push table={table_id}"
                )
            self._ring[tail % self._num_tables] = table_id
            self._tail[0] = tail + 1

    def collect_ready(self, limit: int) -> list[dict[str, object]]:
        if limit <= 0:
            raise ValueError(f"ready limit must be > 0, got {limit}")
        if limit > self._num_tables:
            raise ValueError(f"ready limit {limit} > num_tables {self._num_tables}")
        with self._lock:
            head = int(self._head[0])
            tail = int(self._tail[0])
            used = tail - head
            if used < 0 or used > self._num_tables:
                raise RuntimeError(f"ready ring used {used}")
            take = min(limit, used)
            table_ids = [
                int(self._ring[(head + index) % self._num_tables])
                for index in range(take)
            ]
            self._head[0] = head + take
        packets: list[dict[str, object]] = []
        for table_id in table_ids:
            if table_id < 0 or table_id >= self._num_tables:
                raise RuntimeError(f"ready ring table_id {table_id} is out of range")
            rec = self._rows[table_id]
            if int(rec["ready"]) != 1:
                raise RuntimeError(f"ring table {table_id} is not marked ready")
            rec["ready"] = 0
            packets.append(
                {
                    "table_id": table_id,
                    "seat": int(rec["seat"]),
                    "n_hand": int(rec["n_hand"]),
                    "n_legal": int(rec["n_legal"]),
                    "n_v3": int(rec["n_v3"]),
                    "lead": bool(rec["lead"]),
                    "observe_sec": float(rec["observe_sec"]),
                    "rank_sec": float(rec["rank_sec"]),
                    "featurize_sec": float(rec["featurize_sec"]),
                    "team_belief_sec": float(rec["team_belief_sec"]),
                    "cpu_sec": float(rec["cpu_sec"]),
                }
            )
        return packets

    def count_ready(self) -> int:
        with self._lock:
            used = int(self._tail[0]) - int(self._head[0])
        if used < 0 or used > self._num_tables:
            raise RuntimeError(f"ready ring used {used}")
        return used

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        if self._owner:
            self._shm.unlink()

    def _row(self, table_id: int) -> np.void:
        if table_id < 0 or table_id >= self._num_tables:
            raise ValueError(f"table_id out of range: {table_id}")
        return self._rows[table_id]


def _table_loop(
    table_id: int,
    seed: int,
    shm_name: str,
    num_tables: int,
    ready_name: str,
    ready_lock: object,
    parent_wake: DupFd,
    action_codes,
    action_wake: DupFd,
    done_q: mp.Queue,
    error_q: mp.Queue,
) -> None:
    slots: _FeatureSlots | None = None
    ready_slots: _ReadySlots | None = None
    wake_fd: int | None = None
    parent_wake_fd: int | None = None
    try:
        _apply_worker_env()
        _pin_worker_threads()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        wake_fd = action_wake.detach()
        if wake_fd < 0:
            raise RuntimeError(f"table {table_id} eventfd is invalid: {wake_fd}")
        parent_wake_fd = parent_wake.detach()
        if parent_wake_fd < 0:
            raise RuntimeError(
                f"table {table_id} parent eventfd is invalid: {parent_wake_fd}"
            )
        slots = _FeatureSlots.attach(shm_name, num_tables)
        ready_slots = _ReadySlots.attach(ready_name, num_tables, ready_lock)
        ranker = StructuralCandidateRanker()
        table = _start_table()
        game_apply_sec = 0.0
        game_cpu_sec = 0.0
        pending_reset_sec = 0.0
        while True:
            cpu0 = time.process_time()
            t0 = time.perf_counter()
            job = _observation_job(table)
            t1 = time.perf_counter()
            ranked_result = _rank_and_label(job, ranker)
            slots.fill(
                table_id,
                {
                    "state": ranked_result["state"],
                    "candidates": ranked_result["candidates"],
                    "mask": ranked_result["mask"],
                    "history": job["history"],
                    "team_belief_labels": ranked_result["team_belief_labels"],
                    "team_belief_mask": ranked_result["team_belief_mask"],
                },
            )
            n_hand = int(ranked_result["n_hand"])
            n_legal = int(ranked_result["n_legal"])
            n_v3 = int(ranked_result["n_v3"])
            if n_hand < 0 or n_legal <= 0 or n_v3 <= 0 or n_v3 > n_legal:
                raise RuntimeError(
                    f"table {table_id} rank counts invalid: "
                    f"hand={n_hand} legal={n_legal} v3={n_v3}"
                )
            ready_slots.publish(
                table_id,
                seat=int(job["seat"]),
                n_hand=n_hand,
                n_legal=n_legal,
                n_v3=n_v3,
                lead=bool(ranked_result["lead"]),
                observe_sec=t1 - t0,
                rank_sec=float(ranked_result["rank_sec"]),
                featurize_sec=float(ranked_result["featurize_sec"]),
                team_belief_sec=float(ranked_result["team_belief_sec"]),
                cpu_sec=time.process_time() - cpu0,
            )
            os.eventfd_write(parent_wake_fd, 1)
            woke = os.eventfd_read(wake_fd)
            action_code = int(action_codes[table_id])
            if action_code == _ACTION_STOP:
                return
            if woke > 1:
                action_code = int(action_codes[table_id])
                if action_code == _ACTION_STOP:
                    return
                raise RuntimeError(
                    f"table {table_id} eventfd count {woke} != 1 action={action_code}"
                )
            if woke != 1:
                raise RuntimeError(f"table {table_id} eventfd count {woke} != 1")
            if action_code == _ACTION_EMPTY:
                raise RuntimeError(f"table {table_id} woke with empty action")
            if action_code < 0:
                raise RuntimeError(
                    f"table {table_id} invalid action code {action_code}"
                )
            cpu1 = time.process_time()
            t3 = time.perf_counter()
            rewards = _apply_decision(table, job, ranked_result, action_code)
            t4 = time.perf_counter()
            game_apply_sec += t4 - t3
            game_cpu_sec += time.process_time() - cpu1
            if rewards is None:
                continue
            episode_steps = table.steps
            if episode_steps <= 0:
                raise RuntimeError(f"table {table_id} finished with no steps")
            done_q.put(
                {
                    "table_id": table_id,
                    "episode_steps": episode_steps,
                    "rewards": rewards,
                    "apply_sec": game_apply_sec,
                    "table_reset_sec": pending_reset_sec,
                    "cpu_sec": game_cpu_sec,
                }
            )
            game_apply_sec = 0.0
            game_cpu_sec = 0.0
            t5 = time.perf_counter()
            table = _start_table()
            ranker = StructuralCandidateRanker()
            pending_reset_sec = time.perf_counter() - t5
    except Exception as exc:
        error_q.put({"table_id": table_id, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if parent_wake_fd is not None:
            os.close(parent_wake_fd)
        if wake_fd is not None:
            os.close(wake_fd)
        if ready_slots is not None:
            ready_slots.close()
        if slots is not None:
            slots.close()


class PersistentTables:
    def __init__(
        self,
        *,
        ctx: mp.context.BaseContext,
        num_tables: int,
        infer_batch: int,
        seed: int,
        device: torch.device,
        infer_model: torch.nn.Module,
        infer_lock: threading.Lock,
    ) -> None:
        if num_tables <= 0:
            raise ValueError(f"num-tables must be > 0, got {num_tables}")
        if infer_batch <= 0:
            raise ValueError(f"infer-batch must be > 0, got {infer_batch}")
        if infer_batch > num_tables:
            raise ValueError(
                f"infer-batch {infer_batch} exceeds num-tables {num_tables}"
            )
        infer_param = next(infer_model.parameters())
        if (infer_param.device.type, infer_param.device.index) != (
            device.type,
            device.index,
        ):
            raise ValueError(
                f"infer model device {infer_param.device} != tables device {device}"
            )
        self._num_tables = num_tables
        self._infer_batch = infer_batch
        self._device = device
        self._infer_model = infer_model
        self._infer_lock = infer_lock
        self._graph_infer = _CompiledInfer(infer_model, device, infer_batch)
        self._ready_overflow_streak = 0
        self._slots = _FeatureSlots.create(num_tables)
        self._ready_lock = ctx.Lock()
        self._ready = _ReadySlots.create(num_tables, self._ready_lock)
        self._pending: list[list[dict[str, object]]] = [[] for _ in range(num_tables)]
        self._game_over_msgs = [0] * num_tables
        self._procs: list[mp.Process] = []
        self._action_codes = ctx.Array("i", [_ACTION_EMPTY] * num_tables, lock=False)
        self._action_code_np = np.frombuffer(self._action_codes, dtype=np.int32)
        if self._action_code_np.shape != (num_tables,):
            raise RuntimeError(
                f"action code view {self._action_code_np.shape} != ({num_tables},)"
            )
        self._action_wake_fds = [os.eventfd(0) for _ in range(num_tables)]
        self._parent_wake_fd = os.eventfd(0)
        self._wake_lock = threading.Lock()
        self._wake_errors: list[BaseException] = []
        self._wake_dispatch_sec = 0.0
        self._wake_q: Queue = Queue(maxsize=num_tables)
        self._wake_thread = threading.Thread(
            target=self._wake_loop,
            name="danks-action-wake",
            daemon=False,
        )
        self._wake_thread.start()
        try:
            self._done_q: mp.Queue = ctx.Queue()
            self._error_q: mp.Queue = ctx.Queue()
            for table_id in range(num_tables):
                proc = ctx.Process(
                    target=_table_loop,
                    args=(
                        table_id,
                        seed + table_id + 1,
                        self._slots.name,
                        num_tables,
                        self._ready.name,
                        self._ready_lock,
                        DupFd(self._parent_wake_fd),
                        self._action_codes,
                        DupFd(self._action_wake_fds[table_id]),
                        self._done_q,
                        self._error_q,
                    ),
                )
                proc.start()
                if proc.exitcode is not None:
                    raise RuntimeError(
                        f"table {table_id} exited immediately with {proc.exitcode}"
                    )
                if proc.pid is None or proc.pid <= 0:
                    raise RuntimeError(f"table {table_id} has no pid")
                self._procs.append(proc)
        except BaseException:
            self.close()
            raise

    def _ingest_ranked_packet(
        self,
        packet: object,
        *,
        accums: list[dict[str, float | int]],
        batch: list[dict[str, object]],
        seen: set[int],
        rank_calls: dict[str, float | int],
    ) -> None:
        if not isinstance(packet, dict):
            raise TypeError(f"ranked packet must be a dict, got {type(packet).__name__}")
        table_id = int(packet["table_id"])
        if table_id < 0 or table_id >= self._num_tables:
            raise ValueError(f"ranked table_id out of range: {table_id}")
        if table_id in seen:
            raise RuntimeError(f"table {table_id} appeared twice in infer batch")
        seat = int(packet["seat"])
        if seat not in range(4):
            raise ValueError(f"table {table_id} seat out of range: {seat}")
        observe_sec = float(packet["observe_sec"])
        rank_sec = float(packet["rank_sec"])
        featurize_sec = float(packet["featurize_sec"])
        team_belief_sec = float(packet["team_belief_sec"])
        cpu_sec = float(packet["cpu_sec"])
        if (
            observe_sec < 0.0
            or rank_sec < 0.0
            or featurize_sec < 0.0
            or team_belief_sec < 0.0
            or cpu_sec < 0.0
        ):
            raise ValueError(
                f"table {table_id} ranked timings are negative: "
                f"observe={observe_sec} rank={rank_sec} featurize={featurize_sec} "
                f"belief={team_belief_sec} cpu={cpu_sec}"
            )
        n_hand = int(packet["n_hand"])
        n_legal = int(packet["n_legal"])
        n_v3 = int(packet["n_v3"])
        lead = bool(packet["lead"])
        if n_hand < 0 or n_legal <= 0 or n_v3 <= 0 or n_v3 > n_legal:
            raise RuntimeError(
                f"table {table_id} rank counts invalid: "
                f"hand={n_hand} legal={n_legal} v3={n_v3}"
            )
        accums[table_id]["observe_sec"] += observe_sec
        accums[table_id]["rank_sec"] += rank_sec
        accums[table_id]["featurize_sec"] += featurize_sec
        accums[table_id]["team_belief_sec"] += team_belief_sec
        accums[table_id]["cpu_sec"] += cpu_sec
        accums[table_id]["observe_calls"] += 1
        rank_calls["n"] = int(rank_calls["n"]) + 1
        rank_calls["sum"] = float(rank_calls["sum"]) + rank_sec
        if rank_sec > float(rank_calls["max"]):
            rank_calls["max"] = rank_sec
        if rank_sec > _SLOW_RANK_SEC:
            print(
                f"slow_rank={rank_sec:.2f}s table={table_id} lead={int(lead)} "
                f"hand={n_hand} legal={n_legal} v3={n_v3}",
                flush=True,
            )
        seen.add(table_id)
        batch.append(packet)

    def _copy_running_rows(
        self, batch: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        copied_rows: list[dict[str, object]] = []
        for packet in batch:
            table_id = int(packet["table_id"])
            row = dict(self._slots.copy_table(table_id))
            row["seat"] = int(packet["seat"])
            copied_rows.append(row)
        return copied_rows

    def _coalesce_ready(
        self,
        packets: list[dict[str, object]],
        *,
        take_limit: int,
        stop_event: threading.Event | None,
    ) -> list[dict[str, object]]:
        coalesce_min = min(_INFER_COALESCE_MIN, take_limit)
        if coalesce_min <= 0:
            raise RuntimeError(f"coalesce min {coalesce_min} <= 0")
        if len(packets) >= coalesce_min or len(packets) >= take_limit:
            return packets
        deadline = time.perf_counter() + _INFER_COALESCE_SEC
        while len(packets) < coalesce_min and len(packets) < take_limit:
            if stop_event is not None and stop_event.is_set():
                raise _CollectStopped()
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            if not select.select([self._parent_wake_fd], [], [], remaining)[0]:
                break
            woke = os.eventfd_read(self._parent_wake_fd)
            if woke < 1:
                raise RuntimeError(f"parent eventfd count {woke} < 1")
            more = self._ready.collect_ready(take_limit - len(packets))
            if more:
                packets.extend(more)
        if len(packets) > take_limit:
            raise RuntimeError(
                f"coalesced ready {len(packets)} > take_limit {take_limit}"
            )
        return packets

    def _pull_ready_batch(
        self,
        *,
        accums: list[dict[str, float | int]],
        rank_calls: dict[str, float | int],
        stop_event: threading.Event | None,
        min_update_steps: int,
        buffer: list[dict[str, object]],
        block: bool,
    ) -> tuple[list[dict[str, object]], int, float, int]:
        batch: list[dict[str, object]] = []
        seen: set[int] = set()
        ready_nowait = 0
        wait_block_this = 0.0
        self._raise_table_errors()
        take_limit = min(self._infer_batch, self._num_tables)
        packets = self._ready.collect_ready(take_limit)
        if packets:
            ready_nowait = len(packets)
        elif block:
            t_block0 = time.perf_counter()
            while not packets:
                if stop_event is not None and stop_event.is_set():
                    raise _CollectStopped()
                self._raise_table_errors()
                self._drain_game_overs(accums, buffer)
                if len(buffer) >= min_update_steps:
                    break
                if not select.select([self._parent_wake_fd], [], [], 1.0)[0]:
                    self._assert_alive()
                    continue
                woke = os.eventfd_read(self._parent_wake_fd)
                if woke < 1:
                    raise RuntimeError(f"parent eventfd count {woke} < 1")
                packets = self._ready.collect_ready(take_limit)
            wait_block_this = time.perf_counter() - t_block0
        if packets:
            packets = self._coalesce_ready(
                packets,
                take_limit=take_limit,
                stop_event=stop_event,
            )
        for packet in packets:
            self._ingest_ranked_packet(
                packet,
                accums=accums,
                batch=batch,
                seen=seen,
                rank_calls=rank_calls,
            )
        q_left = self._ready.count_ready()
        if q_left < 0 or q_left > self._num_tables:
            raise RuntimeError(f"ready count {q_left} is out of range")
        self._note_ready_overflow(len(batch), q_left)
        return batch, q_left, wait_block_this, ready_nowait

    def _note_ready_overflow(self, batch_size: int, q_left: int) -> None:
        if batch_size < 0 or batch_size > self._infer_batch:
            raise RuntimeError(
                f"infer batch {batch_size} outside 0..{self._infer_batch}"
            )
        if q_left > self._infer_batch:
            self._ready_overflow_streak += 1
        else:
            self._ready_overflow_streak = 0
        drain_limit = (self._num_tables + self._infer_batch - 1) // self._infer_batch
        if self._ready_overflow_streak > drain_limit:
            raise RuntimeError(
                f"q_left stayed above infer-batch {self._infer_batch} for "
                f"{self._ready_overflow_streak} consecutive GPU calls "
                f"(q_left={q_left}, drain_limit={drain_limit})"
            )

    def _wake_loop(self) -> None:
        failed = False
        while True:
            job = self._wake_q.get()
            try:
                if job is None:
                    return
                if failed:
                    continue
                table_ids, action_codes = job
                if len(table_ids) != len(action_codes):
                    raise RuntimeError(
                        f"wake job size mismatch ids={len(table_ids)} "
                        f"actions={len(action_codes)}"
                    )
                t0 = time.perf_counter()
                codes = self._action_code_np
                for table_id, action_code in zip(table_ids, action_codes, strict=True):
                    if action_code < 0:
                        raise ValueError(
                            f"table {table_id} action {action_code} is negative"
                        )
                    codes[table_id] = action_code
                    os.eventfd_write(self._action_wake_fds[table_id], 1)
                elapsed = time.perf_counter() - t0
                with self._wake_lock:
                    self._wake_dispatch_sec += elapsed
            except BaseException as exc:
                with self._wake_lock:
                    self._wake_errors.append(exc)
                failed = True
            finally:
                self._wake_q.task_done()

    def _raise_wake_errors(self) -> None:
        with self._wake_lock:
            if self._wake_errors:
                raise self._wake_errors[0]
        if not self._wake_thread.is_alive():
            raise RuntimeError("action wake thread died")

    def _wake_dispatch_snapshot(self) -> float:
        self._raise_wake_errors()
        with self._wake_lock:
            dispatch_sec = self._wake_dispatch_sec
            self._wake_dispatch_sec = 0.0
        return dispatch_sec

    def _commit_and_wake(
        self,
        *,
        batch: list[dict[str, object]],
        copied_rows: list[dict[str, object]],
        actions: np.ndarray,
        logps: np.ndarray,
        values: np.ndarray,
    ) -> None:
        expected = (len(batch),)
        if actions.shape != expected or logps.shape != expected or values.shape != expected:
            raise ValueError(
                f"policy shape mismatch: action={actions.shape} logp={logps.shape} "
                f"value={values.shape}, expected {expected}"
            )
        if len(copied_rows) != len(batch):
            raise RuntimeError(
                f"copied rows {len(copied_rows)} != batch {len(batch)}"
            )
        table_ids: list[int] = []
        action_codes: list[int] = []
        for index, packet in enumerate(batch):
            table_id = int(packet["table_id"])
            action_code = int(actions[index])
            if action_code < 0:
                raise ValueError(f"table {table_id} action {action_code} is negative")
            row = copied_rows[index]
            row["action"] = action_code
            row["logp"] = float(logps[index])
            row["value"] = float(values[index])
            self._pending[table_id].append(row)
            table_ids.append(table_id)
            action_codes.append(action_code)
        self._raise_wake_errors()
        self._wake_q.put((table_ids, action_codes))

    def collect(
        self,
        *,
        min_update_steps: int,
        stop_event: threading.Event | None = None,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, float],
        int,
        dict[str, float | int],
    ]:
        self._assert_alive()
        if min_update_steps <= 0:
            raise ValueError(f"min_update_steps must be > 0, got {min_update_steps}")
        buffer: list[dict[str, object]] = []
        accums = [
            {
                "observe_sec": 0.0,
                "rank_sec": 0.0,
                "featurize_sec": 0.0,
                "team_belief_sec": 0.0,
                "apply_sec": 0.0,
                "table_reset_sec": 0.0,
                "cpu_sec": 0.0,
                "observe_calls": 0,
                "finished_episodes": 0,
                "steps": 0,
            }
            for _ in range(self._num_tables)
        ]
        batch_wait_sec = 0.0
        feature_copy_sec = 0.0
        h2d_sec = 0.0
        infer_sec = 0.0
        gpu_smi_sec = 0.0
        d2h_sec = 0.0
        dispatch_sec = 0.0
        infer_calls = 0
        ready_samples: list[float] = []
        wait_samples: list[float] = []
        size_samples: list[float] = []
        rank_calls: dict[str, float | int] = {"n": 0, "sum": 0.0, "max": 0.0}
        q_left_samples: list[float] = []
        loop0 = time.perf_counter()
        pull_kwargs = {
            "accums": accums,
            "rank_calls": rank_calls,
            "stop_event": stop_event,
            "min_update_steps": min_update_steps,
            "buffer": buffer,
        }
        with self._wake_lock:
            self._wake_dispatch_sec = 0.0
        try:
            while len(buffer) < min_update_steps:
                if stop_event is not None and stop_event.is_set():
                    raise _CollectStopped()
                self._raise_wake_errors()
                self._drain_game_overs(accums, buffer)
                if len(buffer) >= min_update_steps:
                    break
                t_wait0 = time.perf_counter()
                batch, q_left, wait_block_this, ready_nowait = self._pull_ready_batch(
                    block=True,
                    **pull_kwargs,
                )
                batch_wait_sec += time.perf_counter() - t_wait0
                while batch:
                    table_ids = [int(packet["table_id"]) for packet in batch]
                    t_h2d0 = time.perf_counter()
                    with self._infer_lock:
                        with torch.inference_mode():
                            self._graph_infer.submit(self._slots, table_ids)
                            t_inf0 = time.perf_counter()
                            infer_calls += 1
                            t_copy0 = time.perf_counter()
                            copied_rows = self._copy_running_rows(batch)
                            feature_copy_sec += time.perf_counter() - t_copy0
                            actions, logps, values = self._graph_infer.take_numpy()
                            t_inf1 = time.perf_counter()
                    h2d_sec += t_inf0 - t_h2d0
                    infer_sec += t_inf1 - t_inf0
                    if ready_nowait > self._num_tables:
                        raise RuntimeError(
                            f"ready_nowait {ready_nowait} > num_tables {self._num_tables}"
                        )
                    q_left_samples.append(float(q_left))
                    ready_samples.append(float(ready_nowait))
                    wait_samples.append(wait_block_this)
                    size_samples.append(float(len(batch)))
                    self._commit_and_wake(
                        batch=batch,
                        copied_rows=copied_rows,
                        actions=actions,
                        logps=logps,
                        values=values,
                    )
                    self._drain_game_overs(accums, buffer)
                    if len(buffer) >= min_update_steps:
                        break
                    t_wait1 = time.perf_counter()
                    batch, q_left, wait_block_this, ready_nowait = self._pull_ready_batch(
                        block=False,
                        **pull_kwargs,
                    )
                    batch_wait_sec += time.perf_counter() - t_wait1
        finally:
            dispatch_sec = self._wake_dispatch_snapshot()
        self._drain_game_overs(accums, buffer)
        if len(buffer) < min_update_steps:
            raise RuntimeError(
                f"update batch has {len(buffer)} steps, required >= {min_update_steps}"
            )
        finished_episodes = sum(int(accum["finished_episodes"]) for accum in accums)
        if infer_calls <= 0 and finished_episodes <= 0:
            raise RuntimeError("collect produced no infer batches and no finished episodes")
        accum_steps = sum(int(accum["steps"]) for accum in accums)
        if len(buffer) != accum_steps:
            raise RuntimeError(
                f"row buffer {len(buffer)} != table step sum {accum_steps}"
            )
        collect_wall = time.perf_counter() - loop0
        if collect_wall <= 0.0:
            raise RuntimeError(f"collect timing collapsed to {collect_wall}")
        table_perfs: list[dict[str, object]] = []
        for table_id, accum in enumerate(accums):
            pid = self._procs[table_id].pid
            if pid is None or pid <= 0:
                raise RuntimeError(f"table {table_id} has no pid")
            cpu_sec = float(accum["cpu_sec"])
            table_perfs.append(
                {
                    "observe_sec": float(accum["observe_sec"]),
                    "rank_sec": float(accum["rank_sec"]),
                    "featurize_sec": float(accum["featurize_sec"]),
                    "team_belief_sec": float(accum["team_belief_sec"]),
                    "apply_sec": float(accum["apply_sec"]),
                    "table_reset_sec": float(accum["table_reset_sec"]),
                    "observe_calls": int(accum["observe_calls"]),
                    "finished_episodes": int(accum["finished_episodes"]),
                    "steps": int(accum["steps"]),
                    "rss_bytes": _rss_bytes(pid),
                    "cpu_util_pct": 100.0 * cpu_sec / collect_wall,
                }
            )
        if (
            len(ready_samples) != infer_calls
            or len(wait_samples) != infer_calls
            or len(size_samples) != infer_calls
            or len(q_left_samples) != infer_calls
        ):
            raise RuntimeError(
                f"pipeline samples ready={len(ready_samples)} wait={len(wait_samples)} "
                f"size={len(size_samples)} q_left={len(q_left_samples)} "
                f"infer_calls={infer_calls}"
            )
        rank_call_n = int(rank_calls["n"])
        observe_calls = sum(int(accum["observe_calls"]) for accum in accums)
        if infer_calls == 0:
            if rank_call_n != 0 or observe_calls != 0:
                raise RuntimeError(
                    f"leftover collect must have 0 rank/observe, "
                    f"rank_call_n={rank_call_n} observe_calls={observe_calls}"
                )
            ready_min = ready_mean = ready_max = 0.0
            wait_min = wait_mean = wait_max = 0.0
            size_min = size_mean = size_max = 0.0
            q_left_min = q_left_mean = q_left_max = 0.0
        else:
            if rank_call_n <= 0:
                raise RuntimeError(f"collect ingested no rank calls, n={rank_call_n}")
            if rank_call_n != observe_calls:
                raise RuntimeError(
                    f"rank_call_n {rank_call_n} != observe_calls {observe_calls}"
                )
            ready_min, ready_mean, ready_max = _min_mean_max(ready_samples, name="ready_nowait")
            wait_min, wait_mean, wait_max = _min_mean_max(wait_samples, name="wait_block")
            size_min, size_mean, size_max = _min_mean_max(size_samples, name="batch_size")
            q_left_min, q_left_mean, q_left_max = _min_mean_max(q_left_samples, name="q_left")
        batches_ready32 = 0
        for value in wait_samples:
            if value == 0.0:
                batches_ready32 += 1
        wait_sum = 0.0
        for value in wait_samples:
            wait_sum += value
        pending_steps = 0
        for rows in self._pending:
            pending_steps += len(rows)
        pipeline: dict[str, float | int] = {
            "batches_ready32": batches_ready32,
            "ready_nowait_min": ready_min,
            "ready_nowait_mean": ready_mean,
            "ready_nowait_max": ready_max,
            "wait_block_min_s": wait_min,
            "wait_block_mean_s": wait_mean,
            "wait_block_max_s": wait_max,
            "wait_block_sum_s": wait_sum,
            "pending_steps": pending_steps,
            "batch_size_min": size_min,
            "batch_size_mean": size_mean,
            "batch_size_max": size_max,
            "rank_call_n": rank_call_n,
            "rank_call_max": float(rank_calls["max"]),
            "q_left_mean": q_left_mean,
            "q_left_max": q_left_max,
        }
        return (
            buffer,
            table_perfs,
            {
                "batch_wait_sec": batch_wait_sec,
                "feature_copy_sec": feature_copy_sec,
                "h2d_sec": h2d_sec,
                "infer_sec": infer_sec,
                "gpu_smi_sec": gpu_smi_sec,
                "d2h_sec": d2h_sec,
                "dispatch_sec": dispatch_sec,
            },
            infer_calls,
            pipeline,
        )

    def sync_infer_from(self, train_model: torch.nn.Module) -> None:
        eager = _eager_module(train_model)
        src_params = list(eager.parameters())
        dst_params = list(self._infer_model.parameters())
        if len(src_params) != len(dst_params):
            raise RuntimeError(
                f"infer sync param count {len(dst_params)} != train {len(src_params)}"
            )
        src_bufs = list(eager.buffers())
        dst_bufs = list(self._infer_model.buffers())
        if len(src_bufs) != len(dst_bufs):
            raise RuntimeError(
                f"infer sync buffer count {len(dst_bufs)} != train {len(src_bufs)}"
            )
        with self._infer_lock:
            for dst, src in zip(dst_params, src_params, strict=True):
                if dst.shape != src.shape:
                    raise RuntimeError(
                        f"infer sync param shape {tuple(dst.shape)} != {tuple(src.shape)}"
                    )
                dst.data.copy_(src.data)
            for dst, src in zip(dst_bufs, src_bufs, strict=True):
                if dst.shape != src.shape:
                    raise RuntimeError(
                        f"infer sync buffer shape {tuple(dst.shape)} != {tuple(src.shape)}"
                    )
                dst.data.copy_(src.data)
            self._infer_model.eval()
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            elif self._device.type == "npu":
                torch.npu.synchronize(self._device)

    def sync_infer_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if not weights:
            raise RuntimeError("infer sync weights are empty")
        with self._infer_lock:
            result = self._infer_model.load_state_dict(weights, strict=True)
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    f"infer weight sync mismatch missing={missing} unexpected={unexpected}"
                )
            self._infer_model.eval()
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            elif self._device.type == "npu":
                torch.npu.synchronize(self._device)

    def close(self) -> None:
        parent_wake_fd = getattr(self, "_parent_wake_fd", None)
        if parent_wake_fd is not None:
            os.eventfd_write(parent_wake_fd, 1)
        wake_q = getattr(self, "_wake_q", None)
        wake_thread = getattr(self, "_wake_thread", None)
        if wake_q is not None and wake_thread is not None and wake_thread.is_alive():
            wake_q.join()
            wake_q.put(None)
            wake_thread.join(timeout=15.0)
            if wake_thread.is_alive():
                raise RuntimeError("action wake thread did not stop")
        done_q = getattr(self, "_done_q", None)
        if done_q is not None:
            while True:
                try:
                    done_q.get_nowait()
                except Empty:
                    break
        action_codes = getattr(self, "_action_codes", None)
        wake_fds = getattr(self, "_action_wake_fds", [])
        if action_codes is not None:
            for table_id in range(len(wake_fds)):
                action_codes[table_id] = _ACTION_STOP
            for wake_fd in wake_fds:
                os.eventfd_write(wake_fd, 1)
        deadline = time.monotonic() + 15.0
        for proc in getattr(self, "_procs", []):
            remain = deadline - time.monotonic()
            if remain > 0:
                proc.join(timeout=remain)
        for proc in getattr(self, "_procs", []):
            if proc.is_alive():
                proc.terminate()
        for table_id, proc in enumerate(getattr(self, "_procs", [])):
            proc.join(timeout=5)
            if proc.is_alive():
                raise RuntimeError(f"table {table_id} did not stop")
        for wake_fd in getattr(self, "_action_wake_fds", []):
            os.close(wake_fd)
        self._action_wake_fds = []
        parent_wake_fd = getattr(self, "_parent_wake_fd", None)
        if parent_wake_fd is not None:
            os.close(parent_wake_fd)
            self._parent_wake_fd = None
        ready = getattr(self, "_ready", None)
        if ready is not None:
            ready.close()
        slots = getattr(self, "_slots", None)
        if slots is not None:
            slots.close()

    def _record_game_over(
        self,
        accums: list[dict[str, float | int]],
        payload: object,
        buffer: list[dict[str, object]],
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"game-over payload must be a dict, got {type(payload).__name__}")
        table_id = int(payload["table_id"])
        if table_id < 0 or table_id >= self._num_tables:
            raise ValueError(f"game-over table_id out of range: {table_id}")
        apply_sec = float(payload["apply_sec"])
        reset_sec = float(payload["table_reset_sec"])
        cpu_sec = float(payload["cpu_sec"])
        episode_steps = int(payload["episode_steps"])
        if apply_sec < 0.0 or reset_sec < 0.0 or cpu_sec < 0.0:
            raise ValueError(
                f"table {table_id} game timings are negative: "
                f"play={apply_sec} reset={reset_sec} cpu={cpu_sec}"
            )
        if episode_steps <= 0:
            raise ValueError(f"table {table_id} episode_steps={episode_steps} is not positive")
        if "rewards" not in payload:
            raise KeyError(f"table {table_id} finished without rewards")
        rewards_raw = payload["rewards"]
        if not isinstance(rewards_raw, (list, tuple)) or len(rewards_raw) != 4:
            raise ValueError(f"table {table_id} rewards must be 4 values, got {rewards_raw!r}")
        rewards = tuple(int(value) for value in rewards_raw)
        accums[table_id]["apply_sec"] += apply_sec
        accums[table_id]["table_reset_sec"] += reset_sec
        accums[table_id]["cpu_sec"] += cpu_sec
        self._game_over_msgs[table_id] += 1
        pending = self._pending[table_id]
        if len(pending) < episode_steps:
            saved = [len(rows) for rows in self._pending]
            raise RuntimeError(
                "game-end step count mismatch: "
                f"table={table_id} saved_steps={len(pending)} episode_steps={episode_steps} "
                f"infer_packets={int(accums[table_id]['observe_calls'])} "
                f"game_over_msgs={self._game_over_msgs[table_id]} "
                f"buffer_steps={len(buffer)} "
                f"finished_episodes={sum(int(item['finished_episodes']) for item in accums)} "
                f"max_saved_steps={max(saved)} "
                f"tables_with_saved={sum(1 for n in saved if n > 0)}"
            )
        buffer.extend(_assign_gae(pending[:episode_steps], rewards))
        self._pending[table_id] = pending[episode_steps:]
        accums[table_id]["finished_episodes"] += 1
        accums[table_id]["steps"] += episode_steps

    def _drain_game_overs(
        self,
        accums: list[dict[str, float | int]],
        buffer: list[dict[str, object]],
    ) -> None:
        while True:
            try:
                payload = self._done_q.get_nowait()
            except Empty:
                return
            self._record_game_over(accums, payload, buffer)

    def _assert_alive(self) -> None:
        for table_id, proc in enumerate(self._procs):
            if not proc.is_alive():
                self._raise_table_errors()
                raise RuntimeError(
                    f"table {table_id} exited with code {proc.exitcode}"
                )

    def _raise_table_errors(self) -> None:
        try:
            message = self._error_q.get_nowait()
        except Empty:
            return
        if not isinstance(message, dict):
            raise TypeError(f"table error must be a dict, got {type(message).__name__}")
        raise RuntimeError(f"table {message.get('table_id')} failed: {message.get('error')}")


_STACK_SPECS: tuple[tuple[str, np.dtype, torch.dtype], ...] = (
    ("state", np.dtype(np.float32), torch.float32),
    ("candidates", np.dtype(np.float32), torch.float32),
    ("mask", np.dtype(np.float32), torch.float32),
    ("history", np.dtype(np.float32), torch.float32),
    ("action", np.dtype(np.int64), torch.long),
    ("logp", np.dtype(np.float32), torch.float32),
    ("advantage", np.dtype(np.float32), torch.float32),
    ("returns", np.dtype(np.float32), torch.float32),
    ("team_belief_labels", np.dtype(np.float32), torch.float32),
    ("team_belief_mask", np.dtype(np.float32), torch.float32),
    ("value", np.dtype(np.float32), torch.float32),
)
def _stack_rows_cpu(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("update batch is empty")
    n_rows = len(rows)
    stacked: dict[str, np.ndarray] = {}
    for key, np_dtype, _ in _STACK_SPECS:
        if key in (
            "state",
            "candidates",
            "mask",
            "history",
            "team_belief_labels",
            "team_belief_mask",
        ):
            value = np.ascontiguousarray(
                np.stack([row[key] for row in rows]),
                dtype=np_dtype,
            )
        else:
            value = np.ascontiguousarray(
                [row[key] for row in rows],
                dtype=np_dtype,
            )
        if value.shape[0] != n_rows:
            raise RuntimeError(
                f"{key} rows {value.shape[0]} != stacked rows {n_rows}"
            )
        if value.dtype != np_dtype:
            raise RuntimeError(f"{key} dtype {value.dtype} != {np_dtype}")
        stacked[key] = value
    return stacked


def _cpu_batch_to_device(
    cpu_batch: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    n_rows: int | None = None
    for key, np_dtype, torch_dtype in _STACK_SPECS:
        if key == "value":
            continue
        src = cpu_batch[key]
        if not isinstance(src, np.ndarray):
            raise TypeError(f"{key} must be ndarray, got {type(src).__name__}")
        if src.dtype != np_dtype:
            raise TypeError(f"{key} dtype {src.dtype} != {np_dtype}")
        if n_rows is None:
            n_rows = int(src.shape[0])
        elif src.shape[0] != n_rows:
            raise RuntimeError(f"{key} rows {src.shape[0]} != state rows {n_rows}")
        cpu_tensor = torch.from_numpy(src)
        if device.type == "cuda":
            if not cpu_tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()
            tensors[key] = cpu_tensor.to(device, dtype=torch_dtype, non_blocking=True)
        else:
            tensors[key] = cpu_tensor.to(device, dtype=torch_dtype)
    if n_rows is None:
        raise RuntimeError("cpu batch is empty")
    return tensors


def _split_update_minibatches(batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    if "state" not in batch:
        raise KeyError("update batch missing state")
    n_rows = int(batch["state"].shape[0])
    if n_rows < _OFFICIAL_MIN_STEPS:
        raise RuntimeError(
            f"update batch has {n_rows} steps, required >= {_OFFICIAL_MIN_STEPS}"
        )
    for key, value in batch.items():
        if value.shape[0] != n_rows:
            raise RuntimeError(
                f"{key} rows {value.shape[0]} != state rows {n_rows}"
            )
    return [batch]


def _mean_train_info(infos: list[dict[str, float]]) -> dict[str, float]:
    if not infos:
        raise RuntimeError("expected at least 1 minibatch info, got 0")
    keys = list(infos[0].keys())
    if not keys:
        raise RuntimeError("minibatch train info is empty")
    for index, info in enumerate(infos):
        if list(info.keys()) != keys:
            raise KeyError(f"minibatch {index} train info keys {list(info.keys())} != {keys}")
    averaged: dict[str, float] = {}
    for key in keys:
        total = 0.0
        for info in infos:
            total += float(info[key])
        averaged[key] = total / len(infos)
    averaged["minibatches"] = float(len(infos))
    return averaged


def _load_model(path: str, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a dict")
    model = build_selector_from_checkpoint(payload, device=device)
    if selector_model_type(model) != FEATURE_VERSION:
        raise ValueError(
            f"online loop requires {FEATURE_VERSION}, got {selector_model_type(model)}"
        )
    return model


def _model_config(model: torch.nn.Module) -> dict[str, int]:
    state_linear = model.state_encoder[0]
    candidate_linear = model.candidate_encoder[0]
    if not isinstance(state_linear, torch.nn.Linear):
        raise TypeError("state_encoder must start with Linear")
    if not isinstance(candidate_linear, torch.nn.Linear):
        raise TypeError("candidate_encoder must start with Linear")
    return {
        "hidden_dim": int(state_linear.out_features),
        "candidate_hidden_dim": int(candidate_linear.out_features),
    }


def _eager_module(module: torch.nn.Module) -> torch.nn.Module:
    orig = getattr(module, "_orig_mod", None)
    if orig is None:
        return module
    if not isinstance(orig, torch.nn.Module):
        raise TypeError(f"_orig_mod is {type(orig).__name__}, not nn.Module")
    return orig


def _experiment_stamp() -> str:
    stamp = os.environ.get("DANKS_RUN_ID", "run").strip()
    if not stamp:
        raise RuntimeError("empty DANKS_RUN_ID")
    if any(ch in stamp for ch in "/\\"):
        raise RuntimeError(f"DANKS_RUN_ID has path chars: {stamp!r}")
    return stamp


def _resolve_output_dir(value: str) -> Path:
    path = Path(value)
    if path.suffix == ".pt":
        raise ValueError(
            f"--output must be an experiment directory, got checkpoint file {path}"
        )
    return path


def _checkpoint_path(
    output_dir: Path, stamp: str, update: int, batch_size: int
) -> Path:
    if not stamp:
        raise ValueError("experiment stamp is empty")
    del stamp  # public builds omit calendar stamps from filenames
    if update <= 0:
        raise ValueError(f"update must be > 0, got {update}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return output_dir / f"top10_selector_online_{update}-{batch_size}.pt"


def _save_model(path: Path, model: torch.nn.Module) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": cpu_state_dict(_eager_module(model)),
            "state_dim": STATE_DIM,
            "candidate_dim": CANDIDATE_DIM,
            "feature_version": FEATURE_VERSION,
            "history_protocol": HISTORY_PROTOCOL,
            "history_event_semantics": HISTORY_EVENT_SEMANTICS,
            "history_length": HISTORY_LENGTH,
            "history_event_dim": HISTORY_EVENT_DIM,
            "team_belief_protocol": TEAM_BELIEF_PROTOCOL,
            "candidate_capacity": TOPK,
            "action_support": STRUCTURED_TOPK,
            "model_type": selector_model_type(model),
            "model_config": _model_config(model),
            "kind": "top10_online",
        },
        tmp,
    )
    tmp.replace(path)


def _bind_device_arg(value: str) -> str:
    if value == "auto":
        if npu_available():
            return "npu:0"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if value in {"cuda", "npu"}:
        return f"{value}:0"
    return value


def _bind_device_pair(train_arg: str, infer_arg: str) -> tuple[torch.device, torch.device]:
    train_device = initialize_device(_bind_device_arg(train_arg))
    infer_device = initialize_device(_bind_device_arg(infer_arg))
    if (infer_device.type, infer_device.index) == (train_device.type, train_device.index):
        raise ValueError(
            f"infer-device {infer_device} must differ from --device {train_device}"
        )
    if infer_device.type != train_device.type:
        raise ValueError(
            f"infer-device type {infer_device.type} != train type {train_device.type}"
        )
    if infer_device.type == "cpu":
        raise ValueError("async sample/train requires accelerator devices")
    return train_device, infer_device


def _clone_infer_model(
    train_model: torch.nn.Module, infer_device: torch.device
) -> torch.nn.Module:
    infer_model = copy.deepcopy(train_model).to(infer_device)
    infer_model.eval()
    for param in infer_model.parameters():
        param.requires_grad_(False)
    enabled = 0
    for module in infer_model.modules():
        if type(module).__name__ == "CardTokenEncoder":
            module.static_batch = True
            enabled += 1
    if enabled <= 0:
        raise RuntimeError("infer clone has no CardTokenEncoder for static CUDA graphs")
    return infer_model


def _bind_thread_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)


def _resolve_table_layout(args: argparse.Namespace) -> tuple[int, int]:
    num_tables = int(args.num_tables)
    infer_batch = int(args.infer_batch)
    if num_tables <= 0:
        raise ValueError(f"num-tables must be > 0, got {num_tables}")
    if infer_batch <= 0:
        raise ValueError(f"infer-batch must be > 0, got {infer_batch}")
    if infer_batch > num_tables:
        raise ValueError(
            f"infer-batch {infer_batch} exceeds num-tables {num_tables}"
        )
    return num_tables, infer_batch


def main() -> None:
    from DanKS.training.ray_online import run_online

    run_online(parse_args())


def _main_local_mp() -> None:
    args = parse_args()
    torch.set_num_threads(32)
    if torch.get_num_threads() != 32:
        raise RuntimeError(f"local train torch threads={torch.get_num_threads()} != 32")
    if args.updates <= 0:
        raise ValueError(f"updates must be > 0, got {args.updates}")
    num_tables, infer_batch = _resolve_table_layout(args)
    if args.min_update_steps < _OFFICIAL_MIN_STEPS:
        raise ValueError(
            f"min-update-steps must be >= {_OFFICIAL_MIN_STEPS}, "
            f"got {args.min_update_steps}"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_device, infer_device = _bind_device_pair(args.device, args.infer_device)
    if args.init_checkpoint:
        model = _load_model(args.init_checkpoint, train_device)
    else:
        model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM).to(
            train_device
        )
    infer_model = _clone_infer_model(model, infer_device)
    torch.set_float32_matmul_precision("high")
    infer_lock = threading.Lock()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model = torch.compile(model)
    agent = Top10PPOAgent(
        model,
        optimizer,
        PPOConfig(
            clip_ratio=0.2,
            train_iters=2,
            amp=True,
            amp_dtype=torch.bfloat16,
        ),
    )
    output_dir = _resolve_output_dir(args.output)
    exp_stamp = _experiment_stamp()
    previous_ckpt: Path | None = None
    print(f"experiment={exp_stamp} output_dir={output_dir}", flush=True)
    ctx = mp.get_context("spawn")
    if ctx.get_start_method() != "spawn":
        raise RuntimeError(f"start method must be spawn, got {ctx.get_start_method()}")
    tables = PersistentTables(
        ctx=ctx,
        num_tables=num_tables,
        infer_batch=infer_batch,
        seed=args.seed,
        device=infer_device,
        infer_model=infer_model,
        infer_lock=infer_lock,
    )
    batch_q: Queue = Queue(maxsize=2)
    sampler_errors: list[BaseException] = []
    stop_sampling = threading.Event()

    def _sample_loop() -> None:
        _bind_thread_device(infer_device)
        try:
            while not stop_sampling.is_set():
                collect_started = time.perf_counter()
                payload = tables.collect(
                    min_update_steps=args.min_update_steps,
                    stop_event=stop_sampling,
                )
                collect_sec = time.perf_counter() - collect_started
                if collect_sec <= 0.0:
                    raise RuntimeError(f"collect_sec must be > 0, got {collect_sec}")
                rows, table_perfs, collect_times, infer_calls, pipeline = payload
                batch_q.put(
                    (
                        (rows, table_perfs, collect_times, infer_calls, pipeline),
                        collect_sec,
                    )
                )
        except _CollectStopped:
            return
        except BaseException as exc:
            if stop_sampling.is_set():
                return
            sampler_errors.append(exc)
            raise

    sampler = threading.Thread(target=_sample_loop, name="online-sample", daemon=False)
    sampler.start()

    def _take_sample() -> tuple[
        tuple[
            dict[str, np.ndarray],
            list[dict[str, object]],
            dict[str, float],
            int,
            dict[str, float | int],
        ],
        float,
    ]:
        while True:
            if sampler_errors:
                raise sampler_errors[0]
            if not sampler.is_alive() and batch_q.empty():
                raise RuntimeError("sampler thread died")
            try:
                return batch_q.get(timeout=1.0)
            except Empty:
                continue

    try:
        for update in range(args.updates):
            cpu0 = time.process_time()
            started = time.perf_counter()
            (
                rows,
                table_perfs,
                collect_times,
                infer_calls,
                pipeline,
            ), collect_sec = _take_sample()
            if not isinstance(rows, list):
                raise TypeError(f"rows is {type(rows).__name__}")
            t_stack0 = time.perf_counter()
            cpu_batch = _stack_rows_cpu(rows)
            if int(cpu_batch["state"].shape[0]) != len(rows):
                raise RuntimeError(
                    f"stacked rows {cpu_batch['state'].shape[0]} != {len(rows)}"
                )
            step_count = int(cpu_batch["state"].shape[0])
            if step_count < args.min_update_steps:
                raise RuntimeError(
                    f"update batch has {step_count} steps, required >= {args.min_update_steps}"
                )
            _bind_thread_device(train_device)
            model.train()
            batch = _cpu_batch_to_device(cpu_batch, train_device)
            minibatches = _split_update_minibatches(batch)
            stack_sec = time.perf_counter() - t_stack0
            t_upd0 = time.perf_counter()
            minibatch_infos: list[dict[str, float]] = []
            for minibatch in minibatches:
                minibatch_infos.append(agent.update(minibatch))
            if train_device.type == "cuda":
                torch.cuda.synchronize(train_device)
            elif train_device.type == "npu":
                torch.npu.synchronize(train_device)
            update_sec = time.perf_counter() - t_upd0
            tables.sync_infer_from(model)
            info = _mean_train_info(minibatch_infos)
            values = np.asarray(cpu_batch["value"], dtype=np.float64)
            returns = np.asarray(cpu_batch["returns"], dtype=np.float64)
            if values.shape != (step_count,) or returns.shape != (step_count,):
                raise RuntimeError(
                    f"value/returns shape {values.shape}/{returns.shape} "
                    f"!= ({step_count},)"
                )
            info["explained_variance"] = _explained_variance(returns, values)
            del cpu_batch, batch, minibatches
            save_sec = 0.0
            if (update + 1) % _SAVE_EVERY_UPDATES == 0:
                t_save0 = time.perf_counter()
                ckpt_path = _checkpoint_path(
                    output_dir, exp_stamp, update + 1, args.min_update_steps
                )
                _save_model(ckpt_path, model)
                if previous_ckpt is not None:
                    if previous_ckpt == ckpt_path:
                        raise RuntimeError(f"checkpoint path reused: {ckpt_path}")
                    previous_ckpt.unlink()
                previous_ckpt = ckpt_path
                save_sec = time.perf_counter() - t_save0
            finished = time.perf_counter()
            total_sec = finished - started
            parent_cpu_sec = time.process_time() - cpu0
            if total_sec <= 0:
                raise RuntimeError(f"total_sec must be > 0, got {total_sec}")
            _print_perf_report(
                update=update + 1,
                updates=args.updates,
                steps=step_count,
                num_tables=num_tables,
                infer_batch=infer_batch,
                infer_calls=infer_calls,
                device=train_device,
                collect_wall=collect_sec,
                total_wall=total_sec,
                parent_times={
                    **collect_times,
                    "stack_sec": stack_sec,
                    "update_sec": update_sec,
                    "save_sec": save_sec,
                    "cpu_util_pct": 100.0 * parent_cpu_sec / total_sec,
                },
                parent_rss=_rss_bytes(os.getpid()),
                parent_gpu=_device_memory_bytes(train_device),
                table_perfs=table_perfs,
                train_info=info,
                pipeline=pipeline,
            )
    finally:
        stop_sampling.set()
        parent_wake_fd = getattr(tables, "_parent_wake_fd", None)
        if parent_wake_fd is not None:
            os.eventfd_write(parent_wake_fd, 1)
        while True:
            try:
                batch_q.get_nowait()
            except Empty:
                break
        sampler.join(timeout=20.0)
        tables.close()
        if sampler.is_alive():
            raise RuntimeError("sampler thread did not stop")
        if sampler_errors:
            raise sampler_errors[0]


if __name__ == "__main__":
    main()
