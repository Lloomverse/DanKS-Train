"""Ray online loop: one GPU-node actor, local shm collect + PPO like single-machine."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import random
import resource
import socket
import shutil
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import ray
import torch
import torch.distributed as dist
from ray.exceptions import RayTaskError
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from DanKS.training.accelerator import cpu_state_dict
from DanKS.training.model import (
    EfficientTeamBeliefTop10Selector,
    build_selector_from_checkpoint,
    selector_model_type,
)
from DanKS.training.ppo import PPOConfig, Top10PPOAgent
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
    TOPK,
)
from DanKS.training.train_online import (
    PersistentTables,
    V3_ROOT,
    _OFFICIAL_MIN_STEPS,
    _SAVE_EVERY_UPDATES,
    _WORKER_ENV,
    _bind_thread_device,
    _checkpoint_path,
    _CollectStopped,
    _device_memory_bytes,
    _explained_variance,
    _experiment_stamp,
    _mean_train_info,
    _model_config,
    _print_perf_report,
    _resolve_output_dir,
    _resolve_table_layout,
    _rss_bytes,
    _cpu_batch_to_device,
    _split_update_minibatches,
    _stack_rows_cpu,
)


def _require_ray_driver() -> None:
    if not sys.executable:
        raise RuntimeError("sys.executable is empty")


def _ray_working_dir() -> str:
    danks_src = V3_ROOT / "DanKS"
    if not danks_src.is_dir():
        raise RuntimeError(f"DanKS package missing: {danks_src}")
    guandan_src = V3_ROOT.parent.parent / "guandan"
    if not guandan_src.is_dir():
        raise RuntimeError(f"guandan package missing: {guandan_src}")
    pack = Path(os.environ.get("DANKS_RAY_WD", str(Path.cwd() / ".danks_ray_wd")))
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    shutil.copytree(
        danks_src,
        pack / "DanKS",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt"),
    )
    shutil.copytree(
        guandan_src,
        pack / "guandan",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    required = (
        pack / "DanKS" / "__init__.py",
        pack / "DanKS" / "training" / "ray_online.py",
        pack / "DanKS" / "training" / "train_online.py",
        pack / "guandan" / "__init__.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"ray working_dir missing files: {missing}")
    return str(pack)


def _connect_cluster() -> None:
    address = os.environ.pop("RAY_ADDRESS", None)
    if not address:
        raise RuntimeError("RAY_ADDRESS is required for cluster jobs")
    runtime_env = {
        "working_dir": _ray_working_dir(),
        "excludes": ["**/*.pt", "**/__pycache__/**"],
        "env_vars": _native_worker_env(),
    }
    if ray.is_initialized():
        connected = ray.get_runtime_context().gcs_address
        raise RuntimeError(
            f"ray already initialized at {connected}, need a fresh connect to {address}"
        )
    ray.init(address=address, runtime_env=runtime_env)
    connected = ray.get_runtime_context().gcs_address
    if connected != address:
        ray.shutdown()
        raise RuntimeError(f"connected {connected}, need {address}")
    resources = ray.cluster_resources()
    available = ray.available_resources()
    gpus = float(resources.get("GPU", 0.0))
    avail_gpus = float(available.get("GPU", 0.0))
    if gpus < 4.0:
        raise RuntimeError(f"cluster GPU={gpus} < 4 on cluster")
    if avail_gpus < 4.0:
        raise RuntimeError(
            f"available GPU={avail_gpus} < 4 on cluster (cluster GPU={gpus})"
        )


def _select_gpu_node() -> tuple[str, int]:
    """Choose one live node that can host both two-GPU DDP bundles."""

    candidates: list[tuple[str, int, float]] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        gpus = float(resources.get("GPU", 0.0))
        cpus = int(float(resources.get("CPU", 0.0)))
        node_ip = str(node.get("NodeManagerAddress", ""))
        if gpus >= 4.0 and cpus >= 2 and node_ip:
            candidates.append((node_ip, cpus, gpus))
    if not candidates:
        raise RuntimeError(
            "no live node has >=4 GPUs and >=2 CPUs; "
            f"nodes={ray.nodes()!r}"
        )
    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    node_ip, node_cpus, _ = candidates[0]
    return node_ip, node_cpus


def _native_worker_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in _WORKER_ENV.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"WORKER_ENV {key!r}={value!r} must be str")
        if key.startswith("DANKS_"):
            env[key] = value
    if not env:
        raise RuntimeError("WORKER_ENV has no DANKS_ keys")
    return env


def _cpu_init_weights(init_checkpoint: str) -> dict[str, torch.Tensor]:
    if init_checkpoint:
        payload = torch.load(init_checkpoint, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("checkpoint must be a dict")
        model = build_selector_from_checkpoint(payload, device=torch.device("cpu"))
        if selector_model_type(model) != FEATURE_VERSION:
            raise ValueError(
                f"online loop requires {FEATURE_VERSION}, got {selector_model_type(model)}"
            )
        weights = cpu_state_dict(model)
    else:
        model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM)
        weights = cpu_state_dict(model)
    if not weights:
        raise RuntimeError("init weights are empty")
    return weights


def _require_visible_cuda(expected: int) -> list[torch.device]:
    if expected <= 0:
        raise ValueError(f"expected CUDA count must be > 0, got {expected}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU actor has no CUDA")
    count = int(torch.cuda.device_count())
    if count != expected:
        raise RuntimeError(f"GPU actor visible CUDA count {count} != {expected}")
    return [torch.device(f"cuda:{index}") for index in range(expected)]


def _raise_nofile(minimum: int) -> None:
    if minimum <= 0:
        raise ValueError(f"nofile minimum must be > 0, got {minimum}")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    cap = hard if hard != resource.RLIM_INFINITY else minimum
    if cap < minimum:
        raise RuntimeError(
            f"RLIMIT_NOFILE hard={hard} < {minimum}; "
            "raise nofile on the GPU Ray node before spawning tables"
        )
    want = minimum if hard == resource.RLIM_INFINITY else min(minimum, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


def _cloexec_open_fds() -> None:
    keep = {0, 1, 2}
    try:
        names = os.listdir("/proc/self/fd")
    except OSError as exc:
        raise RuntimeError("cannot list /proc/self/fd to set CLOEXEC") from exc
    for name in names:
        try:
            fd = int(name)
        except ValueError:
            continue
        if fd in keep:
            continue
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        except OSError:
            continue
        if flags & fcntl.FD_CLOEXEC:
            continue
        fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)


def _enable_static_infer(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    enabled = 0
    for module in model.modules():
        if type(module).__name__ == "CardTokenEncoder":
            module.static_batch = True
            enabled += 1
    if enabled <= 0:
        raise RuntimeError("infer model has no CardTokenEncoder")
    return model


def _force_physical_gpu_visibility(rank: int) -> list[str]:
    """Repair stale Ray GPU ids before torch initializes CUDA.

    Ray's node resource table on this cluster contains stale ids (5,6,7,8),
    while nvidia-smi exposes physical devices 0..7.  The resource reservation
    still protects four GPU slots; this only fixes the process-local mapping.
    """

    if rank not in (0, 1):
        raise ValueError(f"GPU visibility rank must be 0 or 1, got {rank}")
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    physical = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    usage = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free: list[str] = []
    for line in usage.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index, used, total = fields
        if index in physical and int(used) <= max(2048, int(total) // 20):
            free.append(index)
    if len(free) < 4:
        raise RuntimeError(
            "need at least 4 GPUs with <=5% memory used; "
            f"physical={physical!r} usage={usage.stdout!r}"
        )
    free.sort(key=int)
    selected = free[rank * 2:(rank + 1) * 2]
    if len(selected) != 2:
        raise RuntimeError(f"rank {rank} got invalid physical GPU slice {selected!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    return selected


def _eager_train(module: torch.nn.Module) -> torch.nn.Module:
    orig = getattr(module, "_orig_mod", None)
    if orig is None:
        return module
    if not isinstance(orig, torch.nn.Module):
        raise TypeError(f"_orig_mod is {type(orig).__name__}, not nn.Module")
    return orig


def _unwrap_task_error(exc: BaseException) -> BaseException:
    if isinstance(exc, RayTaskError):
        cause = getattr(exc, "cause", None)
        if isinstance(cause, BaseException):
            return cause
    return exc


@ray.remote(num_gpus=2, num_cpus=0)
class OnlineActor:
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        infer_batch: int,
        num_tables: int,
        seed: int,
        lr: float,
        min_update_steps: int,
        rank: int,
    ) -> None:
        if infer_batch <= 0 or infer_batch > num_tables:
            raise ValueError(f"infer_batch {infer_batch} outside 1..{num_tables}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if min_update_steps < _OFFICIAL_MIN_STEPS:
            raise ValueError(
                f"min-update-steps must be >= {_OFFICIAL_MIN_STEPS}, "
                f"got {min_update_steps}"
            )
        if rank not in (0, 1):
            raise ValueError(f"DDP rank must be 0 or 1, got {rank}")
        if num_tables % 2 or infer_batch % 2 or min_update_steps % 2:
            raise ValueError(
                "num-tables, infer-batch, and min-update-steps must be even "
                "for two learner groups"
            )
        num_tables //= 2
        infer_batch //= 2
        min_update_steps //= 2
        if min_update_steps < _OFFICIAL_MIN_STEPS:
            raise ValueError(
                f"per-group min-update-steps={min_update_steps} is below "
                f"official minimum {_OFFICIAL_MIN_STEPS}"
            )
        if not weights:
            raise RuntimeError("init weights are empty")
        physical_gpu_ids = _force_physical_gpu_visibility(rank)
        for key, value in _native_worker_env().items():
            os.environ[key] = value
        visible_cpus = sorted(os.sched_getaffinity(0))
        split = len(visible_cpus) // 2
        cpu_ids = visible_cpus[:split] if rank == 0 else visible_cpus[split:]
        if not cpu_ids:
            raise RuntimeError(f"rank {rank} received no CPUs from {visible_cpus}")
        os.sched_setaffinity(0, set(cpu_ids))
        nthreads = min(32, len(cpu_ids))
        torch.set_num_threads(nthreads)
        got_threads = torch.get_num_threads()
        if got_threads != nthreads:
            raise RuntimeError(
                f"train actor torch threads={got_threads}, want {nthreads}"
            )
        torch.set_float32_matmul_precision("high")
        train_device, infer_device = _require_visible_cuda(2)
        if (infer_device.type, infer_device.index) == (train_device.type, train_device.index):
            raise RuntimeError(f"infer-device {infer_device} equals train {train_device}")
        train_model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM).to(train_device)
        result = train_model.load_state_dict(weights, strict=True)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"train load mismatch missing={missing} unexpected={unexpected}"
            )
        infer_model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM).to(infer_device)
        infer_result = infer_model.load_state_dict(weights, strict=True)
        infer_missing = list(infer_result.missing_keys)
        infer_unexpected = list(infer_result.unexpected_keys)
        if infer_missing or infer_unexpected:
            raise RuntimeError(
                f"infer load mismatch missing={infer_missing} unexpected={infer_unexpected}"
            )
        infer_model = _enable_static_infer(infer_model)
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=lr)
        # DDP already installs parameter hooks.  Combining it with a compiled
        # module here causes a fresh dynamic-shape backward compile on nearly
        # every rollout batch; eager DDP is materially faster for this loop.
        compiled = train_model
        _raise_nofile(4096 + 16 * num_tables)
        _cloexec_open_fds()
        ctx = mp.get_context("spawn")
        if ctx.get_start_method() != "spawn":
            raise RuntimeError(f"start method must be spawn, got {ctx.get_start_method()}")
        infer_lock = threading.Lock()
        self._train_device = train_device
        self._infer_device = infer_device
        self._model = compiled
        self._rank = rank
        self._ddp_model: torch.nn.parallel.DistributedDataParallel | None = None
        self._ddp_store: dist.TCPStore | None = None
        self._cpu_ids = cpu_ids
        self._physical_gpu_ids = physical_gpu_ids
        self._min_update_steps = min_update_steps
        self._num_tables = num_tables
        self._infer_batch = infer_batch
        self._agent = Top10PPOAgent(
            compiled,
            optimizer,
            PPOConfig(
                clip_ratio=0.2,
                train_iters=2,
                amp=True,
                amp_dtype=torch.bfloat16,
            ),
        )
        self._tables = PersistentTables(
            ctx=ctx,
            num_tables=num_tables,
            infer_batch=infer_batch,
            seed=seed,
            device=infer_device,
            infer_model=infer_model,
            infer_lock=infer_lock,
        )
        self._batch_q: Queue = Queue(maxsize=2)
        self._sampler_errors: list[BaseException] = []
        self._stop_sampling = threading.Event()

        def _sample_loop() -> None:
            _bind_thread_device(infer_device)
            try:
                while not self._stop_sampling.is_set():
                    collect_started = time.perf_counter()
                    payload = self._tables.collect(
                        min_update_steps=min_update_steps,
                        stop_event=self._stop_sampling,
                    )
                    collect_sec = time.perf_counter() - collect_started
                    if collect_sec <= 0.0:
                        raise RuntimeError(f"collect_sec must be > 0, got {collect_sec}")
                    rows, table_perfs, collect_times, infer_calls, pipeline = payload
                    self._batch_q.put(
                        (
                            (
                                rows,
                                table_perfs,
                                collect_times,
                                infer_calls,
                                pipeline,
                            ),
                            collect_sec,
                        )
                    )
            except _CollectStopped:
                return
            except BaseException as exc:
                if self._stop_sampling.is_set():
                    return
                self._sampler_errors.append(exc)
                raise

        self._sampler = threading.Thread(target=_sample_loop, name="online-sample", daemon=False)
        self._sampler.start()

    def ping(self) -> str:
        if self._tables is None:
            raise RuntimeError("online actor ping before tables")
        _eager_train(self._model)
        if torch.cuda.device_count() != 2:
            raise RuntimeError(
                f"rank {self._rank} actor must see exactly 2 local GPUs, "
                f"saw {torch.cuda.device_count()}"
            )
        return {
            "status": "ok",
            "rank": self._rank,
            "local_devices": [str(self._train_device), str(self._infer_device)],
            "physical_gpu_ids": self._physical_gpu_ids,
            "visible_gpus": int(torch.cuda.device_count()),
            "cpu_count": len(self._cpu_ids),
            "node_ip": ray.util.get_node_ip_address(),
        }

    def create_rendezvous(self) -> tuple[str, int]:
        if self._rank != 0 or self._ddp_store is not None:
            raise RuntimeError("only rank 0 may create the DDP rendezvous")
        host = ray.util.get_node_ip_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = int(probe.getsockname()[1])
        self._ddp_store = dist.TCPStore(
            host_name=host,
            port=port,
            world_size=2,
            is_master=True,
            timeout=timedelta(minutes=5),
            wait_for_workers=False,
        )
        return host, port

    def init_distributed(self, host: str, port: int) -> str:
        if self._ddp_model is not None:
            raise RuntimeError(f"rank {self._rank} DDP already initialized")
        if self._rank == 0:
            if self._ddp_store is None:
                raise RuntimeError("rank 0 rendezvous was not created")
        else:
            self._ddp_store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=2,
                is_master=False,
                timeout=timedelta(minutes=5),
            )
        dist.init_process_group(
            backend="nccl",
            store=self._ddp_store,
            rank=self._rank,
            world_size=2,
            timeout=timedelta(minutes=5),
        )
        self._ddp_model = torch.nn.parallel.DistributedDataParallel(
            self._model,
            device_ids=[0],
            output_device=0,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        self._agent.model = self._ddp_model
        return f"rank={self._rank} device=cuda:0/cuda:1 cpus={len(self._cpu_ids)}"

    def _take_sample(self) -> tuple[
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
            if self._sampler_errors:
                raise self._sampler_errors[0]
            if not self._sampler.is_alive() and self._batch_q.empty():
                raise RuntimeError("sampler thread died")
            try:
                return self._batch_q.get(timeout=1.0)
            except Empty:
                continue

    def step(self) -> dict[str, object]:
        if self._ddp_model is None or not dist.is_initialized():
            raise RuntimeError(f"rank {self._rank} step before DDP initialization")
        cpu0 = time.process_time()
        started = time.perf_counter()
        (
            rows,
            table_perfs,
            collect_times,
            infer_calls,
            pipeline,
        ), collect_sec = self._take_sample()
        if not isinstance(rows, list):
            raise TypeError(f"rows is {type(rows).__name__}")
        t_stack0 = time.perf_counter()
        cpu_batch = _stack_rows_cpu(rows)
        if int(cpu_batch["state"].shape[0]) != len(rows):
            raise RuntimeError(
                f"stacked rows {cpu_batch['state'].shape[0]} != {len(rows)}"
            )
        step_count = int(cpu_batch["state"].shape[0])
        if step_count < self._min_update_steps:
            raise RuntimeError(
                f"update batch has {step_count} steps, required >= {self._min_update_steps}"
            )
        _bind_thread_device(self._train_device)
        self._model.train()
        batch = _cpu_batch_to_device(cpu_batch, self._train_device)
        minibatches = _split_update_minibatches(batch)
        stack_sec = time.perf_counter() - t_stack0
        t_upd0 = time.perf_counter()
        minibatch_infos = [self._agent.update(minibatch) for minibatch in minibatches]
        torch.cuda.synchronize(self._train_device)
        update_sec = time.perf_counter() - t_upd0
        self._tables.sync_infer_from(self._model)
        info = _mean_train_info(minibatch_infos)
        values = np.asarray(cpu_batch["value"], dtype=np.float64)
        returns = np.asarray(cpu_batch["returns"], dtype=np.float64)
        if values.shape != (step_count,) or returns.shape != (step_count,):
            raise RuntimeError(
                f"value/returns shape {values.shape}/{returns.shape} != ({step_count},)"
            )
        info["explained_variance"] = _explained_variance(returns, values)
        del cpu_batch, batch, minibatches
        finished = time.perf_counter()
        total_sec = finished - started
        parent_cpu_sec = time.process_time() - cpu0
        if total_sec <= 0:
            raise RuntimeError(f"total_sec must be > 0, got {total_sec}")
        parent_gpu = _device_memory_bytes(self._train_device)
        if parent_gpu is None:
            raise RuntimeError("train device reported no GPU memory stats")
        return {
            "steps": step_count,
            "table_perfs": table_perfs,
            "collect_times": collect_times,
            "infer_calls": infer_calls,
            "pipeline": pipeline,
            "collect_sec": collect_sec,
            "stack_sec": stack_sec,
            "update_sec": update_sec,
            "total_sec": total_sec,
            "parent_cpu_sec": parent_cpu_sec,
            "parent_rss": _rss_bytes(os.getpid()),
            "parent_gpu": parent_gpu,
            "train_info": info,
        }

    def export_checkpoint(self) -> dict[str, object]:
        model = _eager_train(self._model)
        return {
            "model_state_dict": cpu_state_dict(model),
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
        }

    def stop(self) -> None:
        self._stop_sampling.set()
        parent_wake_fd = getattr(self._tables, "_parent_wake_fd", None)
        if parent_wake_fd is not None:
            os.eventfd_write(parent_wake_fd, 1)

    def close(self) -> None:
        self.stop()
        while True:
            try:
                self._batch_q.get_nowait()
            except Empty:
                break
        self._sampler.join(timeout=20.0)
        self._tables.close()
        if self._sampler.is_alive():
            raise RuntimeError("sampler thread did not stop")
        if self._sampler_errors:
            raise self._sampler_errors[0]
        if dist.is_initialized():
            dist.destroy_process_group()
        self._ddp_store = None


def _save_checkpoint_dict(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def run_online(args: argparse.Namespace) -> None:
    _require_ray_driver()
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
    _connect_cluster()
    weights = _cpu_init_weights(args.init_checkpoint)
    node_ip, node_cpus = _select_gpu_node()
    cpus_each = node_cpus // 2
    if cpus_each <= 0:
        raise RuntimeError(f"invalid per-group CPU allocation {cpus_each} on {node_ip}")
    print(
        f"ddp_prepare node={node_ip} node_cpus={node_cpus} "
        f"cpus_each={cpus_each} bundles=2x(CPU={cpus_each},GPU=2)",
        flush=True,
    )
    pg = placement_group(
        [
            {"CPU": cpus_each, "GPU": 2, f"node:{node_ip}": 0.001},
            {"CPU": cpus_each, "GPU": 2, f"node:{node_ip}": 0.001},
        ],
        strategy="PACK",
        name=f"danks-ddp-{os.getpid()}",
    )
    print("ddp_prepare waiting_for_placement_group", flush=True)
    ready, _ = ray.wait([pg.ready()], timeout=60.0)
    if not ready:
        remove_placement_group(pg)
        raise TimeoutError(
            "placement group did not become ready in 60s; "
            f"target_node={node_ip} node_cpus={node_cpus}"
        )
    print("ddp_prepare placement_group_ready", flush=True)
    actors = [
        OnlineActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=rank,
            ),
            num_cpus=cpus_each,
            num_gpus=2,
        ).remote(
            weights,
            infer_batch,
            num_tables,
            args.seed,
            args.lr,
            args.min_update_steps,
            rank,
        )
        for rank in (0, 1)
    ]
    print("ddp_prepare actors_created", flush=True)
    print("ddp_prepare waiting_for_actor_ping", flush=True)
    ping_refs = [actor.ping.remote() for actor in actors]
    ready, _ = ray.wait(ping_refs, num_returns=2, timeout=180.0)
    if len(ready) != 2:
        raise TimeoutError("actor initialization/ping did not finish in 180s")
    pings = ray.get(ping_refs)
    if [ping.get("rank") for ping in pings] != [0, 1]:
        raise RuntimeError(f"DDP actor ranks incorrect: {pings!r}")
    if any(
        ping.get("status") != "ok"
        or ping.get("local_devices") != ["cuda:0", "cuda:1"]
        or ping.get("visible_gpus") != 2
        for ping in pings
    ):
        raise RuntimeError(f"DDP actor local GPU mapping incorrect: {pings!r}")
    physical_ids = [
        gpu_id
        for ping in pings
        for gpu_id in ping.get("physical_gpu_ids", [])
    ]
    if len(physical_ids) != 4 or len(set(physical_ids)) != 4:
        raise RuntimeError(f"DDP physical GPU mapping overlaps or is incomplete: {pings!r}")
    if pings[0].get("node_ip") != pings[1].get("node_ip"):
        raise RuntimeError(f"DDP actors landed on different nodes: {pings!r}")
    if any(ping.get("cpu_count", 0) <= 0 for ping in pings):
        raise RuntimeError(f"DDP actor CPU affinity is empty: {pings!r}")
    print(
        "ddp_groups=" + repr([
            {key: ping[key] for key in ("rank", "node_ip", "cpu_count", "local_devices")}
            for ping in pings
        ]),
        flush=True,
    )
    print("ddp_prepare creating_rendezvous", flush=True)
    rendezvous = ray.get(actors[0].create_rendezvous.remote())
    print(f"ddp_prepare rendezvous={rendezvous[0]}:{rendezvous[1]}", flush=True)
    init_refs = [actor.init_distributed.remote(*rendezvous) for actor in actors]
    ready, _ = ray.wait(init_refs, num_returns=2, timeout=180.0)
    if len(ready) != 2:
        raise TimeoutError("DDP process-group initialization did not finish in 180s")
    ray.get(init_refs)
    print("ddp_prepare process_group_ready", flush=True)
    output_dir = _resolve_output_dir(args.output)
    exp_stamp = _experiment_stamp()
    previous_ckpt: Path | None = None
    print(f"experiment={exp_stamp} output_dir={output_dir} cluster={os.environ.get('RAY_ADDRESS', '')}", flush=True)
    run_error: BaseException | None = None
    try:
        for update in range(args.updates):
            results = ray.get([actor.step.remote() for actor in actors])
            if len(results) != 2:
                raise RuntimeError(f"expected 2 DDP results, got {len(results)}")
            result = {
                "steps": sum(int(item["steps"]) for item in results),
                "table_perfs": [perf for item in results for perf in item["table_perfs"]],
                "collect_times": {
                    key: max(float(item["collect_times"][key]) for item in results)
                    for key in results[0]["collect_times"]
                },
                "infer_calls": sum(int(item["infer_calls"]) for item in results),
                "pipeline": {
                    key: (
                        sum(float(item["pipeline"][key]) for item in results)
                        if key == "rank_call_n"
                        else (
                            sum(float(item["pipeline"][key]) for item in results) / 2.0
                            if key in {"batch_size_mean", "ready_nowait_mean", "q_left_mean"}
                            else max(float(item["pipeline"][key]) for item in results)
                        )
                    )
                    for key in results[0]["pipeline"]
                },
                "collect_sec": max(float(item["collect_sec"]) for item in results),
                "stack_sec": max(float(item["stack_sec"]) for item in results),
                "update_sec": max(float(item["update_sec"]) for item in results),
                "total_sec": max(float(item["total_sec"]) for item in results),
                "parent_cpu_sec": sum(float(item["parent_cpu_sec"]) for item in results),
                "parent_rss": max(int(item["parent_rss"]) for item in results),
                "parent_gpu": {
                    key: max(int(item["parent_gpu"][key]) for item in results)
                    for key in results[0]["parent_gpu"]
                },
                "train_info": {
                    key: sum(float(item["train_info"][key]) for item in results) / 2.0
                    for key in results[0]["train_info"]
                },
            }
            save_sec = 0.0
            if (update + 1) % _SAVE_EVERY_UPDATES == 0:
                t_save0 = time.perf_counter()
                ckpt_path = _checkpoint_path(
                    output_dir, exp_stamp, update + 1, args.min_update_steps
                )
                payload = ray.get(actors[0].export_checkpoint.remote())
                _save_checkpoint_dict(ckpt_path, payload)
                if previous_ckpt is not None:
                    if previous_ckpt == ckpt_path:
                        raise RuntimeError(f"checkpoint path reused: {ckpt_path}")
                    previous_ckpt.unlink()
                previous_ckpt = ckpt_path
                save_sec = time.perf_counter() - t_save0
            collect_times = result["collect_times"]
            if not isinstance(collect_times, dict):
                raise TypeError(f"collect_times is {type(collect_times).__name__}")
            table_perfs = result["table_perfs"]
            if not isinstance(table_perfs, list):
                raise TypeError(f"table_perfs is {type(table_perfs).__name__}")
            pipeline = result["pipeline"]
            if not isinstance(pipeline, dict):
                raise TypeError(f"pipeline is {type(pipeline).__name__}")
            train_info = result["train_info"]
            if not isinstance(train_info, dict):
                raise TypeError(f"train_info is {type(train_info).__name__}")
            parent_gpu = result["parent_gpu"]
            if not isinstance(parent_gpu, dict):
                raise TypeError(f"parent_gpu is {type(parent_gpu).__name__}")
            total_sec = float(result["total_sec"]) + save_sec
            parent_cpu_sec = float(result["parent_cpu_sec"])
            if total_sec <= 0:
                raise RuntimeError(f"total_sec must be > 0, got {total_sec}")
            _print_perf_report(
                update=update + 1,
                updates=args.updates,
                steps=int(result["steps"]),
                num_tables=num_tables,
                infer_batch=infer_batch,
                infer_calls=int(result["infer_calls"]),
                device=torch.device("cuda"),
                collect_wall=float(result["collect_sec"]),
                total_wall=total_sec,
                parent_times={
                    **collect_times,
                    "stack_sec": float(result["stack_sec"]),
                    "update_sec": float(result["update_sec"]),
                    "save_sec": save_sec,
                    "cpu_util_pct": 100.0 * parent_cpu_sec / total_sec,
                },
                parent_rss=int(result["parent_rss"]),
                parent_gpu=parent_gpu,
                table_perfs=table_perfs,
                train_info=train_info,
                pipeline=pipeline,
            )
    except BaseException as exc:
        run_error = _unwrap_task_error(exc)
        if run_error is not exc:
            raise run_error from exc
        raise
    finally:
        close_error: BaseException | None = None
        try:
            ray.get([actor.stop.remote() for actor in actors])
        except BaseException as exc:
            close_error = _unwrap_task_error(exc)
        try:
            ray.get([actor.close.remote() for actor in actors])
        except BaseException as exc:
            if close_error is None:
                close_error = _unwrap_task_error(exc)
        try:
            for actor in actors:
                ray.kill(actor)
        except BaseException as exc:
            if close_error is None:
                close_error = exc
        try:
            remove_placement_group(pg)
        except BaseException as exc:
            if close_error is None:
                close_error = exc
        ray.shutdown()
        if close_error is not None:
            if run_error is not None:
                raise ExceptionGroup(
                    "run and shutdown failed",
                    [run_error, close_error],
                )
            raise close_error
