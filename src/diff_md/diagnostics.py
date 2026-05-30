"""Diagnostic utilities for Diff-MD.

Provides startup diagnostics, hardware info tables, and configuration summaries
for mdrun and optimize workflows.
"""

import os
import random
import subprocess
import platform
import textwrap
from typing import Optional
from datetime import datetime

import jax
import jax.numpy as jnp

from .logger import Logger


def get_cpu_info() -> dict:
    """Gather CPU information."""
    info = {
        "model": "unknown",
        "cores": os.cpu_count() or 0,
        "architecture": platform.machine(),
    }
    
    # Try to get detailed CPU info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    info["model"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass
    
    return info


def get_gpu_info() -> list[dict]:
    """Gather GPU information using nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": int(parts[2]),
                        "driver": parts[3],
                        "compute_cap": parts[4],
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    return gpus


def get_memory_info() -> dict:
    """Get system memory information."""
    info = {"total_gb": 0.0, "available_gb": 0.0}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["total_gb"] = kb / (1024 ** 2)
                elif line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    info["available_gb"] = kb / (1024 ** 2)
    except (FileNotFoundError, PermissionError):
        pass
    return info


def format_hardware_table(
    title: str = "DIFF-MD",
    mpi_size: int = 1,
    mpi_rank: int = 0,
    show_warnings: bool = True,
) -> str:
    """Generate a formatted hardware diagnostics table.
    
    Parameters
    ----------
    title : str
        Title for the diagnostics block.
    mpi_size : int
        Number of MPI ranks.
    mpi_rank : int
        Current MPI rank (used for per-rank GPU info).
    show_warnings : bool
        Whether to include warnings about suboptimal settings.
    
    Returns
    -------
    str
        Formatted multi-line string with hardware diagnostics.
    """
    width = 70
    sep = "=" * width
    
    lines = [
        sep,
        f"  {title} — STARTUP DIAGNOSTICS".center(width),
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(width),
        sep,
        "",
        "  JAX / XLA Configuration",
        "  " + "-" * 30,
    ]
    
    # JAX info
    backend = jax.default_backend()
    devices = jax.devices()
    device_str = f"{len(devices)} × {devices[0].platform.upper()}" if devices else "none"
    
    lines.extend([
        f"    JAX version:          {jax.__version__}",
        f"    Backend:              {backend.upper()}",
        f"    Devices:              {device_str}",
        f"    jax_enable_x64:       {jax.config.jax_enable_x64}",
        f"    jax_debug_nans:       {jax.config.jax_debug_nans}",
        "",
    ])
    
    # MPI info
    lines.extend([
        "  MPI Configuration",
        "  " + "-" * 30,
        f"    MPI ranks:            {mpi_size}",
    ])
    
    # CPU info
    cpu = get_cpu_info()
    lines.extend([
        "",
        "  CPU Information",
        "  " + "-" * 30,
        f"    Model:                {cpu['model'][:40]}",
        f"    Cores:                {cpu['cores']}",
        f"    Architecture:         {cpu['architecture']}",
    ])
    
    # Memory info
    mem = get_memory_info()
    if mem["total_gb"] > 0:
        lines.extend([
            "",
            "  System Memory",
            "  " + "-" * 30,
            f"    Total:                {mem['total_gb']:.1f} GB",
            f"    Available:            {mem['available_gb']:.1f} GB",
        ])
    
    # GPU info (if backend is GPU/CUDA)
    if backend.lower() in ("gpu", "cuda"):
        gpus = get_gpu_info()
        if gpus:
            lines.extend([
                "",
                "  GPU Information",
                "  " + "-" * 30,
            ])
            for gpu in gpus:
                lines.append(
                    f"    [{gpu['index']}] {gpu['name']}: "
                    f"{gpu['memory_mb']} MB, CC {gpu['compute_cap']}, driver {gpu['driver']}"
                )
        
        # CUDA environment
        lines.extend([
            "",
            "  CUDA Environment",
            "  " + "-" * 30,
            f"    CUDA_VISIBLE_DEVICES:        {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
            f"    XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'unset')}",
            f"    XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'unset')}",
        ])
    
    lines.append("")
    lines.append(sep)
    
    # Warnings
    if show_warnings:
        warnings = []
        
        # Check for common issues
        preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true").lower()
        if preallocate == "false":
            warnings.append(
                "XLA_PYTHON_CLIENT_PREALLOCATE=false — causes massive slowdown and CUDA crashes. "
                "Remove from your environment or set to 'true'."
            )
        
        if not jax.config.jax_enable_x64 and backend.lower() in ("gpu", "cuda"):
            warnings.append(
                "jax_enable_x64=False (float32 mode) — for atomistic PME training, "
                "float32 may cause gradient explosion. Set JAX_ENABLE_X64=true if needed."
            )
        
        if warnings:
            lines.append("")
            lines.append("  ⚠️  WARNINGS")
            lines.append("  " + "-" * 30)
            for w in warnings:
                # Word-wrap long warnings
                wrapped = [w[i:i+60] for i in range(0, len(w), 60)]
                lines.append(f"    • {wrapped[0]}")
                for cont in wrapped[1:]:
                    lines.append(f"      {cont}")
            lines.append("")
            lines.append(sep)
    
    return "\n".join(lines)


def format_simulation_config(
    n_atoms: int,
    n_steps: int,
    dt: float,
    temperature: Optional[float] = None,
    ensemble: str = "NVT",
    integrator: str = "Velocity Verlet",
    electrostatics: Optional[str] = None,
    lj_cutoff: Optional[float] = None,
    box_size: Optional[tuple] = None,
    grad_method: Optional[str] = None,
    n_trainable_params: Optional[int] = None,
) -> str:
    """Generate a formatted simulation configuration summary.

    All physical quantities should be in standard MD units (nm, ps, K, kJ/mol).

    Returns
    -------
    str
        Formatted multi-line string with simulation parameters.
    """
    width = 70
    sep = "-" * width

    lines = [
        "",
        "  Simulation Configuration",
        "  " + sep,
        f"    Atoms:                {n_atoms:,}",
        f"    Steps:                {n_steps:,}",
        f"    Timestep:             {dt:.4f} ps",
        f"    Total time:           {n_steps * dt:.2f} ps",
    ]

    if temperature is not None:
        lines.append(f"    Temperature:          {temperature:.1f} K")

    lines.append(f"    Ensemble:             {ensemble}")
    lines.append(f"    Integrator:           {integrator}")

    if electrostatics:
        lines.append(f"    Electrostatics:       {electrostatics}")

    if lj_cutoff:
        lines.append(f"    LJ cutoff:            {lj_cutoff:.3f} nm")

    if box_size:
        lines.append(f"    Box size:             {box_size[0]:.2f} × {box_size[1]:.2f} × {box_size[2]:.2f} nm")

    if grad_method:
        lines.append("")
        lines.append("  Training Configuration")
        lines.append("  " + "-" * 30)
        lines.append(f"    Gradient method:      {grad_method}")

        if n_trainable_params:
            lines.append(f"    Trainable params:     {n_trainable_params}")

    lines.append("  " + sep)
    lines.append("")

    return "\n".join(lines)


def _gpu_mem_mb_diag() -> tuple:
    """Return (used_MB, total_MB) for the visible GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().splitlines()[0].split(", ")
            return int(used), int(total)
    except Exception:
        pass
    return 0, 0


def format_run_header(
    title: str,
    config,
    nn_options=None,
    params=None,
    mpi_size: int = 1,
    restart_path: Optional[str] = None,
    restart_mode: Optional[str] = None,
    extra_lines: Optional[list] = None,
) -> str:
    """Format the wide banner header printed at the start of every run.

    Mirrors the header block used in the Olivia test scripts so that
    ``optimize.main``, ``mdrun``, etc. all produce consistent startup info.

    Parameters
    ----------
    title : str
        Title string, e.g. ``"DIFF-MD OPTIMIZE"``.
    config : Config
        Simulation config object (diff_md.config.Config).
    nn_options : NNoptions, optional
        Training options.  When provided, training-specific fields are
        included (grad_method, n_epochs, optimizer, clipping bounds, …).
    params : GeneralModel, optional
        Current parameter object.  When provided, prints LJ_param values.
    mpi_size : int
        Number of MPI ranks.
    restart_path : str, optional
        Path to checkpoint being restarted from.  Printed when set.
    restart_mode : str, optional
        Effective restart mode used for the current run.
    extra_lines : list[str], optional
        Additional lines appended before the closing separator.

    Returns
    -------
    str
        Formatted multi-line header string (no trailing newline).
    """
    import numpy as _np

    width = 80
    sep = "=" * width

    _baro_names = {0: "none", 1: "berendsen", 2: "c-rescale"}
    _baro_type_names = {
        0: "none", 1: "isotropic", 2: "semiisotropic", 3: "surface_tension"
    }

    is_npt = (getattr(config, "ensemble", "NVT").upper() == "NPT"
              or int(getattr(config, "barostat", 0)) > 0)
    ct = int(getattr(config, "coulombtype", 0))
    elec_str = {0: "none", 1: "PME", 2: "reaction-field"}.get(ct, f"type={ct}")
    barostat_val  = int(getattr(config, "barostat", 0))
    barostat_type = int(getattr(config, "barostat_type", 0) or 0)
    thermostat_str = getattr(config, "thermostat", "v-rescale")
    n_steps        = int(getattr(config, "n_steps", 0))
    dt             = float(getattr(config, "outer_ts", 0.0))
    n_particles    = int(getattr(config, "n_particles", 0))
    nlist_method   = getattr(config, "nlist_method", "cell")
    ns_nlist       = int(getattr(config, "ns_nlist", 1))
    rv             = float(getattr(config, "rv", 0.0))
    skin           = float(getattr(config, "skin", 0.0))
    tau_t          = getattr(config, "tau_t", None)
    tau_p          = getattr(config, "tau_p", None)
    target_pressure     = getattr(config, "target_pressure", None)
    target_temperature  = getattr(config, "target_temperature", None)
    n_print        = int(getattr(config, "n_print", 0))
    pme_order      = int(getattr(config, "pme_order", 4))
    nrexcl         = int(getattr(config, "nrexcl", 3))
    ff_family      = getattr(config, "ff_family", "unknown")
    combining_rule = getattr(config, "combining_rule", "unknown")
    ensemble_str   = getattr(config, "ensemble", "NVT").upper()

    gpu_used, gpu_total = _gpu_mem_mb_diag()

    backend   = jax.default_backend()
    n_devices = len(jax.devices())
    dtype_str = "float64" if bool(jax.config.jax_enable_x64) else "float32"

    lines = [
        sep,
        f"  {title}".center(width),
        sep,
        f"  Date:              {datetime.now().isoformat()}",
        "",
        "  ── System ──────────────────────────────────────────────────────────",
        f"  n_particles:       {n_particles:,}",
        f"  n_steps:           {n_steps:,}",
        f"  timestep:          {dt:.4f} ps  →  total = {n_steps * dt:.2f} ps",
        f"  n_print:           {n_print}",
        f"  Ensemble:          {ensemble_str}",
        f"  Thermostat:        {thermostat_str}"
        + (f"  tau_t={tau_t:.2f} ps" if tau_t else ""),
    ]

    if target_temperature is not None:
        lines.append(f"  Target temp.:      {target_temperature:.1f} K")

    if is_npt:
        lines.append(
            f"  Barostat:          {_baro_names.get(barostat_val, barostat_val)}"
            f"  type={_baro_type_names.get(barostat_type, barostat_type)}"
            + (f"  tau_p={tau_p:.2f} ps" if tau_p else "")
        )
        if target_pressure is not None:
            lines.append(f"  Target pressure:   {_np.asarray(target_pressure)} bar")
        lines.append(f"  Box size:          {_np.asarray(config.box_size)} nm")

    lines += [
        "",
        "  ── Neighbour list / Force-field ─────────────────────────────────────",
        f"  FF family:         {ff_family}  ({combining_rule})",
        f"  Electrostatics:    {elec_str}"
        + (f"  (order={pme_order})" if ct == 1 else ""),
        f"  LJ cutoff rv:      {rv:.3f} nm"
        f"  rc={getattr(config, 'rc', 0.0):.3f} nm"
        f"  rlj={getattr(config, 'rlj', 0.0):.3f} nm",
        f"  Nlist method:      {nlist_method}  ns_nlist={ns_nlist}"
        + (f"  skin={skin:.3f} nm" if skin else ""),
        f"  nrexcl:            {nrexcl}",
    ]

    if nn_options is not None:
        n_epochs       = int(getattr(nn_options, "n_epochs", 0))
        grad_method    = getattr(nn_options, "grad_method", "reverse")
        teacher_forcing = bool(getattr(nn_options, "teacher_forcing", False))
        equilibration  = int(getattr(nn_options, "equilibration", 0))
        fd_eps         = float(getattr(nn_options, "fd_epsilon", 1e-4))
        clip_s_min     = float(getattr(nn_options, "clip_sigma_min", 0.05))
        clip_s_max     = float(getattr(nn_options, "clip_sigma_max", 2.0))
        clip_e_min     = float(getattr(nn_options, "clip_epsilon_min", 0.001))
        clip_e_max     = float(getattr(nn_options, "clip_epsilon_max", 100.0))
        clear_xla      = int(getattr(nn_options, "clear_xla_cache", 0))

        opt = getattr(nn_options, "optimizer", None)
        opt_name = "unknown"
        if opt is not None:
            try:
                r = repr(opt).lower()
                for candidate in ("adam", "sgd", "rmsprop", "adagrad",
                                  "lamb", "lion", "noisy_sgd"):
                    if candidate in r:
                        opt_name = candidate
                        break
                else:
                    opt_name = repr(opt).split("(")[0]
            except Exception:
                opt_name = type(opt).__name__

        lines += [
            "",
            "  ── Training ─────────────────────────────────────────────────────────",
            f"  n_epochs:          {n_epochs}",
            f"  Grad method:       {grad_method}"
            + (f"  (fd_epsilon={fd_eps:.1e})" if grad_method == "finite_diff" else ""),
            f"  Teacher forcing:   {teacher_forcing}",
            f"  Equilibration:     {equilibration} steps",
            f"  Optimizer:         {opt_name}",
            f"  Param clipping:    σ=[{clip_s_min}, {clip_s_max}] nm  "
            f"ε=[{clip_e_min}, {clip_e_max}] kJ/mol",
            f"  Clear XLA cache:   {'every ' + str(clear_xla) + ' epoch(s)' if clear_xla else 'off'}",
        ]

        if params is not None:
            lj = getattr(params, "LJ_param", None)
            n_lj = int(lj.shape[0]) if lj is not None else 0
            lines.append(
                f"  Trainable params:  {n_lj}"
                + (f"  (LJ_param = {_np.asarray(lj)})" if lj is not None else "")
            )

    if restart_path:
        lines += [
            "",
            "  ── Restart ──────────────────────────────────────────────────────────",
            f"  Checkpoint:        {restart_path}",
        ]
        if restart_mode:
            lines.append(f"  Restart mode:      {restart_mode}")

    # ── Runtime & Hardware (unified) ──────────────────────────────────────
    total_devices = mpi_size * n_devices
    _devices = jax.devices()
    device_label = _devices[0].platform.upper() if _devices else "CPU"
    if mpi_size > 1:
        device_info = f"{total_devices} × {device_label} total ({n_devices}/rank)"
    else:
        device_info = f"{n_devices} × {device_label}"

    lines += [
        "",
        "  ── Runtime & Hardware ────────────────────────────────────────────────",
        f"  MPI ranks:         {mpi_size}",
        f"  Devices:           {device_info}",
        f"  Working dtype:     {dtype_str}",
        f"  JAX version:       {jax.__version__}",
        f"  jax_debug_nans:    {jax.config.jax_debug_nans}",
    ]

    cpu = get_cpu_info()
    lines.append(f"  CPU:               {cpu['model'][:45]}  ({cpu['cores']} cores)")

    mem = get_memory_info()
    if mem["total_gb"] > 0:
        lines.append(
            f"  System memory:     {mem['total_gb']:.1f} GB total,"
            f" {mem['available_gb']:.1f} GB available"
        )

    if gpu_total > 0:
        lines.append(
            f"  GPU memory:        {gpu_used}/{gpu_total} MB"
            f"  ({100*gpu_used/gpu_total:.1f}% used at startup)"
        )
        gpus = get_gpu_info()
        for g in gpus:
            lines.append(
                f"  GPU [{g['index']}]:          {g['name']},"
                f" CC {g['compute_cap']}, driver {g['driver']}"
            )

    xla_preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "unset")
    xla_mem_frac    = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "unset")
    cvd             = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    lines += [
        f"  CUDA_VISIBLE_DEV:  {cvd}",
        f"  XLA_PREALLOCATE:   {xla_preallocate}"
        f"  MEM_FRACTION={xla_mem_frac}",
    ]

    # ── Warnings ──
    warnings = []
    if xla_preallocate == "false":
        warnings.append(
            "XLA_PYTHON_CLIENT_PREALLOCATE=false — may cause CUDA OOM. "
            "Remove or set to 'true'."
        )
    if not jax.config.jax_enable_x64 and backend.lower() in ("gpu", "cuda"):
        warnings.append(
            "jax_enable_x64=False (float32) — may cause gradient explosion "
            "for atomistic PME training."
        )
    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"  ⚠️  {w}")

    if extra_lines:
        lines.append("")
        lines.extend(
            (f"  {l}" if not l.startswith("  ") else l) for l in extra_lines
        )

    lines.append(sep)
    return "\n".join(lines)


def print_startup_diagnostics(
    title: str = "DIFF-MD",
    mpi_comm=None,
    logger=None,
    config=None,
    nn_options=None,
    params=None,
    restart_path: Optional[str] = None,
    restart_mode: Optional[str] = None,
):
    """Print startup diagnostics (rank 0 only).

    When *config* is provided, prints the rich ``format_run_header`` banner
    (system, FF, training, runtime) in addition to the hardware table.
    Without *config*, prints only the hardware table (legacy behaviour).

    Parameters
    ----------
    title : str
        Title for the diagnostics block.
    mpi_comm : MPI.Comm, optional
        MPI communicator.  Only rank 0 prints.
    logger : Logger, optional
        Logger instance.  If None, uses print().
    config : Config, optional
        Simulation config.  When set, ``format_run_header`` is used.
    nn_options : NNoptions, optional
        Training options (passed through to ``format_run_header``).
    params : GeneralModel, optional
        Parameters (passed through to ``format_run_header``).
    restart_path : str, optional
        Checkpoint path being restarted from.
    restart_mode : str, optional
        Effective restart mode used for the current run.
    """
    if mpi_comm is not None:
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
    else:
        rank = 0
        size = 1

    if rank != 0:
        return

    if config is not None:
        text = format_run_header(
            title=title,
            config=config,
            nn_options=nn_options,
            params=params,
            mpi_size=size,
            restart_path=restart_path,
            restart_mode=restart_mode,
        )
    else:
        text = format_hardware_table(title=title, mpi_size=size, mpi_rank=rank)

    if logger is not None:
        logger.info("\n" + text)
    else:
        print(text, flush=True)


_SIGNATURE_WIDTH = 72


def _parse_quotes_file(quotes_path: str) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    with open(quotes_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue
            quote, author, date = parts
            if quote and author and date:
                entries.append((quote, author, date))
    return entries


def format_run_signature_block(quote: str, author: str, date: str) -> str:
    rule = "=" * _SIGNATURE_WIDTH
    body_width = _SIGNATURE_WIDTH - 2
    wrapped_quote = textwrap.fill(
        f'"{quote}"',
        width=body_width,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    citation = f"-- {author}, {date}"
    if len(citation) > body_width:
        citation = citation[: body_width - 1] + "…"
    citation_line = citation.rjust(_SIGNATURE_WIDTH)
    return "\n".join([rule, wrapped_quote, citation_line, rule])


def print_run_signature(quotes_path: Optional[str] = None) -> None:
    """Append a GROMACS-style signed-quote footer to the run log.

    Reads ``quotes.txt`` (next to this module by default).  Each non-comment,
    non-blank line must be ``quote | author | date`` (pipe-delimited).  When
    the file is absent, the call is a silent no-op (that is the user's "off"
    switch).  Always rank-0-gated through ``Logger.rank0``.
    """
    if quotes_path is None:
        quotes_path = os.path.join(os.path.dirname(__file__), "quotes.txt")
    if not os.path.isfile(quotes_path):
        return
    try:
        entries = _parse_quotes_file(quotes_path)
    except OSError:
        return
    if not entries:
        return
    quote, author, date = random.choice(entries)
    block = format_run_signature_block(quote, author, date)
    Logger.rank0.info("\n" + block)
