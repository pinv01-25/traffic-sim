"""
Shared helpers for traffic-sim analysis scripts.
"""
from pathlib import Path


def find_sim_dir(suffix: str, cwd: Path | None = None) -> Path | None:
    """
    Find a simulation directory by suffix convention.

    Search order:
      1. sim_{suffix}           (exact conventional name)
      2. Any dir ending in _{suffix}  (e.g. ab_run_A, my_sim_B)
         — picks the one with most recent modification time

    Args:
        suffix: "A", "B", or "C"
        cwd:    Directory to search in (default: Path.cwd())

    Returns:
        Path to the directory, or None if not found.
    """
    root = cwd or Path.cwd()

    # 1. Exact conventional name
    exact = root / f"sim_{suffix}"
    if exact.is_dir():
        return exact

    # 2. Any directory ending with _{suffix}
    candidates = [
        d for d in root.iterdir()
        if d.is_dir() and d.name.endswith(f"_{suffix}")
    ]
    if not candidates:
        return None

    # Pick most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def require_sim_dir(suffix: str, cwd: Path | None = None) -> Path:
    """Like find_sim_dir but raises SystemExit with a clear message if not found."""
    d = find_sim_dir(suffix, cwd)
    if d is None:
        root = cwd or Path.cwd()
        print(
            f"ERROR: No se encontró directorio para condición '{suffix}'.\n"
            f"  Buscado en: {root}\n"
            f"  Esperado:   sim_{suffix}/  o cualquier carpeta que termine en _{suffix}/\n"
            f"  Existentes: {[p.name for p in root.iterdir() if p.is_dir()]}"
        )
        raise SystemExit(1)
    return d
