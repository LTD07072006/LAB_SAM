"""Single-file coordinator for the Week 6 localisation experiments.

Modes:

``compare``
    Compare the baseline detector, ROI refiner, and restored v3 detector and
    write annotated PNGs/contact sheet.
``demo``
    Run ``main_localization.py`` on a small image sample.
``benchmark``
    Run the repository's coarse-versus-refined A/B benchmark.
``all``
    Run all applicable modes in that order.

The coordinator resolves one Python interpreter first and uses that exact
interpreter for dependency checks and every child process. This avoids the
common Windows problem where ``pip`` installs into a different Python than
the one used to launch a script.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REQUIRED_MODULES = ("numpy", "PIL", "torch", "torchvision", "timm")


def default_v3_checkpoint() -> Path:
    """Return the first v3 checkpoint layout present in this checkout."""
    candidates = (
        ROOT / "v3_outputs" / "best_v3_fpn.pth",
        ROOT / "v3_outputs" / "best" / "best_v3_fpn.pth",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _configure_console() -> None:
    """Keep Windows console output from failing on Vietnamese messages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


class RunnerError(RuntimeError):
    pass


def _path(value: Path) -> Path:
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


def _python_candidates() -> Iterable[Path]:
    # Prefer the project environment, then the parent workspace environment
    # used by this checkout, and only then the interpreter running the runner.
    names = ("python.exe", "python") if os.name == "nt" else ("python", "python3")
    for folder in (ROOT / ".venv" / "Scripts", ROOT / ".venv" / "bin", ROOT.parent / ".venv" / "Scripts", ROOT.parent / ".venv" / "bin"):
        for name in names:
            yield folder / name
    yield Path(sys.executable)
    found = shutil.which("python")
    if found:
        yield Path(found)


def resolve_python(requested: Optional[Path]) -> Path:
    if requested is not None:
        candidate = _path(requested)
        if not candidate.is_file():
            raise RunnerError(f"Không tìm thấy Python được chỉ định: {candidate}")
        return candidate.resolve()
    seen: set[str] = set()
    for candidate in _python_candidates():
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    raise RunnerError("Không tìm thấy python.exe. Hãy truyền --python D:\\path\\to\\python.exe")


def check_dependencies(python: Path) -> None:
    code = (
        "import importlib.util; "
        "mods=" + repr(REQUIRED_MODULES) + "; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print('missing=' + ','.join(missing))"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise RunnerError(f"Không kiểm tra được dependencies bằng {python}:\n{result.stderr.strip()}")
    line = next((item for item in result.stdout.splitlines() if item.startswith("missing=")), "missing=")
    missing = [item for item in line.removeprefix("missing=").split(",") if item]
    if missing:
        install = f'"{python}" -m pip install numpy pillow torch torchvision timm opencv-python'
        raise RunnerError(
            f"Python {python} còn thiếu: {', '.join(missing)}\n"
            f"Cài đúng vào môi trường này bằng:\n{install}"
        )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RunnerError(f"{label} không tồn tại: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RunnerError(f"{label} không tồn tại: {path}")


def _add_option(command: list[str], option: str, value: Optional[Path | str]) -> None:
    if value is not None and str(value) != "":
        command.extend([option, str(value)])


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def _run(command: list[str], env: dict[str, str]) -> None:
    print(f"\n$ {_command_text(command)}")
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise RunnerError(f"{Path(command[1]).name} stopped with exit code {result.returncode}")


def _stage_demo_samples(source: Path, output_dir: Path, maximum: int) -> Path:
    """Limit demo images without assuming main_localization has --max-images."""
    if maximum <= 0:
        return source
    images = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    selected = images[:maximum]
    if len(selected) == len(images):
        return source
    staged = output_dir / "demo_samples"
    staged.mkdir(parents=True, exist_ok=True)
    for image in selected:
        target = staged / image.name
        shutil.copy2(image, target)
    return staged


def _validate_common(args: argparse.Namespace) -> None:
    if args.max_images < 0:
        raise RunnerError("--max-images phải >= 0; dùng 0 để chạy toàn bộ")
    if not 0.0 <= args.threshold <= 1.0:
        raise RunnerError("--threshold phải nằm trong [0, 1]")


def _compare_command(args: argparse.Namespace, python: Path, output_root: Path) -> list[str]:
    _require_file(args.labels, "Labels JSON")
    _require_dir(args.dataset, "Dataset root")
    _require_file(args.baseline, "Baseline checkpoint")
    _require_file(args.roi, "ROI checkpoint")
    _require_file(args.v3, "v3 checkpoint")
    command = [str(python), str(ROOT / "compare_v3_roi.py")]
    for option, value in (("--labels", args.labels), ("--dataset", args.dataset), ("--baseline", args.baseline), ("--roi", args.roi), ("--v3", args.v3)):
        _add_option(command, option, value)
    _add_option(command, "--output-dir", output_root / "compare_v3_roi")
    command.extend(["--split", args.split, "--max-images", str(args.max_images), "--selection", args.selection, "--threshold", str(args.threshold)])
    _add_option(command, "--device", args.device)
    return command


def _demo_command(args: argparse.Namespace, python: Path, output_root: Path) -> list[str]:
    _require_dir(args.samples, "Demo samples")
    _require_file(args.baseline, "Baseline checkpoint")
    _require_file(args.roi, "ROI checkpoint")
    samples = _stage_demo_samples(args.samples, output_root, args.max_images)
    command = [
        str(python), str(ROOT / "main_localization.py"),
        "--samples", str(samples),
        "--model", str(args.baseline),
        "--roi-checkpoint", str(args.roi),
        "--output", str(output_root / "demo" / "localization_results.json"),
        "--threshold", str(args.threshold),
    ]
    _add_option(command, "--device", args.device)
    _add_option(command, "--calibration", args.calibration)
    _add_option(command, "--mesh", args.mesh)
    if args.no_uncertainty:
        command.append("--no-uncertainty")
    if args.sequence:
        command.append("--sequence")
    return command


def _benchmark_command(args: argparse.Namespace, python: Path, output_root: Path) -> list[str]:
    _require_file(args.labels, "Labels JSON")
    _require_dir(args.dataset, "Dataset root")
    _require_file(args.baseline, "Baseline checkpoint")
    _require_file(args.roi, "ROI checkpoint")
    if args.benchmark_model == "v3":
        raise RunnerError(
            "benchmark_ab.py hiện benchmark baseline-versus-ROI. "
            "Dùng --mode compare để so sánh checkpoint v3, hoặc khôi phục benchmark_v3.py."
        )
    if args.max_images:
        print("WARNING: benchmark_ab.py hiện tại không có --max-images; benchmark sẽ chạy toàn bộ source test/fire.")
    command = [
        str(python), str(ROOT / "benchmark_ab.py"),
        "--output", str(output_root / "benchmark" / "benchmark_ab.json"),
        "--threshold", str(args.threshold),
    ]
    _add_option(command, "--device", args.device)
    _add_option(command, "--calibration", args.calibration)
    _add_option(command, "--mesh", args.mesh)
    return command


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compare", "demo", "benchmark", "all"), default="all")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--python", dest="python_executable", type=Path, default=None)
    parser.add_argument("--roi", "--roi-checkpoint", dest="roi", type=Path, default=ROOT / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--v3", type=Path, default=default_v3_checkpoint())
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "fire-detection-from-cctv")
    parser.add_argument("--samples", type=Path, default=ROOT / "fire-samples")
    parser.add_argument("--baseline", "--model", dest="baseline", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "week6_run")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--selection", choices=("even", "diverse"), default="even")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-model", choices=("baseline", "v3"), default="baseline")
    parser.add_argument("--sequence", action="store_true")
    args = parser.parse_args()

    for name in ("labels", "dataset", "samples", "baseline", "roi", "v3", "calibration", "mesh", "output_dir"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, _path(value))
    _validate_common(args)
    python = resolve_python(args.python_executable)
    print(f"python={python}")
    check_dependencies(python)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    commands: list[list[str]] = []
    if args.mode in ("compare", "all"):
        commands.append(_compare_command(args, python, output_root))
    if args.mode in ("demo", "all"):
        commands.append(_demo_command(args, python, output_root))
    if args.mode in ("benchmark", "all"):
        commands.append(_benchmark_command(args, python, output_root))
    for command in commands:
        _run(command, env)
    print(f"\nWeek 6 finished. Outputs: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
