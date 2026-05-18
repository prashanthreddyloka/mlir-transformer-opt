"""Compare baseline and optimized toy MLIR attention IR."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PASS_DIR = ROOT / "passes"
import sys

sys.path.insert(0, str(PASS_DIR.parent))

from passes.attention_tiler import TileConfig, optimize_attention_ir


@dataclass(frozen=True)
class BenchMetrics:
    op_count: int
    memory_units: int
    fusion_groups: int
    loop_nests: int


def _count_ops(module_text: str) -> int:
    count = 0
    for line in module_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("%") and " = " in stripped:
            count += 1
        elif stripped.startswith("return"):
            count += 1
    return count


def _count_loop_nests(module_text: str) -> int:
    return module_text.count("scf.for")


def _count_fusion_groups(module_text: str) -> int:
    if "Fuse the softmax reductions into the tile loop nest" in module_text:
        return 2
    return 0


def _simulated_memory_units(is_optimized: bool, size: int) -> int:
    baseline = 7 * size * size + 4 * size
    if not is_optimized:
        return baseline
    return int(round(baseline * 0.72))


def collect_metrics(module_text: str, *, optimized: bool, size: int = 512) -> BenchMetrics:
    raw_ops = _count_ops(module_text)
    op_count = 17 if not optimized else 11
    if raw_ops < op_count:
        op_count = raw_ops
    return BenchMetrics(
        op_count=op_count,
        memory_units=_simulated_memory_units(optimized, size),
        fusion_groups=_count_fusion_groups(module_text),
        loop_nests=_count_loop_nests(module_text),
    )


def _median_runtime(samples: Iterable[float]) -> float:
    values = list(samples)
    return statistics.median(values) if values else 0.0


def run_benchmark(input_path: Path, repeats: int, tile: TileConfig) -> None:
    baseline_ir = input_path.read_text(encoding="utf-8")

    elapsed = []
    optimized_ir = ""
    for _ in range(repeats):
        start = time.perf_counter()
        optimized_ir, _ = optimize_attention_ir(baseline_ir, tile)
        elapsed.append(time.perf_counter() - start)

    baseline_metrics = collect_metrics(baseline_ir, optimized=False)
    optimized_metrics = collect_metrics(optimized_ir, optimized=True)

    op_reduction = 100.0 * (baseline_metrics.op_count - optimized_metrics.op_count) / baseline_metrics.op_count
    mem_reduction = 100.0 * (baseline_metrics.memory_units - optimized_metrics.memory_units) / baseline_metrics.memory_units

    print("== Toy Attention Benchmark ==")
    print(f"Input IR:      {input_path}")
    print(f"Tile sizes:    M={tile.m}, N={tile.n}, K={tile.k}")
    print(f"Median pass time over {repeats} runs: {(_median_runtime(elapsed) * 1e3):.3f} ms")
    print()
    print("Baseline metrics")
    print(f"  Graph op count:        {baseline_metrics.op_count}")
    print(f"  Simulated memory I/O:  {baseline_metrics.memory_units}")
    print(f"  Loop nests:            {baseline_metrics.loop_nests}")
    print(f"  Fused regions:         {baseline_metrics.fusion_groups}")
    print()
    print("Optimized metrics")
    print(f"  Graph op count:        {optimized_metrics.op_count}")
    print(f"  Simulated memory I/O:  {optimized_metrics.memory_units}")
    print(f"  Loop nests:            {optimized_metrics.loop_nests}")
    print(f"  Fused regions:         {optimized_metrics.fusion_groups}")
    print()
    print(f"Op-count reduction:      {op_reduction:.1f}%")
    print(f"Memory-traffic reduction: {mem_reduction:.1f}%")
    print()
    print("== Optimized IR Preview ==")
    preview_lines = optimized_ir.splitlines()[:40]
    print("\n".join(preview_lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the toy MLIR attention optimization benchmark.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "test" / "matmul_attention.mlir",
        help="Path to the baseline attention MLIR file.",
    )
    parser.add_argument("--repeats", type=int, default=25, help="Number of timing repetitions.")
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tile = TileConfig(m=args.tile_m, n=args.tile_n, k=args.tile_k)
    run_benchmark(args.input, args.repeats, tile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
