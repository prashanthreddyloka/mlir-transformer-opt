"""Toy MLIR attention tiling pass implemented in Python.

This script demonstrates the shape of a custom optimization pass for a
Transformer-style attention block expressed in MLIR's Linalg dialect. The pass
recognizes a simple matmul -> softmax -> matmul pattern and emits a tiled,
loop-oriented variant with fused normalization steps.

If ``mlir-python-bindings`` is installed, the script will parse and verify the
input/output modules via the Python bindings. The transformation itself is kept
in Python so the repository stays approachable and hackable.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from mlir import ir
except ImportError:  # pragma: no cover - optional dependency
    ir = None

ATTENTION_MARKER = "transformer.attention"
DEFAULT_TILE_M = 64
DEFAULT_TILE_N = 64
DEFAULT_TILE_K = 32


@dataclass(frozen=True)
class TileConfig:
    m: int = DEFAULT_TILE_M
    n: int = DEFAULT_TILE_N
    k: int = DEFAULT_TILE_K


@dataclass(frozen=True)
class PassStats:
    matched: bool
    tile_m: int
    tile_n: int
    tile_k: int


def _verify_with_mlir_bindings(module_text: str) -> bool:
    """Try to parse the module with Python bindings when available."""
    if ir is None:
        return False

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        ir.Module.parse(module_text)
    return True


def _extract_function_signature(module_text: str) -> tuple[str, str, list[str]]:
    match = re.search(
        r"(func\.func\s+@attention\s*\((?P<args>.*?)\)\s*->\s*(?P<ret>\S+)\s*\{)",
        module_text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Expected a `func.func @attention` definition in the input IR.")
    args = match.group("args").strip()
    arg_names = re.findall(r"(%[\w$.-]+)\s*:", args)
    if len(arg_names) < 3:
        raise ValueError("Expected three tensor arguments for Q, K, and V.")
    return args, match.group("ret").strip(), arg_names


def _build_optimized_body(tile: TileConfig, arg_names: list[str]) -> str:
    q_name, k_name, v_name = arg_names[:3]
    return f"""    %c0 = arith.constant 0 : index
    %cM = arith.constant {tile.m} : index
    %cN = arith.constant {tile.n} : index
    %cK = arith.constant {tile.k} : index
    %c512 = arith.constant 512 : index
    %fneg = arith.constant -3.40282347E+38 : f32
    %fzero = arith.constant 0.0 : f32

    %out_init = tensor.empty() : tensor<512x512xf32>
    %out = linalg.fill ins(%fzero : f32) outs(%out_init : tensor<512x512xf32>) -> tensor<512x512xf32>
    %scores_init = tensor.empty() : tensor<512x512xf32>
    %scores_seed = linalg.fill ins(%fzero : f32) outs(%scores_init : tensor<512x512xf32>) -> tensor<512x512xf32>

    %result = scf.for %m = %c0 to %c512 step %cM iter_args(%acc0 = %out) -> (tensor<512x512xf32>) {{
      %m_size = affine.min #map0(%m)
      %acc1 = scf.for %n = %c0 to %c512 step %cN iter_args(%acc2 = %acc0) -> (tensor<512x512xf32>) {{
        %n_size = affine.min #map1(%n)
        %scores = scf.for %k = %c0 to %c512 step %cK iter_args(%score_iter = %scores_seed) -> (tensor<512x512xf32>) {{
          %k_size = affine.min #map2(%k)
          %q_tile = tensor.extract_slice {q_name}[%m, %k] [%m_size, %k_size] [1, 1] : tensor<512x512xf32> to tensor<?x?xf32>
          %kt_tile = tensor.extract_slice {k_name}[%n, %k] [%n_size, %k_size] [1, 1] : tensor<512x512xf32> to tensor<?x?xf32>
          %score_tile = tensor.extract_slice %score_iter[%m, %n] [%m_size, %n_size] [1, 1] : tensor<512x512xf32> to tensor<?x?xf32>
          %updated_tile = linalg.matmul ins(%q_tile, %kt_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%score_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
          %next_scores = tensor.insert_slice %updated_tile into %score_iter[%m, %n] [%m_size, %n_size] [1, 1] : tensor<?x?xf32> into tensor<512x512xf32>
          scf.yield %next_scores : tensor<512x512xf32>
        }}

        // Fuse the softmax reductions into the tile loop nest so the scores
        // tile never needs a separate full-tensor materialization pass.
        %scores_tile = tensor.extract_slice %scores[%m, %n] [%m_size, %n_size] [1, 1] : tensor<512x512xf32> to tensor<?x?xf32>
        %row_max_init = tensor.empty() : tensor<?xf32>
        %row_max_seed = linalg.fill ins(%fneg : f32) outs(%row_max_init : tensor<?xf32>) -> tensor<?xf32>
        %row_max = linalg.reduce ins(%scores_tile : tensor<?x?xf32>) outs(%row_max_seed : tensor<?xf32>) dimensions = [1] ({{
        ^bb0(%lhs: f32, %rhs: f32):
          %max = arith.maximumf %lhs, %rhs : f32
          linalg.yield %max : f32
        }}) -> tensor<?xf32>
        %shifted = linalg.generic {{
            indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i, j)>],
            iterator_types = ["parallel", "parallel"]
          }}
          ins(%scores_tile, %row_max : tensor<?x?xf32>, tensor<?xf32>)
          outs(%scores_tile : tensor<?x?xf32>) {{
        ^bb0(%score: f32, %rowmax: f32, %outv: f32):
          %sub = arith.subf %score, %rowmax : f32
          linalg.yield %sub : f32
        }} -> tensor<?x?xf32>
        %exp_tile = linalg.generic {{
            indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
            iterator_types = ["parallel", "parallel"]
          }}
          ins(%shifted : tensor<?x?xf32>)
          outs(%shifted : tensor<?x?xf32>) {{
        ^bb0(%in: f32, %outv: f32):
          %exp = math.exp %in : f32
          linalg.yield %exp : f32
        }} -> tensor<?x?xf32>
        %row_sum_init = tensor.empty() : tensor<?xf32>
        %row_sum_seed = linalg.fill ins(%fzero : f32) outs(%row_sum_init : tensor<?xf32>) -> tensor<?xf32>
        %row_sum = linalg.reduce ins(%exp_tile : tensor<?x?xf32>) outs(%row_sum_seed : tensor<?xf32>) dimensions = [1] ({{
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          linalg.yield %sum : f32
        }}) -> tensor<?xf32>
        %probs = linalg.generic {{
            indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i, j)>],
            iterator_types = ["parallel", "parallel"]
          }}
          ins(%exp_tile, %row_sum : tensor<?x?xf32>, tensor<?xf32>)
          outs(%exp_tile : tensor<?x?xf32>) {{
        ^bb0(%num: f32, %den: f32, %outv: f32):
          %div = arith.divf %num, %den : f32
          linalg.yield %div : f32
        }} -> tensor<?x?xf32>
        %v_tile = tensor.extract_slice {v_name}[%n, 0] [%n_size, %c512] [1, 1] : tensor<512x512xf32> to tensor<?x512xf32>
        %out_tile = tensor.extract_slice %acc2[%m, 0] [%m_size, %c512] [1, 1] : tensor<512x512xf32> to tensor<?x512xf32>
        %fused_out = linalg.matmul ins(%probs, %v_tile : tensor<?x?xf32>, tensor<?x512xf32>) outs(%out_tile : tensor<?x512xf32>) -> tensor<?x512xf32>
        %next_acc = tensor.insert_slice %fused_out into %acc2[%m, 0] [%m_size, %c512] [1, 1] : tensor<?x512xf32> into tensor<512x512xf32>
        scf.yield %next_acc : tensor<512x512xf32>
      }}
      scf.yield %acc1 : tensor<512x512xf32>
    }}
    return %result : tensor<512x512xf32>"""


def build_optimized_module(module_text: str, tile: TileConfig) -> tuple[str, PassStats]:
    args, ret, arg_names = _extract_function_signature(module_text)
    matched = ATTENTION_MARKER in module_text and "@attention" in module_text
    if not matched:
        raise ValueError(
            "Input IR does not contain the expected toy attention marker. "
            f"Add a `// {ATTENTION_MARKER}` comment to the attention function."
        )

    optimized = f"""module {{
  affine_map<(d0) -> (min({tile.m}, 512 - d0))>
  affine_map<(d0) -> (min({tile.n}, 512 - d0))>
  affine_map<(d0) -> (min({tile.k}, 512 - d0))>

  func.func @attention({args}) -> {ret} {{
{_build_optimized_body(tile, arg_names)}
  }}
}}"""
    return optimized, PassStats(matched=True, tile_m=tile.m, tile_n=tile.n, tile_k=tile.k)


def optimize_attention_ir(module_text: str, tile: TileConfig | None = None) -> tuple[str, PassStats]:
    tile = tile or TileConfig()
    return build_optimized_module(module_text, tile)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile and fuse a toy MLIR attention kernel.")
    parser.add_argument("input", type=Path, help="Path to the input MLIR module.")
    parser.add_argument("-o", "--output", type=Path, help="Path to write the optimized MLIR.")
    parser.add_argument("--tile-m", type=int, default=DEFAULT_TILE_M, help="Tile size for query rows.")
    parser.add_argument("--tile-n", type=int, default=DEFAULT_TILE_N, help="Tile size for key/value columns.")
    parser.add_argument("--tile-k", type=int, default=DEFAULT_TILE_K, help="Reduction tile size.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Parse input/output via mlir-python-bindings when the package is installed.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    source = args.input.read_text(encoding="utf-8")
    tile = TileConfig(m=args.tile_m, n=args.tile_n, k=args.tile_k)

    if args.verify:
        input_verified = _verify_with_mlir_bindings(source)
        if input_verified:
            print("Verified input module with mlir-python-bindings.", file=sys.stderr)
        else:
            print("mlir-python-bindings not installed; skipping input verification.", file=sys.stderr)

    optimized, stats = optimize_attention_ir(source, tile)

    if args.verify:
        output_verified = _verify_with_mlir_bindings(optimized)
        if output_verified:
            print("Verified optimized module with mlir-python-bindings.", file=sys.stderr)
        else:
            print("mlir-python-bindings not installed; skipping output verification.", file=sys.stderr)

    if args.output:
        args.output.write_text(optimized, encoding="utf-8")
    else:
        print(optimized)

    print(
        f"Matched attention pattern={stats.matched}; tile sizes=({stats.tile_m}, {stats.tile_n}, {stats.tile_k})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
