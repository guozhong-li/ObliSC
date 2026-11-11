# ObliSC: FO and FO-par

ObliSC is a subspace-collision search framework. This C++17 source release
provides its fixed-budget search base, **ObliSC-FO**, and parallel implementation,
**ObliSC-FO-par**. Both use the same public padded index, candidate budgets,
collision scores, refinement count, and deterministic tie rules.

The full paper, including the appendix, is available in the [technical report](ObliSC-TR.pdf).

This directory contains only the two search implementations, shared utilities,
build configuration, and tests. It does not include competing methods, datasets,
generated indexes, experiment logs, server credentials, or deployment machinery.

## Framework Overview

![ObliSC overall framework](assets/framework.png)

ObliSC separates offline index preparation from online query processing.
ObliSC-FO and ObliSC-FO-par share the same index and search stages:

1. **Shared index.** Load the pretrained centroids and assignments, organize
   items into subspace IMI cells, and pad each cell to its subspace's capacity.
2. **Candidate activation.** Project the query, order cells by centroid-pair
   distance, and scan a fixed number of cells per subspace. Items receive
   collision votes from the visited cells.
3. **Refinement.** Select candidates by collision score and activation rank,
   evaluate their distances to the query, and return the top-k results with
   deterministic ID tie breaking.

ObliSC-FO processes a query's subspaces sequentially. ObliSC-FO-par parallelizes
subspace processing, voting, candidate selection, and distance evaluation. With
identical inputs and parameters, parallel execution preserves the ordered results; it
changes the execution schedule, not the index or search budgets. The packaged
backend and its security limitations are described under **Implementation Scope**.

## Build

Requirements: a C++17 compiler, CMake >= 3.16, and OpenMP. Python 3 is optional
for the CLI smoke test and toy example; neither search executable depends on it.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure
```

Executables: `build/ppsc_fo` and `build/ppsc_fo_par`. Optionally enable
`-DPPSC_NATIVE=ON` for native CPU optimizations; rebuild for a different CPU.

On macOS, install an OpenMP runtime and point CMake to it if necessary:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/path/to/lib/libomp.dylib \
  -DCMAKE_CXX_FLAGS="-I/path/to/include"
```

## Layout

```text
include/oblisc/oblisc.hpp  Shared declarations
src/                      FO pipeline, index, I/O, and metrics
src/parallel/             FO-par pipeline
scripts/make_demo.py       Toy data and index generator
tests/                    Equivalence and CLI tests
assets/framework.png      Framework overview
```

## Inputs and Offline Index

Inputs are headerless binary files in native byte order (little-endian on the
evaluated servers):

* Base and query files: row-major `float32`, with `N*D` and `Q*D` elements.
* Ground truth: row-major `int64`, with `Q*k` zero-based identifiers.
* Public index: `Ns*2*C*(subspace_dim/2)` `float32` centroids, followed by
  `Ns*2*N` `int32` assignments, ordered by subspace then half-subspace.

`--suco-index-path` is required. The loader pads cells but does **not** train
K-means. Supply a SuCo centroid/assignment index with matching dimensions,
centroids, SRHT setting, and seed 42; the external trainer is not bundled.
For angular search, consistently normalize base and query vectors beforehand.

## Quick Start

Run both implementations on a deterministic toy fixture (functional testing
only, not benchmark data):

```bash
python3 scripts/make_demo.py --out data/demo
mkdir -p results
for method in fo fo_par; do
  ./build/ppsc_${method} \
    --dataset-path data/demo/base.f32 --query-path data/demo/query.f32 \
    --groundtruth-path data/demo/gt.i64 --suco-index-path data/demo/index.bin \
    --dataset-size 64 --query-size 4 --dim 8 --k 5 --subspace-num 2 \
    --kmeans-num-centroid 2 --sc-fixed-cells 4 --beta 1 \
    --use-srht false --path-oram false --threads 8 \
    --output-path results/${method}.ids --audit-path results/${method}.audit
done
diff results/fo.ids results/fo_par.ids
```

For real data, replace the paths and dimensions, keeping both methods' settings
identical. Reference defaults: `k=50`, `Ns=8`, `C=50`, `t=2`, `alpha=0.03`,
`beta=0.003`, and SRHT enabled. `--sc-fixed-cells` sets `hat-alpha` (default 100).
Pass dataset-specific settings explicitly; use `--help` for all options.

## Parallelism and Timing

* `--threads T`: total worker budget, capped at 100; set it explicitly.
* `--workers-per-subspace p`: FO-par workers per subspace, default 2 (range 1-5).
  Each query uses `Ns*p <= T` workers; concurrency is `min(Q, floor(T/(Ns*p)))`.
* `--query-groups G`: optionally reduce FO-par query concurrency; 0 uses the default.

Keep inputs, budgets, warmups, compiler, and CPU binding matched for comparisons.
Example OpenMP settings on Linux:

```bash
export OMP_THREAD_LIMIT=100 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=3
export OMP_PROC_BIND=spread,spread,close OMP_PLACES=threads
```

`average query time` is batch wall time divided by query count, whereas
`average query latency` measures individual query duration. Stage timings are
latency components, not additive batch-time components. Reported index time adds
padding time to `--upstream-index-ms` (supply only measured training time);
omitting it excludes training. IDs are saved as ordered top-k rows, audits as
`key=value` text. Candidates break score ties by activation rank then ID; final
results break distance ties by ID.

## Implementation Scope

This release uses **direct memory** and rejects `--path-oram true`. Its heap
selection, direct accesses, and legacy counters do not establish access-pattern
privacy: audits record `access_pattern_protection=false` and `privacy_verified=false`.
SEV capability detection is not remote attestation. Tests verify FO/FO-par
candidate and ordered-result equivalence, not cryptographic security.
