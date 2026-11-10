"""Make tiny deterministic, exhaustive-search fixtures, not benchmark data."""

import argparse
from pathlib import Path
import struct


def write_fixture(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n, dim, ns, c, q, k = 64, 8, 2, 2, 4, 5
    base = [[float((i * 3 + j * 5) % 11) for j in range(dim)] for i in range(n)]
    queries = [base[i] for i in (0, 3, 7, 15)]
    gt = [sorted(range(n), key=lambda i: (sum((a - b) ** 2 for a, b in
          zip(base[i], query)), i))[:k] for query in queries]
    for name, rows, kind in (("base.f32", base, "f"), ("query.f32", queries, "f"),
                             ("gt.i64", gt, "q")):
        with (out / name).open("wb") as stream:
            for row in rows:
                stream.write(struct.pack("<" + kind * len(row), *row))
    with (out / "index.bin").open("wb") as stream:
        centroids = [float(i % 2) for i in range(ns * 2 * c * (dim // ns // 2))]
        stream.write(struct.pack("<" + "f" * len(centroids), *centroids))
        for subspace in range(ns):
            for half in range(2):
                row = [((i // (half + 1)) + subspace) % c for i in range(n)]
                stream.write(struct.pack("<" + "i" * n, *row))
    return n, q, dim, k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("Created toy fixture (N,Q,D,k):", write_fixture(args.out))
