"""Exercise both installed-style CLIs and reject unsupported/invalid inputs."""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location(
    "make_demo", Path(__file__).resolve().parents[1] / "scripts/make_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def main():
    fo, par = map(Path, sys.argv[1:])
    env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="100",
               OMP_MAX_ACTIVE_LEVELS="3")
    with tempfile.TemporaryDirectory(prefix="ppsc-cli-") as tmp:
        root = Path(tmp)
        demo.write_fixture(root)
        common = ["--dataset-path", root / "base.f32", "--query-path", root / "query.f32",
                  "--groundtruth-path", root / "gt.i64", "--suco-index-path", root / "index.bin",
                  "--dataset-size", "64", "--query-size", "4", "--dim", "8", "--k", "5",
                  "--subspace-num", "2", "--kmeans-num-centroid", "2", "--sc-fixed-cells", "4",
                  "--beta", "1", "--use-srht", "false", "--path-oram", "false", "--threads", "16"]

        def call(exe, extra=(), success=True):
            command = list(map(str, [exe, *common, *extra]))
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
            if (result.returncode == 0) != success:
                raise AssertionError((command, result.returncode, result.stdout, result.stderr))
            return result

        def audit(path):
            return dict(line.split("=", 1) for line in path.read_text().splitlines())

        call(fo, ["--output-path", root / "fo.ids", "--audit-path", root / "fo.audit"])
        expected = (root / "fo.ids").read_bytes()
        for p in range(1, 6):
            call(par, ["--workers-per-subspace", p, "--output-path", root / "par.ids",
                       "--audit-path", root / "par.audit"])
            assert (root / "par.ids").read_bytes() == expected
            data = audit(root / "par.audit")
            assert data["variant"] == "FO-par" and data["workers_per_subspace"] == str(p)
            assert data["intra_query_threads"] == str(2 * p)
            assert data["privacy_verified"] == data["access_pattern_protection"] == "false"
            assert data["memory_backend"] == "direct" and data["path_oram"] == "false"
            assert float(data["recall"]) == 1.0
            for key in ("average_query_ms", "average_query_latency_ms", "index_ms"):
                assert math.isfinite(float(data[key])) and float(data[key]) > 0
        call(par, ["--query-groups", "1", "--output-path", root / "one.ids"])
        assert (root / "one.ids").read_bytes() == expected
        for extra in (["--path-oram", "true"], ["--path-oram", "typo"],
                      ["--dataset-size", "-1"], ["--dataset-size", "64junk"],
                      ["--beta", "nan"], ["--beta", "1.1"], ["--query-groups", "-1"],
                      ["--workers-per-subspace", "6"], ["--query-groups", "100"],
                      ["--threads", "1"], ["--padded-bucket-capacity", "1"],
                      ["--suco-index-path", root / "missing.bin"]):
            call(par, extra, success=False)
        (root / "short.f32").write_bytes(b"\x00")
        call(fo, ["--dataset-path", root / "short.f32"], success=False)
        result = subprocess.run([str(fo)], env=env, capture_output=True, timeout=5)
        assert result.returncode != 0
    print("CLI tests passed: five parallel settings, exact IDs/recall, audit, and failures")


if __name__ == "__main__":
    main()
