#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Developer tools for the cpWER scoring path. Not part of the shipped library.

Three jobs a maintainer needs and a user never does:

  rescore   re-score an archived manifest and diff it against a frozen baseline. This is the
            regression gate: inference is NOT re-run, because the generate script does not set a
            seed, so two runs of one checkpoint differ by more than any change under test.
  verify    cross-check this implementation against a reference scorer, when one is available.
  fixture   regenerate the golden fixtures the offline tests assert against.

`verify` needs a reference checkout, passed with --reference-root. No path is baked in: this file
ships in a public repository and must not name an internal host.

Usage::

    python scripts/speechlm2/cpwer_devtools.py rescore --manifest run.jsonl --baseline frozen.jsonl
    python scripts/speechlm2/cpwer_devtools.py fixture --out tests/.../fixtures
    python scripts/speechlm2/cpwer_devtools.py verify --reference-root /path/to/reference/checkout
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemo.collections.speechlm2.parts.metrics.cpwer_report import (  # noqa: E402
    REFERENCE_AXES,
    cpwer_metrics_dict,
    format_cpwer_report,
)
from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import CpWERScoringConfig, score_rows  # noqa: E402

# Keys whose value is a rate, compared with a tolerance; everything else must match exactly.
_RATE_KEYS = ("cpwer", "notag_ceiling")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_rescore = sub.add_parser("rescore", help="re-score a manifest and diff against a frozen baseline")
    p_rescore.add_argument("--manifest", type=Path, required=True)
    p_rescore.add_argument("--baseline", type=Path, help="frozen manifest to diff against; omit to just report")
    p_rescore.add_argument("--normalizer", default=None, help="override the cpWER normalizer family")
    p_rescore.add_argument("--reference-axes", action="store_true", help="score at the reference corner")

    p_fixture = sub.add_parser("fixture", help="regenerate the golden fixtures")
    p_fixture.add_argument("--out", type=Path, required=True)
    p_fixture.add_argument("--cases", type=int, default=52)

    p_verify = sub.add_parser("verify", help="cross-check against a reference scorer checkout")
    p_verify.add_argument("--reference-root", type=Path, required=True)
    p_verify.add_argument("--pairs", type=int, default=8000)

    args = parser.parse_args()
    return {"rescore": _rescore, "fixture": _fixture, "verify": _verify}[args.command](args)


def _rescore(args) -> int:
    """Re-score a manifest offline and diff it against a frozen baseline."""
    rows = _read_jsonl(args.manifest)
    cfg = CpWERScoringConfig(**(REFERENCE_AXES if args.reference_axes else {}))
    if args.normalizer:
        cfg.cpwer_normalizer = args.normalizer
    per_row, corpus, subsets = score_rows(rows, cfg)
    metrics = cpwer_metrics_dict(corpus, subsets, cfg)
    print(format_cpwer_report(metrics, cfg))

    if not args.baseline:
        return 0
    base_rows = _read_jsonl(args.baseline)
    if len(base_rows) != len(rows):
        print(f"\nDIFF: row count {len(rows)} vs baseline {len(base_rows)}")
        return 1
    mismatches = []
    for i, (scored, base) in enumerate(zip(per_row, base_rows)):
        for key, value in scored.items():
            if key not in base:
                continue
            if any(key.endswith(r) for r in _RATE_KEYS):
                if value is None or base[key] is None:
                    if value != base[key]:
                        mismatches.append((i, key, value, base[key]))
                elif abs(value - base[key]) > 1e-9:
                    mismatches.append((i, key, value, base[key]))
            elif value != base[key]:
                mismatches.append((i, key, value, base[key]))
    print(f"\nper-row diff vs {args.baseline.name}: {len(mismatches)} mismatches over {len(rows)} rows")
    for row_index, key, got, want in mismatches[:10]:
        print(f"  row {row_index} {key}: {got!r} != {want!r}")
    return 1 if mismatches else 0


def _fixture(args) -> int:
    """Regenerate the golden fixtures the offline tests assert against.

    Cases are generated from a fixed seed and a fixed vocabulary, so regenerating on another machine
    produces the same file. Three cases are built to abstain, because key ABSENCE on an abstained
    row is a property the tests assert and a fixture of only scorable rows would never exercise it.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args.cases)
    cfg = CpWERScoringConfig(**REFERENCE_AXES)
    per_row, corpus, subsets = score_rows(cases, cfg)

    golden = [{**case, **scored} for case, scored in zip(cases, per_row)]
    _write_jsonl(args.out / "cpwer_golden.jsonl", golden)
    _write_json(args.out / "cpwer_golden_metrics.json", cpwer_metrics_dict(corpus, subsets, cfg))
    _write_json(args.out / "cpwer_golden_provenance.json", _provenance(cfg, len(cases)))

    abstained = sum(1 for row in golden if "cpwer_errors" not in row)
    print(f"wrote {len(golden)} cases to {args.out} ({abstained} abstained, {len(subsets)} subsets)")
    return 0


def _verify(args) -> int:
    """Cross-check per-session counts against a reference scorer, if its checkout is importable."""
    root = args.reference_root.resolve()
    if not root.exists():
        print(f"reference root not found: {root}")
        return 2
    try:
        reference = _load_reference(root)
    except Exception as exc:  # noqa: BLE001 - a dev tool reports rather than traces
        print(f"could not load a reference scorer from {root}: {type(exc).__name__}: {exc}")
        return 2

    cfg = CpWERScoringConfig(**REFERENCE_AXES)
    random.seed(0)
    mismatches = 0
    for _ in range(args.pairs):
        ref_text, hyp_text = _random_pair()
        ours = score_rows([{"text_raw": ref_text, "pred_text_raw": hyp_text}], cfg)[0][0]
        theirs = reference(ref_text, hyp_text)
        if theirs is None:
            continue
        if ours.get("cpwer_errors") != theirs.get("cpwer_errors") or ours.get("cpwer_ref_words") != theirs.get(
            "cpwer_ref_words"
        ):
            mismatches += 1
    print(f"verify: {args.pairs - mismatches}/{args.pairs} pairs agree on (errors, ref_words)")
    return 1 if mismatches else 0


def _build_cases(count: int) -> list:
    """Deterministic cases spanning the behaviours the offline tests care about."""
    random.seed(20260917)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    subsets = ("set-a", "set-b", "set-c")
    cases = []
    for i in range(count):
        n_spk = 1 + i % 4
        ref = " ".join(f"<spk:{s}> " + " ".join(random.choices(words, k=random.randint(1, 6))) for s in range(n_spk))
        if i % 7 == 0:  # a hypothesis that drops a speaker
            hyp = " ".join(f"<spk:{s}> " + " ".join(random.choices(words, k=random.randint(1, 6))) for s in range(1))
        elif i % 11 == 0:  # untagged hypothesis
            hyp = " ".join(random.choices(words, k=random.randint(1, 8)))
        else:
            hyp = " ".join(
                f"<spk:{s}> " + " ".join(random.choices(words, k=random.randint(1, 6))) for s in range(n_spk)
            )
        case = {"id": f"case-{i:03d}", "text_raw": ref, "pred_text_raw": hyp}
        if i % 13:  # one value absent on a few cases, so a missing subset is exercised
            case["subset_for_metrics"] = subsets[i % len(subsets)]
        cases.append(case)
    # Force a few abstains: a reference with no tag at all, under a config that discards untagged
    # reference text, parses to zero streams.
    for i in range(3):
        cases[i] = {
            "id": f"abstain-{i:03d}",
            "text_raw": " ".join(random.choices(words, k=4)),
            "pred_text_raw": "<spk:0> " + " ".join(random.choices(words, k=4)),
            "subset_for_metrics": subsets[i % len(subsets)],
        }
    return cases


def _random_pair() -> tuple:
    words = "one two three four five six seven eight".split()
    n = random.randint(1, 3)
    ref = " ".join(f"<spk:{s}> " + " ".join(random.choices(words, k=random.randint(1, 5))) for s in range(n))
    hyp = " ".join(f"<spk:{s}> " + " ".join(random.choices(words, k=random.randint(1, 5))) for s in range(n))
    return ref, hyp


def _load_reference(root: Path):
    """Return a callable scoring one pair with the reference implementation, or raise.

    Loads leaf modules by path rather than importing the package, because a reference checkout
    usually pulls in unrelated heavy dependencies through its package ``__init__``.
    """
    import importlib.util
    import types

    for pkg, rel in (
        ("nemo_skills", ""),
        ("nemo_skills.evaluation", "/evaluation"),
        ("nemo_skills.evaluation.metrics", "/evaluation/metrics"),
        ("nemo_skills.evaluation.evaluator", "/evaluation/evaluator"),
        ("nemo_skills.evaluation.metrics.utils", "/evaluation/metrics/utils"),
    ):
        module = types.ModuleType(pkg)
        module.__path__ = [str(root / "nemo_skills") + rel]
        sys.modules[pkg] = module
    sys.path.insert(0, str(root))

    spec = importlib.util.spec_from_file_location(
        "nemo_skills.evaluation.evaluator.audio", root / "nemo_skills/evaluation/evaluator/audio.py"
    )
    audio = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audio
    spec.loader.exec_module(audio)

    def score(ref_text: str, hyp_text: str):
        return audio.evaluate_msasr(ref_text, hyp_text, speaker_token="spk", cpwer_normalization="chime8")

    return score


def _provenance(cfg: CpWERScoringConfig, n_cases: int) -> dict:
    import hashlib

    import kaldialign
    import scipy

    from nemo.collections.asr.parts.utils.chime8_spelling_data import ENGLISH_SPELLING, PRE_ENGLISH_SPELLING

    def table_hash(table):
        return hashlib.sha256(json.dumps(table, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    return {
        "cases": n_cases,
        "axes": {name: getattr(cfg, name) for name in sorted(vars(cfg))},
        "python": sys.version.split()[0],
        "kaldialign": kaldialign.__version__,
        "scipy": scipy.__version__,
        "chime8_english_spelling_sha256": table_hash(ENGLISH_SPELLING),
        "chime8_pre_english_spelling_sha256": table_hash(PRE_ENGLISH_SPELLING),
    }


def _read_jsonl(path: Path) -> list:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
