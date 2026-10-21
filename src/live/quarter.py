"""The quarterly forward test: one command per quarter of 13F filings.

Usage:
    uv run python -m src.live.quarter 2026Q3              # grade the 2026Q2 forecast, issue 2026Q3's, commit + push
    uv run python -m src.live.quarter 2026Q2 --no-grade   # the first forecast (nothing to grade yet)
    uv run python -m src.live.quarter 2026Q1 --out-dir /tmp/rehearsal --issued-at 2026-05-15   # rehearsal, no git

For quarter q (the seed funds' quarter-q 13F filings, due 45 days after q ends):
  1. extend the pipeline to q (src.live.rebuild), unless q is already through it
  2. grade the forecast for q-1 (predictions/{q-1}*, as committed) against the q
     filings and append to predictions/scorecard.csv (src.live.grade)
  3. score every seed fund's q filing with the frozen models (src.live.predict)
  4. commit predictions/ only, tag forecast-{q}, push (--no-push to skip the push)

The forecast for q must be committed before the q+1 filings appear (early filers
start the day q+1 ends): git then shows the prediction came first, and grading
skips any fund whose q+1 filing is dated on or before the commit day. Nothing
here fits, tunes or selects a model; see src.live.frozen.
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def preflight(q: str, grade_prev: bool) -> None:
    from src.live import predict
    staged = git("diff", "--cached", "--name-only")
    if staged:
        raise SystemExit(f"other changes are staged ({staged.splitlines()[0]} …): commit or unstage them first")
    p = predict.paths(q)
    if any(git("ls-files", str(x)) for x in p.values()) or any(x.exists() for x in p.values()):
        raise SystemExit(f"a forecast for {q} already exists: forecasts are never replaced")
    if grade_prev:
        prev = predict.paths(predict.prev_quarter(q))["manifest"]
        if not git("ls-files", str(prev)):
            raise SystemExit(f"{prev} is not committed: nothing to grade (use --no-grade for the first forecast)")
    from src.backtest.engine import quarter_end
    nxt_end = quarter_end(predict.next_quarter(q))
    if date.today() > nxt_end:
        print(f"warning: {predict.next_quarter(q)} has ended ({nxt_end}), so early {predict.next_quarter(q)} filings "
              f"may already be public; those funds won't be graded")


def commit_and_push(q: str, manifest: dict, graded: bool, push: bool) -> None:
    from src.live import grade, predict
    p = predict.paths(q)
    files = [str(x) for x in p.values()] + ([str(grade.SCORECARD)] if graded else [])
    git("add", "--", *files)
    hashes = "\n".join(f"  {name}  sha256 {sha}" for name, sha in manifest["files"].items())
    msg = (f"Forecast {q}: frozen C1/C2 predictions from the {q} 13F filings\n\n"
           f"Predicts {manifest['predicts']}. Model sha256 {manifest['model_sha256'][:16]}.\n"
           f"{manifest['data']['c1_funds']} seed funds, {manifest['data']['c1_positions']:,} positions; "
           f"latest filing used {manifest['data']['latest_filing_used']}.\n{hashes}\n"
           + (f"\nGraded the {predict.prev_quarter(q)} forecast into predictions/scorecard.csv.\n" if graded else "")
           + "\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    git("commit", "-m", msg, "--", *files)
    git("tag", "-a", f"forecast-{q}", "-m", f"Forecast {q} (issued {date.today()})")
    print(f"committed {git('rev-parse', '--short', 'HEAD')} and tagged forecast-{q}")
    if push:
        git("push", "origin", "HEAD")
        git("push", "origin", f"forecast-{q}")
        print("pushed to origin")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quarter", help="the newly filed quarter, e.g. 2026Q3")
    ap.add_argument("--no-grade", action="store_true", help="don't grade the previous forecast (first run)")
    ap.add_argument("--no-push", action="store_true", help="commit and tag but don't push")
    ap.add_argument("--offline", action="store_true", help="skip download steps of the rebuild")
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if the quarter is already in the database")
    ap.add_argument("--out-dir", type=Path, help="rehearsal: write here instead of predictions/, no git")
    ap.add_argument("--issued-at", type=date.fromisoformat, help="rehearsal: issue date to grade with")
    args = ap.parse_args(argv)
    q = args.quarter
    # every src module reads db.LAST_QUARTER at import: set it before importing any of them
    os.environ["LAST_QUARTER"] = q
    from src.live import grade, predict, rebuild
    from src.securities import db

    rehearsal = args.out_dir is not None
    out_dir = args.out_dir or predict.PRED_DIR
    if rehearsal and args.out_dir.resolve() == predict.PRED_DIR.resolve():
        raise SystemExit("--out-dir is for rehearsals: point it outside predictions/")
    if not rehearsal:
        preflight(q, not args.no_grade)

    if args.rebuild or not rebuild.has_quarter(q):
        rebuild.run(q, offline=args.offline)
    else:
        print(f"{q} is already through the pipeline: no rebuild")

    graded = False
    with db.connect(read_only=True) as con:
        if not args.no_grade:
            prev = predict.prev_quarter(q)
            rows = grade.grade(con, prev, out_dir, issued=args.issued_at)
            grade.append(rows, out_dir / grade.SCORECARD.name)
            graded = True
            print(rows.filter(rows["subset"] == "all").select("task", "model", "funds", "rows", "macro_f1",
                                                              "log_loss", "ndcg_at_20"))
        manifest = predict.write(con, q, out_dir)
    print(f"wrote the {q} forecast: {manifest['data']}")
    if not rehearsal:
        commit_and_push(q, manifest, graded, push=not args.no_push)


if __name__ == "__main__":
    main(sys.argv[1:])
