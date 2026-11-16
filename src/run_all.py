"""Run the whole analysis in the right order.

Tuning comes first because train.py and card_fraud.py read its output, and
explain.py needs the models those two save. The full run takes about an hour,
most of it hyperparameter search.

Usage:
    python src/run_all.py
    python src/run_all.py --skip-tuning    # reuse reports/*_best_params.json
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STEPS = [
    ["eda.py"],
    ["tuning.py", "--task", "both"],
    ["train.py"],
    ["card_fraud.py"],
    ["explain.py", "--task", "both"],
    ["sensitivity.py"],
    ["imbalance.py"],
    ["demand_forecast.py"],
    ["fraud_detection.py"],
    # Smoke test that the saved model scores raw bookings end to end.
    ["predict.py", "--input-path", "data/raw/hotel_bookings.csv"],
]


def main(skip_tuning: bool) -> None:
    total = time.time()
    for step in STEPS:
        if skip_tuning and step[0] == "tuning.py":
            print("\n### Skipping tuning.py (reusing saved hyperparameters)")
            continue
        print(f"\n### {' '.join(step)}", flush=True)
        start = time.time()
        # Unbuffered, so progress shows up live when the output goes to a log file.
        subprocess.run([sys.executable, str(SRC / step[0]), *step[1:]], cwd=ROOT, check=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"### {step[0]} finished in {(time.time() - start) / 60:.1f} min", flush=True)
    print(f"\nAll steps finished in {(time.time() - total) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    main(args.skip_tuning)
