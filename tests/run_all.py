#!/usr/bin/env python3
"""Run every offline suite in this directory and report which ones failed.

The suites are plain scripts that exit non-zero on failure, so each is run in its
own interpreter: one suite crashing, or leaving a Qt thread behind at shutdown,
cannot take the rest of the run with it. Suites named ``*_live.py`` need hardware
on the bench and are skipped here; run those by hand before an experiment.

This is what CI runs, and it is equally usable locally::

    python tests/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"run_all.py", "__init__.py"}


def suites():
    return sorted(
        name for name in os.listdir(HERE)
        if name.endswith(".py") and name not in SKIP
        and not name.endswith("_live.py")
    )


def main():
    # Every suite already asks for the offscreen platform, but a suite that only
    # imports the scheduler model has no reason to, and CI has no display.
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The suites print em dashes and arrows; a redirected Windows console is
    # cp1252 by default and would turn a passing check into an encoding error.
    env.setdefault("PYTHONUTF8", "1")

    results = []
    for name in suites():
        print(f"\n=== {name} " + "=" * max(0, 60 - len(name)), flush=True)
        code = subprocess.call([sys.executable, os.path.join(HERE, name)],
                               env=env)
        results.append((name, code))

    failed = [name for name, code in results if code]
    print("\n" + "=" * 66)
    for name, code in results:
        verdict = "PASS" if not code else f"FAIL ({code})"
        print(f"  {verdict:>10}  {name}")
    print(f"\n{len(results) - len(failed)}/{len(results)} suites passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
