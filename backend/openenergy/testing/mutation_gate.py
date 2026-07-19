from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class MutationScore:
    killed: int
    survived: int

    @property
    def percent(self) -> float:
        total = self.killed + self.survived
        return 0.0 if total == 0 else self.killed / total * 100.0


def parse_results(output: str) -> MutationScore:
    """Parse mutmut's stable textual status labels, ignoring skipped mutants."""
    killed = len(re.findall(r"\b(?:killed|timeout)\b", output, flags=re.IGNORECASE))
    survived = len(re.findall(r"\bsurvived\b", output, flags=re.IGNORECASE))
    return MutationScore(killed=killed, survived=survived)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    minimum = float(args[0]) if args else 90.0
    result = subprocess.run(
        ["mutmut", "results", "--all", "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        return result.returncode

    score = parse_results(result.stdout)
    print(
        f"mutation score: {score.percent:.1f}% "
        f"({score.killed} killed, {score.survived} survived)"
    )
    return 0 if score.percent >= minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
