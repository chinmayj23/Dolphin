from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from tsd import run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(repo / "configs" / "world_bank.json"))
    parser.add_argument("--workspace-root", default=str(workspace))
    args = parser.parse_args()
    run_pipeline(args.config, args.workspace_root)


if __name__ == "__main__":
    main()

