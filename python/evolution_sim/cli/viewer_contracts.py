from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.env.viewer_contracts import render_viewer_contract_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic viewer contract constants.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("viewer/contracts.generated.mjs"),
        help="Path to the generated viewer contract module.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the tracked generated contract module is stale.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_viewer_contract_module()
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != rendered:
            temp_path = args.output.with_name(f"{args.output.name}.tmp")
            temp_path.write_text(rendered)
            raise SystemExit(
                "viewer contracts are stale; regenerate with "
                "`npm run viewer:contracts:generate` "
                f"(fresh output written to {temp_path})."
            )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
