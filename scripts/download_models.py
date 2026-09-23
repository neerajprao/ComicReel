"""Download model weights for configured pipeline backends.

Stub: actual download logic is added alongside each backend implementation
in later phases. Run with no downloads performed yet.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights for ComicReel backends.")
    parser.add_argument(
        "--profile", default="lightweight", help="Which set of backends to download weights for."
    )
    parser.add_argument("--stage", default=None, help="Only download weights for this stage.")
    args = parser.parse_args()
    print(f"[stub] Would download models for profile={args.profile} stage={args.stage}")
    print("No downloads have been implemented yet — this is a placeholder for later phases.")


if __name__ == "__main__":
    main()
