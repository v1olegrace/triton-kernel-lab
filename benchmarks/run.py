"""Compatibility wrapper for the installed ``tklab-bench`` command."""

from tklab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
