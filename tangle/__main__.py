"""`python -m tangle` -- the console script and the module entry point are one function."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
