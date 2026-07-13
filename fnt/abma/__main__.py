"""Enable ``python -m fnt.abma`` as the headless batch runner."""
import sys

from .run_headless import main

if __name__ == "__main__":
    sys.exit(main())
