#!/usr/bin/env python3
"""Compatibility wrapper for the MetaGlyph instance-generation CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from metaglyph.create_instances import main


if __name__ == "__main__":
    sys.exit(main())
