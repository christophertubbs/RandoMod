"""
@TODO: Put a module wide description here
"""
from __future__ import annotations

import pathlib
import random

RESOURCE_DIRECTORY = pathlib.Path(__file__).parent / "resources"
RANDOM_VALUE_SEED = 987654321

def apply_seed():
    random.seed(RANDOM_VALUE_SEED)