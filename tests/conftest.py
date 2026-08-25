import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SIBLINGS = [
    "alloy-core",
    "alloy-diffuse",
    "alloy-field",
    "alloy-fluid",
    "alloy-macro",
    "alloy-perform",
    "alloy-morph",
    "alloy-pilot",
    "alloy-phase",
    "alloy-props",
    "alloy-lit",
    "alloy-sinter"
]

for s in SIBLINGS:
    p = os.path.join(BASE_DIR, s)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
