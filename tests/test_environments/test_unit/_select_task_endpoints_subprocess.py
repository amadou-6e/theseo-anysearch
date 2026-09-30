"""Standalone helper run in a fresh subprocess to check select_task_endpoints
candidate order does not depend on Python's per-process hash randomization.
Not a pytest test module itself; invoked by test_ifc_bench_export.py.
"""

import json

import numpy as np

from theseo_anysearch.environments.ifc_bench_export import select_task_endpoints

# Hub H connected to three leaves A, B, C arranged as an equilateral
# triangle around it, so every leaf pair ties on distance -- exactly the
# case where a hash-order-dependent tie-break would leak into the result.
graph = {
    "H": {"A", "B", "C"},
    "A": {"H"},
    "B": {"H"},
    "C": {"H"},
}
centers = {
    "H": np.array([0.0, 0.0, 0.0]),
    "A": np.array([1.0, 0.0, 0.0]),
    "B": np.array([-0.5, 0.8660254, 0.0]),
    "C": np.array([-0.5, -0.8660254, 0.0]),
}

candidates, rejections = select_task_endpoints((), graph, centers)
print(json.dumps({"candidates": candidates, "rejections": rejections}))
