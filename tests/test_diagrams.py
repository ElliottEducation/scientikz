"""Every diagram must compile with its defaults and at the edges of every parameter."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diagrams import DIAGRAMS, find  # noqa: E402
from engine import ExprError  # noqa: E402
from render import render  # noqa: E402


def variants(d):
    yield {}
    for p in d.params:
        if p.kind in ("float", "int"):
            yield {p.key: p.lo}
            yield {p.key: p.hi}
        elif p.kind == "bool":
            yield {p.key: not p.default}
        elif p.kind == "choice":
            for o in p.options:
                yield {p.key: o}


CASES = [(d.id, v) for d in DIAGRAMS.values() for v in variants(d)]


@pytest.mark.parametrize("did,values", CASES, ids=[f"{i}-{v}" for i, v in CASES])
def test_compiles(did, values):
    out = render(DIAGRAMS[did].document(values))
    assert out["pdf"].startswith(b"%PDF") and out["png"][:4] == b"\x89PNG" and b"<svg" in out["svg"][:500]


TRICKY = ["1/x", "tan(x)", "ln(x)", "sqrt(x)", "exp(x)", "x^5", "1/(x^2-1)", "abs(sin(x))", "0", "-3",
          "sin(1/x)", "x^(1/3)", "2^x", "e^(-x^2)", "(x-1)(x+2)(x-3)"]


@pytest.mark.parametrize("f", TRICKY)
def test_function_graph_tricky(f):
    render(DIAGRAMS["function"].document({"f": f, "xmin": "-5", "xmax": "5"}))


@pytest.mark.parametrize("f,a,b", [("x^2", "0", "2"), ("sin(x)", "0", "pi"), ("exp(-x)", "-1", "3"),
                                   ("1/(1+x^2)", "-2", "2"), ("-x", "1", "3")])
def test_integral_value_and_compile(f, a, b):
    import math
    want = {"x^2": 8 / 3, "sin(x)": 2.0, "exp(-x)": math.e - math.exp(-3),
            "1/(1+x^2)": 2 * math.atan(2), "-x": -4.0}[f]
    code = DIAGRAMS["integral"].code({"f": f, "a": a, "b": b})
    shown = float(code.split("\\approx ")[1].split("$")[0])
    assert abs(shown - want) < 1e-3
    render(DIAGRAMS["integral"].document({"f": f, "a": a, "b": b}))


@pytest.mark.parametrize("did,values", [
    ("function", {"f": "x^2", "xmin": "3", "xmax": "1"}),
    ("function", {"f": "sqrt(x)", "xmin": "-5", "xmax": "-1"}),
    ("integral", {"f": "1/x", "a": "-1", "b": "1"}),
    ("tangent", {"f": "abs(x)", "x0": "5"}),
    ("normal", {"sigma": "0"}),
    ("function", {"f": "\\input{/etc/passwd}"}),
])
def test_bad_input_is_friendly(did, values):
    with pytest.raises(ExprError):
        DIAGRAMS[did].code(values)


@pytest.mark.parametrize("query,did,expect", [
    ("斜面 45度", "incline", {"theta": 45.0}),
    ("inclined plane", "incline", {}),
    ("y = x^3 - 3x", "function", {"f": "y = x^3 - 3x"}),
    ("单位根 n=8", "roots_of_unity", {"n": 8}),
    ("抛体 v0=30 theta=60", "projectile", {"v0": 30.0, "theta": 60.0}),
    ("正态分布", "normal", {}),
    ("开普勒轨道 e=0.3", "kepler_orbit", {"e": 0.3}),
    ("x^2 的切线", "tangent", {"f": "x^2"}),
])
def test_search(query, did, expect):
    d, values = find(query)
    assert d is not None and d.id == did
    for k, v in expect.items():
        assert values.get(k) == v, (query, values)
