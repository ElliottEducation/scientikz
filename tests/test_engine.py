"""pgfmath must compute exactly what Python computes, for the same formula."""
import math
import os
import re
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Expr, ExprError  # noqa: E402

FORMULAS = ["x^2", "0.5x^2", "-x^2", "x^3 - 3x", "2^x", "e^(-x^2)", "(x+1)(x-1)", "x^2^0.5*0+1",
            "sin(x)", "3cos(2x)", "x sin(x)", "1/(1+x^2)", "abs(x) - 1", "exp(x/2)", "-(x-1)^3", "2*-x",
            "sqrt(x^2+1)", "tan(x/3)"]
XS = [-2.5, -1.5, -0.4, 0.3, 1.2, 2.7]


def pgf_values(exprs):
    lines = ["\\documentclass{article}\\usepackage{tikz}\\begin{document}"]
    for i, f in enumerate(exprs):
        opts = "\\pgfkeys{/pgf/trig format=rad}" if f.uses_trig else "\\pgfkeys{/pgf/trig format=deg}"
        for j, x in enumerate(XS):
            lines.append(f"{opts}\\def\\x{{{x}}}\\pgfmathparse{{{f.pgf()}}}\\typeout{{R:{i}:{j}:\\pgfmathresult}}")
    lines.append("\\end{document}")
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "t.tex"), "w").write("\n".join(lines))
        out = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=d,
                             capture_output=True, text=True).stdout
    return {(int(i), int(j)): float(v) for i, j, v in re.findall(r"R:(\d+):(\d+):(-?[\d.]+)", out)}


def test_pgf_matches_python():
    exprs = [Expr(t) for t in FORMULAS]
    got = pgf_values(exprs)
    for i, f in enumerate(exprs):
        for j, x in enumerate(XS):
            want = f(x)
            if math.isnan(want):
                continue
            assert abs(got[(i, j)] - want) <= 2e-3 * max(1, abs(want)), (FORMULAS[i], x, got[(i, j)], want)


@pytest.mark.parametrize("bad", ["__import__('os')", "x.real", "\\input{/etc/passwd}", "\\write18{rm -rf /}",
                                 "open('x')", "a+b", "[1,2]", "x;y", "lambda: 1", "x" * 200, "", "sin(x, 2)"])
def test_rejects_unsafe_or_invalid(bad):
    with pytest.raises(ExprError):
        Expr(bad)


def test_normalisation():
    assert Expr("y = 2x^2").pgf() == "2*(\\x)^2"
    assert Expr("f(x)=3cos(2x)").pgf() == "3*cos(2*\\x)"
    assert Expr("2^3^2")(0) == 512
