from __future__ import annotations

import re
from typing import Any

from formulaic.errors import FormulaParsingError
from formulaic.parser import DefaultOperatorResolver
from formulaic.parser.types import Factor, Operator, OrderedSet, Term

__all__ = ["FamilyOperatorResolver"]

# Allow bare func or np.<name> (single dot) for function wrapping
_FUNC_WRAP_RE = re.compile(r"^\s*(?P<func>(?:[A-Za-z_]\w*|np\.[A-Za-z_]\w*))\s*\(\s*(?P<inner>.*)\s*\)\s*$")
# Strict NAME[a:b] regex that requires start and end indices
_NAME_INDEX_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<s>\d+)\s*:\s*(?P<e>\d+)\s*\]\s*$")


def _strip_func_wrap(expr: str) -> tuple[str, list[str]]:
    funcs = []
    while True:
        m = _FUNC_WRAP_RE.match(expr)
        if not m:
            break
        funcs.append(m.group("func"))
        expr = m.group("inner").strip()
    return expr, funcs


def _parse_family_nested(term: Term) -> tuple[str, int, int, list[str]]:
    """Peel nested functions like ``f1(f2(...(NAME[a:b])...))`` and extract components.

    Returns the base variable-family name, the start/end of the slice, and any
    wrapping functions (outer-to-inner order).
    """
    s = term.factors[0].expr.strip()
    s, funcs = _strip_func_wrap(s)

    m = _NAME_INDEX_RE.match(s)
    if not m:
        raise FormulaParsingError("Only NAME[a:b] or nested FUNC(...(NAME[a:b])...) are supported.")
    s_i, e_i = int(m.group("s")), int(m.group("e"))
    if s_i > e_i:
        raise FormulaParsingError("Range must be ascending: start <= end.")
    return m.group("name"), s_i, e_i, funcs


def _expand_family(arg: OrderedSet[Term], *, context: Any | None = None) -> OrderedSet[Term]:
    """Expand a term like ``@X[1:5]`` into ``X_1, X_2, ..., X_5``."""
    name, s, e, funcs = _parse_family_nested(next(iter(arg)))
    labels = [f"{name}_{i}" for i in range(s, e + 1)]
    if not funcs:
        return OrderedSet(Term([Factor(lbl, eval_method="lookup")]) for lbl in labels)

    # Re-wrap from inner to outer if functions are present
    out = []
    for lbl in labels:
        expr = lbl
        for fn in reversed(funcs):  # funcs is outer->inner; build inside-out
            expr = f"{fn}({expr})"
        out.append(Term([Factor(expr, eval_method="python")]))
    return OrderedSet(out)


class FamilyOperatorResolver(DefaultOperatorResolver):
    """Adds the ``@`` prefix operator for expanding variable families.

    Example
    -------
    ``Formula("y ~ x + @X[1:5] + @scvi[1:3] + @np.log1p(scvi[1:2])", _parser=parser)``
    """

    @property
    def operators(self):  # noqa: D102
        return [
            *super().operators,
            Operator("@", arity=1, precedence=350, associativity="right", fixity="prefix", to_terms=_expand_family),
        ]
