"""
boolean.py -- the small language the writers write their entity formulas in.

Grammar (case of the operators does not matter):
    formula := or_expr
    or_expr := and_expr ( OR and_expr )*
    and_expr := not_expr ( AND not_expr )*
    not_expr := NOT not_expr | atom
    atom := TERM | '(' formula ')'
    TERM := an entity key (letters, digits, underscore, dot, hyphen)
            or a quoted title: "Armed Forces of Ukraine" (anything but a
            double quote, inside double quotes; straight or curly quotes)

Examples:
    e1 AND (e2 OR e3)
    (e1 OR e2) AND e3 AND NOT e4
    e1 AND NOT (e2 OR e3)
    ("Ukraine" OR "Armed Forces of Ukraine") AND "Black market" AND NOT "Video game"

The query file always stores the key form (the titles live in the formula's
'entities' map); the quoted form is what the writers write in their sheet, and
queries/import_formulas.py turns it into keys (quoted_to_keys below). Quotes
are needed because a title may contain an operator word ("Trinidad and
Tobago") or parentheses ("Mercury (planet)").

Meaning, as sets of documents:
    e1            documents whose chosen node mentions entity e1
    A AND B       documents in both
    A OR B        documents in either
    NOT A         documents not in A (the universe minus A)

A formula that is only a negation (NOT e1) or an OR with a bare negation
(e1 OR NOT e2) would return almost the whole archive; it is accepted by the
grammar but the graph system refuses it unless --allow-wide-not is passed,
because the writers' instruction says NOT is used to remove documents from a
positive set (A AND NOT B), which is how Boolean searchers use it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

QUOTES = "\"“”„‟"  # straight and the curly quotes Word and Excel produce
TOKEN_RE = re.compile(r"\s*(?:(\()|(\))|(AND|OR|NOT)(?![A-Za-z0-9_.\-])|([A-Za-z0-9_.\-]+)|[" + QUOTES + r"]([^" + QUOTES + r"]*)[" + QUOTES + r"])", re.IGNORECASE)


@dataclass(frozen=True)
class Term:
    key: str


@dataclass(frozen=True)
class Not:
    child: "Node"


@dataclass(frozen=True)
class And:
    children: tuple["Node", ...]


@dataclass(frozen=True)
class Or:
    children: tuple["Node", ...]


Node = Union[Term, Not, And, Or]


class FormulaError(ValueError):
    pass


def tokenize(text: str) -> list[tuple[str, str]]:
    pos = 0
    out = []
    text = text.strip()
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise FormulaError(f"cannot read formula at position {pos}: {text[pos:pos+20]!r}")
        lp, rp, op, term, quoted = m.groups()
        if lp:
            out.append(("(", "("))
        elif rp:
            out.append((")", ")"))
        elif op:
            out.append(("OP", op.upper()))
        elif term is not None:
            out.append(("TERM", term))
        else:
            title = " ".join(quoted.split())
            if not title:
                raise FormulaError(f"empty quoted title at position {pos}")
            out.append(("TERM", title))
        pos = m.end()
    return out


def quoted_to_keys(text: str) -> tuple[str, dict[str, str]]:
    """The sheet form with quoted titles -> (the key form, {key: title}).
    Keys are e1, e2, ... in order of first appearance; the same title written
    twice gets the same key. A formula with no quotes is returned unchanged
    with an empty map (it is already in key form)."""
    toks = tokenize(text)
    if not any(k == "TERM" and (" " in v or not re.fullmatch(r"[A-Za-z0-9_.\-]+", v)) for k, v in toks) and not re.search("[" + QUOTES + "]", text):
        return text.strip(), {}
    keys: dict[str, str] = {}
    out = []
    for kind, val in toks:
        if kind == "TERM":
            if val not in keys:
                keys[val] = f"e{len(keys) + 1}"
            out.append(keys[val])
        elif kind == "OP":
            out.append(val)
        else:
            out.append(val)
    s = " ".join(out).replace("( ", "(").replace(" )", ")")
    return s, {k: t for t, k in keys.items()}


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> Node:
        if not self.toks:
            raise FormulaError("empty formula")
        node = self.or_expr()
        if self.i != len(self.toks):
            raise FormulaError(f"unexpected token {self.peek()[1]!r}")
        return node

    def or_expr(self) -> Node:
        parts = [self.and_expr()]
        while self.peek() == ("OP", "OR"):
            self.take()
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def and_expr(self) -> Node:
        parts = [self.not_expr()]
        while self.peek() == ("OP", "AND"):
            self.take()
            parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def not_expr(self) -> Node:
        if self.peek() == ("OP", "NOT"):
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind, val = self.take()
        if kind == "TERM":
            return Term(val)
        if kind == "(":
            node = self.or_expr()
            if self.take() != (")", ")"):
                raise FormulaError("missing closing parenthesis")
            return node
        if kind is None:
            raise FormulaError("formula ends too early")
        raise FormulaError(f"unexpected token {val!r}")


def parse(text: str) -> Node:
    return _Parser(tokenize(text)).parse()


def terms(node: Node) -> list[str]:
    """All entity keys in the formula, in order of first appearance."""
    out: list[str] = []

    def walk(n):
        if isinstance(n, Term):
            if n.key not in out:
                out.append(n.key)
        elif isinstance(n, Not):
            walk(n.child)
        else:
            for c in n.children:
                walk(c)
    walk(node)
    return out


def positive_terms(node: Node) -> list[str]:
    """Entity keys that appear outside any NOT (the ones a document must
    mention to be retrieved). Used by the coordination-level system and by
    the failure protocol."""
    out: list[str] = []

    def walk(n, negated):
        if isinstance(n, Term):
            if not negated and n.key not in out:
                out.append(n.key)
        elif isinstance(n, Not):
            walk(n.child, not negated)
        else:
            for c in n.children:
                walk(c, negated)
    walk(node, False)
    return out


def is_wide_not(node: Node) -> bool:
    """True when the formula's top level is a negation, or an OR has a
    negated branch: such formulas retrieve 'everything except', which is not
    a search. Inside an AND a NOT is fine (A AND NOT B)."""
    if isinstance(node, Not):
        return True
    if isinstance(node, Or):
        return any(isinstance(c, Not) or is_wide_not(c) for c in node.children)
    if isinstance(node, And):
        positives = [c for c in node.children if not isinstance(c, Not)]
        if not positives:
            return True
        return any(is_wide_not(c) for c in positives)
    return False


def evaluate(node: Node, term_sets: dict[str, set[str]], universe: set[str]) -> set[str]:
    """Set semantics. term_sets maps an entity key to the set of document ids
    that mention it (missing key = empty set). universe is needed for NOT."""
    if isinstance(node, Term):
        return set(term_sets.get(node.key, set()))
    if isinstance(node, Not):
        return universe - evaluate(node.child, term_sets, universe)
    if isinstance(node, And):
        # Evaluate negations last and intersect progressively (cheap).
        pos = [c for c in node.children if not isinstance(c, Not)]
        neg = [c for c in node.children if isinstance(c, Not)]
        if pos:
            acc = evaluate(pos[0], term_sets, universe)
            for c in pos[1:]:
                acc &= evaluate(c, term_sets, universe)
        else:
            acc = set(universe)
        for c in neg:
            acc -= evaluate(c.child, term_sets, universe)
        return acc
    if isinstance(node, Or):
        acc: set[str] = set()
        for c in node.children:
            acc |= evaluate(c, term_sets, universe)
        return acc
    raise TypeError(node)


def satisfied_terms(node: Node, term_sets: dict[str, set[str]], doc_id: str) -> list[str]:
    """Positive terms whose set contains doc_id (coordination level)."""
    return [t for t in positive_terms(node) if doc_id in term_sets.get(t, set())]


def drop_terms(node: Node, drop: set[str]) -> "Node | None":
    """The formula without the terms in drop (used to run a formula with today's pages only, without the
    older names of a page that the import adds inside an OR). An OR or AND that loses every child disappears;
    one that keeps one child becomes that child. None when nothing is left."""
    if isinstance(node, Term):
        return None if node.key in drop else node
    if isinstance(node, Not):
        c = drop_terms(node.child, drop)
        return None if c is None else Not(c)
    kids = tuple(c for c in (drop_terms(ch, drop) for ch in node.children) if c is not None)
    if not kids:
        return None
    if len(kids) == 1:
        return kids[0]
    return type(node)(kids)


def to_string(node: Node) -> str:
    if isinstance(node, Term):
        return node.key
    if isinstance(node, Not):
        return f"NOT {to_string(node.child)}" if isinstance(node.child, Term) else f"NOT ({to_string(node.child)})"
    if isinstance(node, And):
        return " AND ".join(to_string(c) if not isinstance(c, Or) else f"({to_string(c)})" for c in node.children)
    if isinstance(node, Or):
        return " OR ".join(to_string(c) for c in node.children)
    raise TypeError(node)
