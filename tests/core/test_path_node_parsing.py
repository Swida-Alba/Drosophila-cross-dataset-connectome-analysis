"""Path-column parsing and the hemisphere-conserved edge filter
(``coana._parse_path_nodes`` / ``coana._path_has_unconserved_edge``).

Regression for the audit finding that replaced ``eval`` on persisted path
strings with ``ast.literal_eval``: a crafted cell must never execute, and
the three persisted shapes (list literal, '->'-joined, already a sequence)
must all keep working.
"""

from coana import _parse_path_nodes, _path_has_unconserved_edge


CONSERVED = {('A_L', 'B_L'), ('B_L', 'C_L')}


def test_parse_list_literal_string():
    assert _parse_path_nodes("['A_L', 'B_L', 'C_L']") == ['A_L', 'B_L', 'C_L']


def test_parse_arrow_joined_string():
    assert _parse_path_nodes('A_L->B_L->C_L') == ['A_L', 'B_L', 'C_L']


def test_parse_passthrough_non_string():
    nodes = ['A_L', 'B_L']
    assert _parse_path_nodes(nodes) is nodes
    assert _parse_path_nodes(None) is None


def test_parse_never_executes_code():
    # A crafted cell must degrade to the '->' fallback (yielding junk node
    # names), never execute the expression.
    malicious = "__import__('os').system('true')"
    parsed = _parse_path_nodes(malicious)
    assert isinstance(parsed, list)
    assert parsed == malicious.split('->')


def test_unconserved_edge_detection():
    assert _path_has_unconserved_edge(None, CONSERVED) is True
    assert _path_has_unconserved_edge("['A_L', 'B_L']", CONSERVED) is False
    assert _path_has_unconserved_edge("['A_L', 'B_L', 'X_L']", CONSERVED) is True
    assert _path_has_unconserved_edge('A_L->B_L->C_L', CONSERVED) is False
    assert _path_has_unconserved_edge(['A_L', 'X_L'], CONSERVED) is True
    # Whitespace around node names is tolerated.
    assert _path_has_unconserved_edge("[' A_L ', ' B_L ']", CONSERVED) is False
