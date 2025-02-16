# ruff: noqa: PLR2004
from typing import Any

import pytest
from pydantic import ValidationError

from langchain_mcts.tree.base import AbstractNode, AbstractTree


def test_imports():
    import langchain_mcts.tree as lib

    expected_imports = {"AbstractNode", "AbstractTree"}
    assert expected_imports.issubset(set(lib.__all__))


class Node(AbstractNode):
    value: float


class Tree(AbstractTree[Node]):
    pass


def test_tree():
    tree = Tree()

    # Tree is empty.
    assert tree.leaves == set()

    # Cannot specify a parent that does not exist.
    with pytest.raises(ValueError, match="(?i)parent"):
        tree.add_node(value=2, parent_ix=0)

    root_ix = tree.add_node(value=2)
    assert root_ix == tree._ROOT_INDEX
    assert tree.root == tree.nodes[root_ix]
    assert tree.root.value == 2

    # Cannot add another root node.
    with pytest.raises(ValueError, match="root"):
        tree.add_node(value=4, parent_ix=None)

    # Invalid node parameters should raise a Pydantic ValidationError.
    with pytest.raises(ValidationError):
        tree.add_node(value="hello", parent_ix=0)

    # Adding these nodes should be valid.
    tree.add_node(value=3, parent_ix=0)
    tree.add_node(value=4, parent_ix=0)
    assert len(tree.nodes) == 3

    assert {tree.nodes[ix].value for ix in tree.leaves} == {3, 4}


def test_tree_comparison():
    a = Tree.from_root_node({"value": 2})

    b = Tree()
    b.add_node(value=2)

    assert a == b
    assert a != "foo"


def test_to_graphviz_empty() -> None:
    tree = Tree()
    dot = tree.to_graphviz()
    assert (
        dot.source
        == """\
digraph {
}
"""
    )


def test_to_graphviz() -> None:
    tree = Tree.from_root_node({"value": 2})
    tree.add_node(value=3, parent_ix=0)
    dot = tree.to_graphviz()
    assert '0 [label="value: 2.0"]' in dot.source
    assert '1 [label="value: 3.0"]' in dot.source
    assert "0 -> 1" in dot.source


def test_to_graphviz_extra_attrs() -> None:
    class CustomTree(AbstractTree[Node]):
        def _get_extra_node_attrs(self, ix: int, node: Node) -> dict[str, Any]:  # noqa: ARG002
            return {"ix": ix}

    tree = CustomTree.from_root_node({"value": 2})
    dot = tree.to_graphviz()
    assert '0 [label="value: 2.0\nix: 0"]' in dot.source
