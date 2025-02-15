import pytest

from langchain_mcts.tree.base import AbstractNode, AbstractTree


def test_imports():
    import langchain_mcts.tree as lib

    expected_imports = {"AbstractNode", "AbstractTree"}
    assert expected_imports.issubset(set(lib.__all__))


class Node(AbstractNode):
    value: float


class Tree(AbstractTree[Node]):
    pass


@pytest.fixture()
def tree() -> Tree:
    tree = Tree()
    tree.add_node(value=2)
    tree.add_node(value=3, parent=0)
    tree.add_node(value=4, parent=0)
    return tree


def test_tree(tree):
    assert {node.value for node in tree.leaves.values()} == {3, 4}
    assert all(tree.nodes[tree.parents[idx]].value == 2 for idx in tree.leaves)  # noqa: PLR2004
    assert tree.root.value == 2  # noqa: PLR2004
