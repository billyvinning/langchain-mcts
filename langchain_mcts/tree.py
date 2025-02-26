from abc import ABC
from typing import Any, ClassVar, Generic, Optional, TypeVar, get_args

import graphviz
from pydantic import BaseModel, Field
from typing_extensions import Self


class AbstractNode(BaseModel):
    @property
    def node_attrs(self) -> dict[str, Any]:
        return self.model_dump()


NodeT = TypeVar("NodeT", bound=AbstractNode)


class AbstractTree(BaseModel, Generic[NodeT], ABC):
    nodes: dict[int, NodeT] = Field(default_factory=dict)
    children: dict[int, set[int]] = Field(default_factory=dict)
    parents: dict[int, int] = Field(default_factory=dict)

    _ROOT_INDEX: ClassVar[int] = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractTree):
            return False
        return (
            self.nodes == other.nodes
            and self.children == other.children
            and self.parents == other.parents
        )

    @classmethod
    def from_root_node(cls, node_kwargs: dict[str, Any], **kwargs) -> Self:
        out = cls(**kwargs)
        out.add_node(parent=None, **node_kwargs)
        return out

    @classmethod
    def create_node(cls, **kwargs) -> NodeT:
        _, node_cls = get_args(cls.model_fields["nodes"].annotation)
        return node_cls.model_validate(kwargs)

    @property
    def root(self) -> NodeT:
        return self.nodes[self._ROOT_INDEX]

    @property
    def leaves(self) -> set[int]:
        return set(self.nodes) - set(self.parents.values())

    def add_node(
        self,
        *,
        parent_ix: Optional[int] = None,
        **node_kwargs,
    ) -> int:
        if parent_ix is None and self.nodes:
            msg = "There can only be one root node."
            raise ValueError(msg)

        ix = len(self.nodes)
        if parent_ix is not None:
            if parent_ix not in self.nodes:
                msg = f"Parent node ({parent_ix}) does not exist."
                raise ValueError(msg)
            if parent_ix not in self.children:
                self.children[parent_ix] = set()
            self.children[parent_ix].add(ix)
            self.parents[ix] = parent_ix
        self.nodes[ix] = self.create_node(**node_kwargs)
        return ix

    def _get_extra_node_attrs(self, ix: int, node: NodeT) -> dict[str, Any]:  # noqa: ARG002
        return {}

    def to_graphviz(self) -> graphviz.Digraph:
        def _get_node_label(attrs: dict[str, Any]) -> str:
            lines = []
            for k, v in attrs.items():
                line = f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
                lines.append(line)
            return "\n".join(lines)

        dot = graphviz.Digraph()
        # Create nodes.
        for ix, node in self.nodes.items():
            node_attrs = node.node_attrs | self._get_extra_node_attrs(ix, node)
            node_label = _get_node_label(node_attrs)
            dot.node(str(ix), node_label)
        # Create edges.
        for parent, children in self.children.items():
            for child in children:
                dot.edge(str(parent), str(child))
        return dot
