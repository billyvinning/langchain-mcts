import math
import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import cached_property
from typing import Any, Callable, Final, Generic, TypeVar

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from langchain_mcts.tree import AbstractNode, AbstractTree


class TreePolicy(str, Enum):
    UCT = auto()
    UCB = auto()


def uct(
    q: float,
    n: int,
    n_parent: int,
    c: float,
) -> float:
    return (q / n) + (c * math.sqrt(2 * n_parent / n))


def ucb(
    q: float,
    n: int,
    n_parent: int,
    c: float,
    # eps: float = 1e-5,
) -> float:
    exploitation = q / n
    exploration = math.sqrt(math.log(n_parent) / n)
    # exploration =  math.sqrt((math.log(n_parent) + 1) / n + 1e-5)
    return exploitation + (c * exploration)


_TREE_POLICY_FNS: Final[
    dict[
        TreePolicy,
        Callable[[float, int, int, float], float],
    ]
] = {TreePolicy.UCT: uct, TreePolicy.UCB: ucb}


class AbstractState(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def next_states(self) -> set[Self]:
        pass

    @property
    @abstractmethod
    def is_terminal_state(self) -> bool:
        pass

    @property
    @abstractmethod
    def reward(self) -> int | float:
        pass


StateT = TypeVar("StateT", bound=AbstractState)


class MonteCarloSearchNode(AbstractNode, Generic[StateT]):
    state: StateT
    n: int = 0
    q: float = 0.0

    @cached_property
    def remaining_states(self) -> list[StateT]:
        return list(self.state.next_states())

    @property
    def is_expanded(self) -> bool:
        return len(self.remaining_states) == 0


NodeT = TypeVar("NodeT", bound=MonteCarloSearchNode)


class MonteCarloSearchTree(AbstractTree[NodeT], Generic[NodeT, StateT]):
    invert_reward: bool = False
    c: float = math.sqrt(2)
    tree_policy: TreePolicy = TreePolicy.UCB

    @classmethod
    def from_root_state(cls, state_kwargs: dict[str, Any], **kwargs) -> Self:
        return cls.from_root_node({"state": state_kwargs}, **kwargs)

    def _get_extra_node_attrs(self, ix: int, node: NodeT) -> dict[str, Any]:
        out = super()._get_extra_node_attrs(ix, node)
        try:
            parent_ix = self.parents[ix]
        except KeyError:
            return out
        parent_node = self.nodes[parent_ix]

        q = node.q
        n = node.n
        n_parent = parent_node.n

        ucb = _TREE_POLICY_FNS[self.tree_policy](q, n, n_parent, 0)
        return out | {self.tree_policy.name.lower(): round(ucb, 3)}

    def _select(self) -> int:
        """Select the most promising node to expand, according to the tree policy."""
        ix = self._ROOT_INDEX
        while not (node := self.nodes[ix]).state.is_terminal_state:
            # Always explore unexplored states if we can.
            if not node.is_expanded:
                return ix
            # Else, if we've already explored all states, go to the best child.
            ix = self._tree_policy(ix, c=self.c)
        return ix

    def _tree_policy(self, parent_ix: int, *, c: float) -> int:
        """Find the most promising child node."""

        def _tree_policy_fn(parent_ix: int, child_ix: int, *, c: float) -> float:
            parent_node = self.nodes[parent_ix]
            child_node = self.nodes[child_ix]

            q = child_node.q
            n = child_node.n
            n_parent = parent_node.n

            return _TREE_POLICY_FNS[self.tree_policy](q, n, n_parent, c)

        return max(
            self.children[parent_ix],
            key=lambda child_ix: _tree_policy_fn(
                parent_ix=parent_ix,
                child_ix=child_ix,
                c=c,
            ),
        )

    def _expand(self, ix: int) -> int:
        """Expand the selected node."""
        if self.nodes[ix].is_expanded:
            return ix
        remaining_states = self.nodes[ix].remaining_states
        selected_state_ix = self._expansion_policy(remaining_states)
        child_state = remaining_states.pop(selected_state_ix)
        return self.add_node(parent=ix, state=child_state.model_dump())

    @staticmethod
    def _expansion_policy(states: list[StateT]) -> int:
        """Select the next state to add to the tree."""
        return random.randint(0, len(states) - 1)

    @staticmethod
    def _rollout_policy(states: list[StateT]) -> StateT:
        """Select the next state to explore."""
        return states[random.randint(0, len(states) - 1)]

    def _simulate(self, ix: int) -> float:
        """Perform rollouts until a terminal state is reached."""
        current_state = self.nodes[ix].state
        while not current_state.is_terminal_state:
            next_states = list(current_state.next_states())
            current_state = self._rollout_policy(next_states)
        return current_state.reward

    def _backpropagate(self, child_ix: int, reward: float) -> None:
        """Backpropagate scores up to all parent states."""
        self.nodes[child_ix].n += 1
        self.nodes[child_ix].q += (-reward) if self.invert_reward else reward

        if (parent_ix := self.parents.get(child_ix)) is not None:
            self._backpropagate(parent_ix, reward)

    def step(self) -> None:
        parent_node_ix = self._select()
        child_node_ix = self._expand(parent_node_ix)
        reward = self._simulate(child_node_ix)
        self._backpropagate(child_node_ix, reward)

    def best_next_state(self, max_leaves: int) -> StateT:
        for _ in range(max_leaves):
            self.step()
        return self.nodes[self._tree_policy(self._ROOT_INDEX, c=0)].state
