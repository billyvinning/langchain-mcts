from collections import Counter
from enum import Enum, auto
from itertools import chain
from typing import Any, Iterable, TypeAlias

import pytest
from pydantic import Field
from typing_extensions import Self

from langchain_mcts.tree import (
    AbstractState,
    MonteCarloSearchNode,
    MonteCarloSearchTree,
)


class Tile(str, Enum):
    NOUGHTS = "o"
    CROSSES = "x"
    EMPTY = "-"


BOARD_NROWS = BOARD_NCOLS = 3
BoardRow: TypeAlias = tuple[Tile, Tile, Tile]
Board: TypeAlias = tuple[BoardRow, BoardRow, BoardRow]


def default_board() -> Board:
    return (
        (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
        (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
        (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
    )


class GameState(Enum):
    ONGOING = auto()
    NOUGHTS_WON = auto()
    CROSSES_WON = auto()
    DRAW = auto()


class NoughtsAndCrossesState(AbstractState):
    board: Board = Field(
        description="The board state.",
        default_factory=default_board,
    )

    def __repr__(self) -> str:
        return "\n".join(" ".join(row) for row in self.board)

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_terminal_state(self) -> bool:
        return self.state() is not GameState.ONGOING

    @property
    def tile_to_play(self) -> Tile:
        counts = Counter(chain(*self.board))
        if counts.get(Tile.CROSSES, 0) < counts.get(Tile.NOUGHTS, 0):
            return Tile.CROSSES
        return Tile.NOUGHTS

    def state(self) -> GameState:
        if self.is_tile_winner(Tile.NOUGHTS):
            return GameState.NOUGHTS_WON
        if self.is_tile_winner(Tile.CROSSES):
            return GameState.CROSSES_WON
        if Tile.EMPTY not in Counter(chain(*self.board)):
            return GameState.DRAW
        return GameState.ONGOING

    def is_tile_winner(self, tile: Tile) -> bool:
        for i in range(len(self.board)):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == tile:
                return True
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == tile:
                return True

        if self.board[0][0] == self.board[1][1] == self.board[2][2] == tile:
            return True

        if self.board[2][0] == self.board[1][1] == self.board[0][2] == tile:
            return True

        return False

    def actions(self) -> Iterable[tuple[Tile, int, int]]:
        tile = self.tile_to_play
        for i, row in enumerate(self.board):
            for j, element in enumerate(row):
                if element != Tile.EMPTY:
                    continue
                yield tile, i, j

    def next_states(self) -> set[Self]:
        return {self.take_action(*action) for action in self.actions()}

    def take_action(self, tile: Tile, i: int, j: int) -> Self:
        if (tile, i, j) not in self.actions():
            msg = "Invalid action."
            raise ValueError(msg)

        board = [list(row) for row in self.board]
        board[i][j] = tile

        return type(self)(board=tuple(tuple(row) for row in board))

    @property
    def reward(self) -> int:
        state = self.state()
        if state is GameState.CROSSES_WON:
            return -1
        if state is GameState.NOUGHTS_WON:
            return 1
        if state is GameState.DRAW:
            return 0
        msg = "Game has not terminated."
        raise ValueError(msg)


class NoughtsAndCrossesNode(MonteCarloSearchNode[NoughtsAndCrossesState]):
    @property
    def node_attrs(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "n": self.n,
            "board": f"\n{self.state}",
        }


class NoughtsAndCrossesMCTS(
    MonteCarloSearchTree[
        NoughtsAndCrossesNode,
        NoughtsAndCrossesState,
    ],
):
    pass


def test_game_mechanics():
    game = NoughtsAndCrossesState()
    assert game.tile_to_play is Tile.NOUGHTS
    assert game.state() is GameState.ONGOING

    game = game.take_action(Tile.NOUGHTS, 0, 0)
    assert game.tile_to_play is Tile.CROSSES
    assert game.state() is GameState.ONGOING

    # Cannot overwrite a tile.
    with pytest.raises(ValueError, match="Invalid action."):
        game = game.take_action(Tile.CROSSES, 0, 0)

    # Crosses must go next.
    with pytest.raises(ValueError, match="Invalid action."):
        game = game.take_action(Tile.NOUGHTS, 1, 0)

    game = game.take_action(Tile.CROSSES, 1, 0)
    assert game.tile_to_play is Tile.NOUGHTS
    assert game.state() is GameState.ONGOING

    game = game.take_action(Tile.NOUGHTS, 0, 1)
    assert game.tile_to_play is Tile.CROSSES
    assert game.state() is GameState.ONGOING

    game = game.take_action(Tile.CROSSES, 2, 1)
    assert game.tile_to_play is Tile.NOUGHTS
    assert game.state() is GameState.ONGOING

    game = game.take_action(Tile.NOUGHTS, 0, 2)
    assert game.state() is GameState.NOUGHTS_WON


@pytest.mark.parametrize(
    "n_rollouts",
    [1000],
)
def test_next_best_move(n_rollouts):
    mcts = NoughtsAndCrossesMCTS.from_root_state(
        {
            "board": (
                (Tile.CROSSES, Tile.CROSSES, Tile.EMPTY),
                (Tile.EMPTY, Tile.EMPTY, Tile.NOUGHTS),
                (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
            ),
        },
        c=2 * (2**0.5),
        invert_reward=False,
    )
    expected_best_next_state = (
        (Tile.CROSSES, Tile.CROSSES, Tile.NOUGHTS),
        (Tile.EMPTY, Tile.EMPTY, Tile.NOUGHTS),
        (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
    )
    assert mcts.best_next_state(n_rollouts).board == expected_best_next_state, len(
        mcts.nodes,
    )
