from __init__ import GameBoard
from pathlib import Path

output_path = Path(__file__).parent / "boards"
output_path.mkdir(exist_ok=True)


def create_board(size=7, filename="game_board.json"):
    """Create a save a game board

    Args:
        size (int, optional): Size of the board. Also number of colors in it. Defaults to 7.
    """

    board = GameBoard(size=size, num_colors=size)
    board.save(filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate n NxN boards")
    parser.add_argument("--n", type=int, default=1, help="Number of boards to generate")
    parser.add_argument("--size", type=int, default=7, help="Size of the board")
    args = parser.parse_args()

    for i in range(args.n):
        create_board(size=args.size, filename=output_path / f"game_board_{i}.json")
        print(f"Board {i} created")
    print("All boards created")
