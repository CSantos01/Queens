import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from scipy.spatial import Voronoi, voronoi_plot_2d


class Case:
    def __init__(self, x, y, color="white"):
        self.x = x
        self.y = y
        self.color = color
        self.queen = False
        self.is_checked = False

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.color})"


class GameBoard:
    def __init__(self, size=7, num_colors=5):
        self.size = size
        self.num_colors = num_colors
        self.board = [[Case(x, y) for x in range(size)] for y in range(size)]
        self.colors = [
            "red",
            "green",
            "blue",
            "orange",
            "yellow",
            "cyan",
            "magenta",
            "purple",
            "brown",
            "pink",
            "yellowgreen",
        ]
        random.shuffle(self.colors)
        self.colors = self.colors[:num_colors]
        self.create_voronoi_zones()

    def create_voronoi_zones(self):
        points = np.random.rand(self.num_colors, 2) * self.size
        vor = Voronoi(points)
        for i in range(self.size):
            for j in range(self.size):
                distances = [np.linalg.norm([i - p[0], j - p[1]]) for p in vor.points]
                closest_region = np.argmin(distances)
                self.board[i][j].color = self.colors[closest_region]

    def ensure_valid_board(self):
        # Ensuring each color forms a single continuous region
        color_groups = {color: set() for color in self.colors}
        for i in range(self.size):
            for j in range(self.size):
                color_groups[self.board[i][j].color].add((i, j))

        for color, positions in color_groups.items():
            visited = set()
            stack = [next(iter(positions))]
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in positions and (nx, ny) not in visited:
                        stack.append((nx, ny))
            if visited != positions:
                return self.create_voronoi_zones()

    def get_colors(self):
        return [[case.color for case in row] for row in self.board]

    def save_board(self, filename):
        with open(filename, "w") as f:
            for row in self.board:
                f.write(",".join([case.color for case in row]) + "\n")

    def load_board(self, filename):
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                for j, color in enumerate(line.strip().split(",")):
                    self.board[i][j].color = color

    def reset_queens(self):
        for row in self.board:
            for case in row:
                case.queen = False
                case.is_checked = False


class GameBoardVisualizer:
    def __init__(self, game_board):
        self.game_board = game_board
        self.size = game_board.size
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xticks(np.arange(-0.5, self.size, 1))
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_yticks(np.arange(-0.5, self.size, 1))
        self.ax.grid(color="black", linestyle="-", linewidth=2)
        self.rects = [
            [
                self.ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        edgecolor="black",
                        facecolor=self.game_board.board[i][j].color,
                    )
                )
                for j in range(self.size)
            ]
            for i in range(self.size)
        ]
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.selected_color = "red"
        self.add_color_buttons()

    def on_click(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < self.size and 0 <= y < self.size:
                self.game_board.board[y][x].color = self.selected_color
                self.rects[y][x].set_facecolor(self.selected_color)
                self.fig.canvas.draw()

    def select_color(self, event):
        self.selected_color = event
        self.fig.canvas.draw()

    def add_color_buttons(self):
        axcolor = plt.axes([0.90, 0.05, 0.1, 0.2], frameon=False)
        self.radio = RadioButtons(axcolor, self.game_board.colors)
        self.radio.on_clicked(self.select_color)

    def show(self):
        plt.show()


class Queen:
    """
    Class to represent a queen in the game board.
    """

    def __init__(self, game_board, x, y):
        self.game_board = game_board
        case = self.game_board.board[y][x]
        self.x = x
        self.y = y
        self.color = case.color
        # First check if the case is already checked by another queen
        if not case.is_checked:
            case.queen = True

            # Check the row, column and the squares diagonally
            if 0 <= x - 1 < self.game_board.size and 0 <= y - 1 < self.game_board.size:
                self.game_board.board[y - 1][x - 1].is_checked = True
            if 0 <= x - 1 < self.game_board.size and 0 <= y + 1 < self.game_board.size:
                self.game_board.board[y + 1][x - 1].is_checked = True
            if 0 <= x + 1 < self.game_board.size and 0 <= y - 1 < self.game_board.size:
                self.game_board.board[y - 1][x + 1].is_checked = True
            if 0 <= x + 1 < self.game_board.size and 0 <= y + 1 < self.game_board.size:
                self.game_board.board[y + 1][x + 1].is_checked = True
            for i in range(self.game_board.size):
                if 0 <= y + i < self.game_board.size:
                    self.game_board.board[y + i][x].is_checked = True
                if 0 <= y - i < self.game_board.size:
                    self.game_board.board[y - i][x].is_checked = True
                if 0 <= x + i < self.game_board.size:
                    self.game_board.board[y][x + i].is_checked = True
                if 0 <= x - i < self.game_board.size:
                    self.game_board.board[y][x - i].is_checked = True

            # Also check all the cases with the same color as the queen
            for i in range(self.game_board.size):
                for j in range(self.game_board.size):
                    if self.game_board.board[j][i].color == self.color:
                        self.game_board.board[j][i].is_checked = True

        else:
            print("The case is already checked by another queen")

    def __repr__(self):
        return f"Queen({self.x}, {self.y}, {self.color})"


def is_legal(board, x, y):
    case = board.board[y][x]
    if case.color == "white":
        return False
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return any(
        0 <= x + dx < board.size
        and 0 <= y + dy < board.size
        and board.board[y + dy][x + dx].color == case.color
        for dx, dy in directions
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Queens game")
    parser.add_argument("--size", type=int, default=7)
    args = parser.parse_args()

    board = GameBoard(size=args.size, num_colors=args.size)
    board.ensure_valid_board()
    visualizer = GameBoardVisualizer(board)
    visualizer.show()
