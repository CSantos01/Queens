import matplotlib.pyplot as plt
import numpy as np
from ortools.sat.python import cp_model

class ColoredQueensSolver:
    def __init__(self, n):
        self.n = n
        self.model = cp_model.CpModel()
        self.board = [[self.model.NewIntVar(0, n - 1, f'cell_{i}_{j}') for j in range(n)] for i in range(n)]
        self.used_colors = [self.model.NewBoolVar(f'color_{c}') for c in range(n)]
    
    def visualize_board(self, board=None, title="Initial Board"):
        """ Draws the board using Matplotlib. """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xticks(np.arange(self.n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linewidth=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        colors = plt.cm.get_cmap("tab10", self.n)  # Generate N distinct colors
        board_data = np.random.randint(0, self.n, (self.n, self.n)) if board is None else board

        for i in range(self.n):
            for j in range(self.n):
                color_id = board_data[i][j]
                ax.add_patch(plt.Rectangle((j, self.n - 1 - i), 1, 1, color=colors(color_id)))
                ax.text(j + 0.5, self.n - 1 - i + 0.5, str(color_id), va='center', ha='center', color="white", fontsize=12)

        plt.title(title)
        plt.show()

    def add_queen_constraints(self):
        """ Ensure no two queens of the same color attack each other """
        for color in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if i < self.n - 1:
                        for k in range(i + 1, self.n):
                            self.model.Add(self.board[i][j] != self.board[k][j])  # Column
                    if j < self.n - 1:
                        for k in range(j + 1, self.n):
                            self.model.Add(self.board[i][j] != self.board[i][k])  # Row
                    for k in range(1, self.n - max(i, j)):  # Diagonal (↘)
                        self.model.Add(self.board[i][j] != self.board[i + k][j + k])
                    for k in range(1, min(i + 1, self.n - j)):  # Diagonal (↙)
                        self.model.Add(self.board[i][j] != self.board[i - k][j + k])

    def enforce_n_colors(self):
        """ Ensure exactly N colors are used on the board """
        for color in range(self.n):
            color_presence = []
            for i in range(self.n):
                for j in range(self.n):
                    presence_var = self.model.NewBoolVar(f'presence_{color}_{i}_{j}')
                    self.model.Add(self.board[i][j] == color).OnlyEnforceIf(presence_var)
                    self.model.Add(self.board[i][j] != color).OnlyEnforceIf(presence_var.Not())
                    color_presence.append(presence_var)

            self.model.Add(sum(color_presence) >= 1).OnlyEnforceIf(self.used_colors[color])

        self.model.Add(sum(self.used_colors) == self.n)  # Ensure exactly N colors

    def enforce_unique_areas(self):
        """ Prevent the same color from forming multiple separate regions """
        for color in range(self.n):
            color_cells = [[self.model.NewBoolVar(f'color_{color}_cell_{i}_{j}') for j in range(self.n)] for i in range(self.n)]
            for i in range(self.n):
                for j in range(self.n):
                    self.model.Add(self.board[i][j] == color).OnlyEnforceIf(color_cells[i][j])
                    self.model.Add(self.board[i][j] != color).OnlyEnforceIf(color_cells[i][j].Not())

            # Ensure all cells of the same color are connected
            start_x, start_y = 0, 0
            queue = [(start_x, start_y)]
            visited = {queue[0]}

            while queue:
                x, y = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.n and 0 <= ny < self.n and (nx, ny) not in visited:
                        self.model.AddImplication(color_cells[x][y], color_cells[nx][ny])
                        queue.append((nx, ny))
                        visited.add((nx, ny))

    def solve(self):
        self.add_queen_constraints()
        self.enforce_n_colors()
        self.enforce_unique_areas()

        solver = cp_model.CpSolver()

        # Show initial board
        self.visualize_board(None, title="Initial Board (Randomized Placeholder)")

        status = solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = np.zeros((self.n, self.n), dtype=int)
            for i in range(self.n):
                for j in range(self.n):
                    solution[i][j] = solver.Value(self.board[i][j])

            # Show solved board
            self.visualize_board(solution, title="Solved Board (Optimized Placement)")
            return solution
        else:
            print("No solution found.")
            return None


# Example Usage
n = 6  # Board size and number of colors
solver = ColoredQueensSolver(n)
solution = solver.solve()

