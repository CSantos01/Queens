import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons

class Case:
    '''
    Class to represent a case in the game board.
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = 'white'
        self.queen = False
        self.is_checked = False

    def __repr__(self):
        return f'({self.x}, {self.y})'

class GameBoard:
    '''
    Class to represent the game board for the Queens game.
    '''
    def __init__(self, size=7):
        self.size = size
        self.board = [[Case(x, y) for x in range(size)] for y in range(size)]

    def get_colors(self):
        transformed_board_colors = np.zeros((self.size, self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                transformed_board_colors[i, j] = self.board[i][j].color
        return transformed_board_colors
    
    def get_queens(self):
        transformed_board_is_queen = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            for j in range(self.size):
                transformed_board_is_queen[i, j] = self.board[i][j].queen
        return transformed_board_is_queen

    def get_checked(self):
        transformed_board_is_checked = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            for j in range(self.size):
                transformed_board_is_checked[i, j] = self.board[i][j].is_checked
        return transformed_board_is_checked
    
    def get_all(self):
        return self.get_colors(), self.get_queens(), self.get_checked()

class GameBoardVisualizer:
    '''
    Class to handle the visualization and interaction of the game board.
    '''
    color_map = {
        'Red': 'red',
        'Green': 'green',
        'Blue': 'blue',
        'Orange': 'orange',
        'Yellow': 'yellow',
        'Cyan': 'cyan',
        'Magenta': 'magenta',
        'Purple': 'purple',
        'Brown': 'brown',
        'Pink': 'pink',
        'Yellowgreen': 'yellowgreen',
        'White': 'white'
    }

    def __init__(self, game_board):
        self.game_board = game_board
        self.size = game_board.size
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xticks(np.arange(-.5, self.size, 1))
        self.ax.set_yticks(np.arange(-.5, self.size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(color='black', linestyle='-', linewidth=2)
        self.rects = [[self.ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, edgecolor='black', facecolor='white')) for j in range(self.size)] for i in range(self.size)]
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.selected_color = 'red'
        self.add_color_buttons()

    def on_click(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < self.size and 0 <= y < self.size:
                self.game_board.board[y][x].color = self.selected_color
                self.rects[y][x].set_facecolor(self.selected_color)
                self.fig.canvas.draw()

    def select_color(self, event):
        self.selected_color = self.color_map[event]
        self.fig.canvas.draw()

    def add_color_buttons(self):
        axcolor = plt.axes([0.90, 0.05, 0.1, 0.2], frameon=False)
        self.radio = RadioButtons(axcolor, list(self.color_map.keys()))
        self.radio.on_clicked(self.select_color)

    def show(self):
        plt.show()

class Queen:
    '''
    Class to represent a queen in the game board.
    '''
    def __init__(self, game_board, x, y):
        self.game_board = game_board
        case = self.game_board.board[y][x]
        # First check if the case is already checked by another queen
        if not case.is_checked:
            self.x = x
            self.y = y
            self.color = case.color
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
            print('The case is already checked by another queen')

    def __repr__(self):
        return f'Queen({self.x}, {self.y}, {self.color})'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Queens Game')
    parser.add_argument('-size', type=int, default=7, help='Size of the game board')
    args = parser.parse_args()

    board = GameBoard(args.size)
    visual = GameBoardVisualizer(board)
    visual.show()
    print(board.get_all())
    queen_1 = Queen(board, 0, 0)
    print(board.get_all())
    queen_2 = Queen(board, 1, 1)


    from pathlib import Path
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    print(board.get_all())
    np.save(output_dir / 'transformed_board.npy', board.get_all())