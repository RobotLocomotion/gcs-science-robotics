import matplotlib.pyplot as plt
from random import choice

class Cell:
    """A cell in the maze. A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.
    """

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def _knock_down_wall(self, wall):
        """Knock down the given wall."""

        self.walls[wall] = False

class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny):
        """Initialize the maze grid.
        The maze consists of nx x ny cells.

        """

        self.nx, self.ny = nx, ny
        self.map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.map[x][y]
    
    def plot(self, linewidth):
        plt.gca().axis('off')
        
        # Pad the maze all around by this amount.
        width = self.nx
        height = self.ny
        
        # Draw the South and West maze borders.
        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x, y).walls['S'] and (x != 0 or y != 0):
                    plt.plot([x, x + 1], [y, y], c='k', linewidth=linewidth)
                if self.cell_at(x, y).walls['W']:
                    plt.plot([x, x], [y, y + 1], c='k', linewidth=linewidth)
                    
        # Draw the North and East maze border, which won't have been drawn
        # by the procedure above.
        plt.plot([0, width - 1], [height, height], c='k', linewidth=linewidth)
        plt.plot([width, width], [0, height], c='k', linewidth=linewidth)
        
    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, -1)),
                 ('N', (0, 1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(0, 0)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = choice(neighbours)
            self.knock_down_wall(current_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1
            
    def knock_down_wall(self, cell, wall):
        cell._knock_down_wall(wall)
        increment = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}[wall]
        neighbor = self.cell_at(cell.x + increment[0], cell.y + increment[1])
        neighbor_wall = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[wall]
        neighbor._knock_down_wall(neighbor_wall)
