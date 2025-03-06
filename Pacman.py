import tkinter as tk
import random

class Maze:
    def __init__(self, canvas, cell_size=30, layout=None):
        self.canvas = canvas
        self.cell_size = cell_size
        # Define a default maze layout if none is provided.
        # Walls are represented by "#" and pellets by "."
        if layout is None:
            self.layout = [
                "###############",
                "#.............#",
                "#.###.###.###.#",
                "#.............#",
                "#.###.#.#.###.#",
                "#.............#",
                "###############"
            ]
        else:
            self.layout = layout
        self.pellets = {}  # Dictionary to store pellet canvas ids by (row, col)
        self.draw_maze()

    def draw_maze(self):
        self.pellets.clear()  # clear previous pellets (if any)
        for r, row in enumerate(self.layout):
            for c, char in enumerate(row):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if char == "#":
                    # Draw wall: using blue fill and black outline.
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black")
                elif char == ".":
                    # Draw pellet: a small yellow circle centered in the cell.
                    pellet_radius = self.cell_size // 6
                    cx = x1 + self.cell_size / 2
                    cy = y1 + self.cell_size / 2
                    pellet = self.canvas.create_oval(
                        cx - pellet_radius, cy - pellet_radius,
                        cx + pellet_radius, cy + pellet_radius,
                        fill="yellow", outline="yellow"
                    )
                    self.pellets[(r, c)] = pellet
                # For blank spaces we do nothing.

    def remove_pellet(self, row, col):
        """Remove a pellet from the canvas and update the maze layout."""
        if (row, col) in self.pellets:
            self.canvas.delete(self.pellets[(row, col)])
            del self.pellets[(row, col)]
            # Update the layout to reflect that the pellet is eaten.
            row_list = list(self.layout[row])
            row_list[col] = " "
            self.layout[row] = "".join(row_list)

class Pacman:
    def __init__(self, canvas, maze, cell_size=30, start_row=1, start_col=1):
        self.canvas = canvas
        self.maze = maze
        self.cell_size = cell_size
        self.row = start_row
        self.col = start_col
        self.direction = "Right"  # Initial direction
        self.id = None
        self.draw()

    def draw(self):
        x1 = self.col * self.cell_size
        y1 = self.row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        # Pac-Man is drawn as a yellow circle.
        self.id = self.canvas.create_oval(x1, y1, x2, y2, fill="yellow", outline="black")

    def move(self, new_row, new_col):
        dx = (new_col - self.col) * self.cell_size
        dy = (new_row - self.row) * self.cell_size
        self.canvas.move(self.id, dx, dy)
        self.row = new_row
        self.col = new_col

    def update_direction(self, new_direction):
        self.direction = new_direction

class Ghost:
    def __init__(self, canvas, maze, cell_size=30, start_row=3, start_col=7):
        self.canvas = canvas
        self.maze = maze
        self.cell_size = cell_size
        self.row = start_row
        self.col = start_col
        self.direction = random.choice(["Up", "Down", "Left", "Right"])
        self.id = None
        self.draw()

    def draw(self):
        x1 = self.col * self.cell_size
        y1 = self.row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        # The ghost is drawn as a red circle.
        self.id = self.canvas.create_oval(x1, y1, x2, y2, fill="red", outline="black")

    def move(self, new_row, new_col):
        dx = (new_col - self.col) * self.cell_size
        dy = (new_row - self.row) * self.cell_size
        self.canvas.move(self.id, dx, dy)
        self.row = new_row
        self.col = new_col

    def choose_direction(self):
        directions = ["Up", "Down", "Left", "Right"]
        valid_directions = []
        for d in directions:
            r, c = self.row, self.col
            if d == "Up":
                r -= 1
            elif d == "Down":
                r += 1
            elif d == "Left":
                c -= 1
            elif d == "Right":
                c += 1
            if self.maze.layout[r][c] != "#":
                valid_directions.append(d)
        if valid_directions:
            self.direction = random.choice(valid_directions)
        else:
            self.direction = None

    def update(self):
        if self.direction is None:
            self.choose_direction()
        r, c = self.row, self.col
        new_r, new_c = r, c
        if self.direction == "Up":
            new_r -= 1
        elif self.direction == "Down":
            new_r += 1
        elif self.direction == "Left":
            new_c -= 1
        elif self.direction == "Right":
            new_c += 1
        # Move only if the next cell is not a wall.
        if self.maze.layout[new_r][new_c] != "#":
            self.move(new_r, new_c)
        else:
            self.choose_direction()

class PacmanGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Pac-Man Game")
        self.cell_size = 30

        # Define the maze layout.
        self.maze_layout = [
            "###############",
            "#.............#",
            "#.###.###.###.#",
            "#.............#",
            "#.###.#.#.###.#",
            "#.............#",
            "###############"
        ]
        self.maze_rows = len(self.maze_layout)
        self.maze_cols = len(self.maze_layout[0])
        canvas_width = self.maze_cols * self.cell_size
        canvas_height = self.maze_rows * self.cell_size

        self.canvas = tk.Canvas(self.window, width=canvas_width, height=canvas_height, bg="black")
        self.canvas.pack()
        self.window.resizable(False, False)

        # Create the maze.
        self.maze = Maze(self.canvas, self.cell_size, layout=self.maze_layout)

        # A score label.
        self.score = 0
        self.label = tk.Label(self.window, text="Score: {}".format(self.score), font=('Helvetica', 14))
        self.label.pack()

        # Create Pac-Man and one Ghost.
        self.pacman = Pacman(self.canvas, self.maze, self.cell_size, start_row=1, start_col=1)
        self.ghost = Ghost(self.canvas, self.maze, self.cell_size, start_row=3, start_col=7)

        # Ensure the canvas has keyboard focus.
        self.canvas.focus_set()
        self.window.bind("<KeyPress>", self.on_key_press)

        self.game_running = True
        self.game_loop()
        self.window.mainloop()

    def on_key_press(self, event):
        key = event.keysym
        if key in ["Up", "Down", "Left", "Right"]:
            self.pacman.update_direction(key)

    def game_loop(self):
        if not self.game_running:
            return

        # Determine Pac-Man's next position based on its current direction.
        current_row = self.pacman.row
        current_col = self.pacman.col
        new_row = current_row
        new_col = current_col
        direction = self.pacman.direction
        if direction == "Up":
            new_row -= 1
        elif direction == "Down":
            new_row += 1
        elif direction == "Left":
            new_col -= 1
        elif direction == "Right":
            new_col += 1

        # Check that the new cell is not a wall.
        if self.maze.layout[new_row][new_col] != "#":
            self.pacman.move(new_row, new_col)
            # Check for pellet consumption.
            if (new_row, new_col) in self.maze.pellets:
                self.maze.remove_pellet(new_row, new_col)
                self.score += 10
                self.label.config(text="Score: {}".format(self.score))

        # Update ghost movement.
        self.ghost.choose_direction()
        self.ghost.update()

        # Check for collision between Pac-Man and the ghost.
        if self.pacman.row == self.ghost.row and self.pacman.col == self.ghost.col:
            self.game_over()
            return

        # Continue the game loop after a short delay.
        self.window.after(300, self.game_loop)

    def game_over(self):
        self.game_running = False
        self.canvas.create_text(self.canvas.winfo_width() / 2,
                                self.canvas.winfo_height() / 2,
                                text="GAME OVER", fill="white",
                                font=("Helvetica", 32))

if __name__ == "__main__":
    PacmanGame()
