GRID_WIDTH = 10
GRID_HEIGHT = 20

DEFAULT_WEIGHTS = {'aggregate_height': -0.6998233751177039,
                   'complete_lines': 9.536445648965877,
                   'holes': -9.244664883780514,
                   'bumpiness': -2.7113804115775544,
                   'highest_point': -3.6276559992864015,
                   'single_holes': -7.41308397447242}


#TODO:  change reward system and add the score to the reward and complete lines deliver from the game.

class RewardSystem:

    def calculate_reward(self, grid, cleared_lines = None, weights=None):
        if weights is None:
            weights = DEFAULT_WEIGHTS

        aggregate_height = self.calculate_aggregate_height(grid)
        complete_lines = cleared_lines if cleared_lines is not None else sum(1 for row in grid if 0 not in row)
        current_holes = self.calculate_holes(grid)
        current_bumpiness = self.calculate_bumpiness(grid)
        highest_point = self.calculate_highest_point(grid)

        # Total Reward Calculation
        total_reward = (
                (weights['aggregate_height'] * aggregate_height) +
                (weights['complete_lines'] * complete_lines) +
                (weights['holes'] * current_holes) +
                (weights['bumpiness'] * current_bumpiness) +
                (weights['highest_point'] * highest_point) +
                (weights['single_holes'] * self.calculate_single_holes(grid))
        )
        return total_reward

    # def is_game_over(self, grid):
    #     """Check if the game is over (top of the grid has blocks)."""
    #     return any(grid[0])

    def calculate_aggregate_height(self, grid):
        """Calculate the aggregate height of all columns."""
        return sum(
            GRID_HEIGHT - next((y for y, cell in enumerate(col) if cell), GRID_HEIGHT) for col in zip(*grid))

    def calculate_holes(self, grid):
        """Calculate the number of holes in the grid."""
        holes = 0
        for x in range(GRID_WIDTH):
            block_found = False
            for y in range(GRID_HEIGHT):
                if grid[y][x] != 0:
                    block_found = True
                elif block_found and grid[y][x] == 0:
                    holes += 1
        return holes

    def calculate_bumpiness(self, grid):
        """Calculate the bumpiness of the grid (difference in height between columns)."""
        column_heights = [GRID_HEIGHT - next((y for y, cell in enumerate(col) if cell), GRID_HEIGHT) for col in
                          zip(*grid)]
        return sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(len(column_heights) - 1))

    def calculate_highest_point(self, grid):
        """Calculate the height of the highest block in the grid."""
        for y in range(GRID_HEIGHT):
            if any(grid[y][x] != 0 for x in range(GRID_WIDTH)):
                return GRID_HEIGHT - y
        return GRID_HEIGHT  # Return max height if no blocks are found

    def calculate_single_holes(self, grid):
        """Calculate the number of single holes in the grid.
        A single hole is defined as an empty cell (0) that has its left and right neighbors filled (1).

        Args:
            grid (list of list of int): The Tetris grid represented as a 2D list, where 0 represents an empty cell
                                        and 1 represents a filled cell.

        Returns:
            int: The number of single holes in the grid.
        """
        rows = len(grid)
        cols = len(grid[0])
        single_holes_count = 0

        # Iterate through each cell in the grid
        for row in range(rows):
            for col in range(cols):
                # Check if the current cell is a hole (0)
                if grid[row][col] == 0:
                    # Initialize a flag to check if it's a single hole
                    is_single_hole = True

                    # Handle edge cases
                    if col == 0:  # Leftmost column: check only the right neighbor
                        if grid[row][col + 1] != 1:
                            is_single_hole = False

                    elif col == cols - 1:  # Rightmost column: check only the left neighbor
                        if grid[row][col - 1] != 1:
                            is_single_hole = False

                    else:  # Middle columns: check both left and right neighbors
                        if grid[row][col - 1] != 1 or grid[row][col + 1] != 1:
                            is_single_hole = False

                    # If it is a single hole, increment the count
                    if is_single_hole:
                        single_holes_count += 1

        return single_holes_count



