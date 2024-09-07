GRID_WIDTH = 10
GRID_HEIGHT = 20

class StateReward:
    def __init__(self, grid, previous_state=None):
        self.grid = grid
        self.previous_state = previous_state
        self.aggregate_height = self.calculate_aggregate_height()
        self.complete_lines = sum(1 for row in grid if 0 not in row)
        self.current_holes = self.calculate_holes()
        self.current_bumpiness = self.calculate_bumpiness()
        self.highest_point = self.calculate_highest_point()
        self.reward = self.calculate_reward()

    def calculate_reward(self):
        """Calculate the reward based on grid state metrics."""
        if self.is_game_over():
            return -100
        # Constants for tuning
        a = -1  # Aggregate height
        b = 0.5  # Complete lines
        c = -0.8  # Holes
        d = -0.3  # Bumpiness
        e = -1  # New holes created
        f = -0.3  # Increase in bumpiness
        g = -0.5  # Height of the highest block

        # Calculate changes from the previous state
        if self.previous_state:
            previous_holes = self.previous_state.current_holes
            previous_bumpiness = self.previous_state.current_bumpiness
            previous_highest_point = self.previous_state.highest_point

            new_holes = max(0, self.current_holes - previous_holes)
            bumpiness_increase = max(0, self.current_bumpiness - previous_bumpiness)
            height_change = self.highest_point - previous_highest_point
        else:
            new_holes = 0
            bumpiness_increase = 0
            height_change = 0

        # Total Reward Calculation
        total_reward = (
            (a * self.aggregate_height) +
            (b * self.complete_lines) +
            (c * self.current_holes) +
            (d * self.current_bumpiness) +
            (e * new_holes) +
            (f * bumpiness_increase) +
            (g * height_change)
        )
        return total_reward

    def is_game_over(self):
        """Check if the game is over (top of the grid has blocks)."""
        return any(self.grid[0])
    def calculate_aggregate_height(self):
        """Calculate the aggregate height of all columns."""
        return sum(GRID_HEIGHT - next((y for y, cell in enumerate(col) if cell), GRID_HEIGHT) for col in zip(*self.grid))

    def calculate_holes(self):
        """Calculate the number of holes in the grid."""
        holes = 0
        for x in range(GRID_WIDTH):
            block_found = False
            for y in range(GRID_HEIGHT):
                if self.grid[y][x] != 0:
                    block_found = True
                elif block_found and self.grid[y][x] == 0:
                    holes += 1
        return holes

    def calculate_bumpiness(self):
        """Calculate the bumpiness of the grid (difference in height between columns)."""
        column_heights = [GRID_HEIGHT - next((y for y, cell in enumerate(col) if cell), GRID_HEIGHT) for col in
                          zip(*self.grid)]
        return sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(len(column_heights) - 1))

    def calculate_highest_point(self):
        """Calculate the height of the highest block in the grid."""
        for y in range(GRID_HEIGHT):
            if any(self.grid[y][x] != 0 for x in range(GRID_WIDTH)):
                return GRID_HEIGHT - y
        return GRID_HEIGHT  # Return max height if no blocks are found
