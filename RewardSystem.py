GRID_WIDTH = 10
GRID_HEIGHT = 20

DEFAULT_WEIGHTS = {
    'aggregate_height': -1,
    'CompleteLines': 0.5,
    'holes': -0.8,
    'Bumpiness': -0.3,
    'highest_point': -1
}


#TODO:  change reward system and add the score to the reward and complete lines deliver from the game.

class RewardSystem:

    def calculate_reward(self, grid, cleared_lines, weights=None):
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
                (weights['CompleteLines'] * complete_lines) +
                (weights['holes'] * current_holes) +
                (weights['Bumpiness'] * current_bumpiness) +
                (weights['highest_point'] * highest_point)
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
