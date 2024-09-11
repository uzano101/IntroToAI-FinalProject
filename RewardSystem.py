GRID_WIDTH = 10
GRID_HEIGHT = 20

DEFAULT_WEIGHTS = {
    'aggregate_height': -0.6998233751177039,
    'complete_lines': 9.536445648965877,
    'holes': -9.244664883780514,
    'bumpiness': -2.7113804115775544,
    'highest_point': -3.6276559992864015,
    'isolation_score': -5.0
}



#TODO:  change reward system and add the score to the reward and complete lines deliver from the game.

class RewardSystem:

    def calculate_reward(self, grid, cleared_lines=None, weights=None, isolation_score=0):
        if weights is None:
            weights = DEFAULT_WEIGHTS  # Ensure weights is a dictionary

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
            (weights['isolation_score'] * isolation_score)
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

    def calculate_isolation_for_locked_shape(self, grid, locked_shape_info):
        """Calculate the isolation score for a single locked shape based on its position on the grid."""
        shape_matrix = locked_shape_info['matrix']
        shape_x, shape_y = locked_shape_info['position']
        isolation_score = 0

        # Direction vectors for the 4 direct neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Check isolation for the entire shape
        for y in range(len(shape_matrix)):
            for x in range(len(shape_matrix[0])):
                if shape_matrix[y][x] != 0:  # If this is part of the shape
                    by, bx = shape_y + y, shape_x + x
                    empty_neighbors = 0
                    total_neighbors = 0

                    # Check all 4 possible direct neighbors
                    for dy, dx in directions:
                        ny, nx = by + dy, bx + dx
                        if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:  # Ensure neighbor is within grid bounds
                            total_neighbors += 1
                            if grid[ny][nx] == 0:
                                empty_neighbors += 1

                    # If block has 2/3 or more empty neighbors, it is considered isolated
                    if total_neighbors > 0 and empty_neighbors >= (total_neighbors * 2 / 3):
                        isolation_score += 1

        return isolation_score



