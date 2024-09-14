GRID_WIDTH = 10
GRID_HEIGHT = 20

DEFAULT_WEIGHTS = {'aggregate_height': 0.2638489398939668,
                   'complete_lines': 1.6779530168219996,
                   'holes': 1.1407294274788815,
                   'bumpiness': 0.1646717469283948,
                   'highest_point': 0.006402588878321655,
                   'etp_score': 0.40935000526739124}

# TODO:  change reward system and add the score to the reward and complete lines deliver from the game.

class RewardSystem:

    def calculate_reward(self, current_state,weights=None):
        if weights is None:
            weights = DEFAULT_WEIGHTS  # Ensure weights is a dictionary

        aggregate_height = self.calculate_aggregate_height(current_state.grid)
        cleared_lines = self.calculate_clear_lines(current_state.grid)
        current_holes = self.calculate_holes(current_state.grid)
        current_bumpiness = self.calculate_bumpiness(current_state.grid)
        highest_point = self.calculate_highest_point(current_state.grid)
        etp_score = self.calculate_etp(current_state)

        # Total Reward Calculation
        total_reward = (
                (weights['complete_lines'] * cleared_lines)
                - (weights['aggregate_height'] * aggregate_height)
                - (weights['holes'] * current_holes)
                - (weights['bumpiness'] * current_bumpiness)
                - (weights['highest_point'] * highest_point)
                + (weights['etp_score'] * etp_score))

        return total_reward

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
        shape_x, shape_y = locked_shape_info['x'], locked_shape_info['y']
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

    def calculate_clear_lines(self, grid):
        return sum(1 for row in grid if 0 not in row)

    def calculate_etp(self, state):
        """Calculate the edge touching points (ETP) for the current tetrimino in the given state."""
        grid = state.grid
        piece_matrix = state.current_tetrimino['matrix']
        x = state.current_tetrimino['x']
        y = state.current_tetrimino['y']
        part_height = len(piece_matrix)
        part_width = len(piece_matrix[0])
        counter = 0

        # Iterate over each block in the piece matrix and its surrounding area
        for py in range(-1, part_height + 1):
            for px in range(-1, part_width + 1):
                by, bx = y + py, x + px
                if (bx < 0 or bx >= GRID_WIDTH or by < 0 or by >= GRID_HEIGHT or (
                        0 <= by < GRID_HEIGHT and 0 <= bx < GRID_WIDTH and grid[by][bx] != 0)):
                    is_full = True
                else:
                    is_full = False

                if is_full:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        temp_y, temp_x = py + dy, px + dx
                        if 0 <= temp_y < part_height and 0 <= temp_x < part_width and piece_matrix[temp_y][temp_x]:
                            counter += 1

        return counter

