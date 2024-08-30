import pygame
import random
from DQLAgent import DQLAgent

pygame.init()

# Configuration
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 25
CUBE_SIZE = 20
FRAME_WIDTH = 4
INTERNAL_PADDING = 2
UI_WIDTH = 200

# Calculated dimensions
GRID_PIXEL_WIDTH = GRID_WIDTH * BLOCK_SIZE + 2 * INTERNAL_PADDING
GRID_PIXEL_HEIGHT = GRID_HEIGHT * BLOCK_SIZE + 2 * INTERNAL_PADDING
SCREEN_WIDTH = GRID_PIXEL_WIDTH + UI_WIDTH + 2 * FRAME_WIDTH
SCREEN_HEIGHT = GRID_PIXEL_HEIGHT + 2 * FRAME_WIDTH
CUBE_OFFSET = (BLOCK_SIZE - CUBE_SIZE) // 2  # Offset to center the cube in the block

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (50, 50, 50)

# Tetrimino colors
COLORS = {
    'I': (0, 240, 240),
    'J': (0, 0, 240),
    'L': (240, 160, 0),
    'O': (240, 240, 0),
    'S': (0, 240, 0),
    'T': (160, 0, 240),
    'Z': (240, 0, 0),
}

# Tetrimino shapes
TETRIMINOS = {
    'I': [[1, 1, 1, 1]],
    'J': [[1, 0, 0],
          [1, 1, 1]],
    'L': [[0, 0, 1],
          [1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'S': [[0, 1, 1],
          [1, 1, 0]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'Z': [[1, 1, 0],
          [0, 1, 1]],
}


class Tetris:
    def __init__(self, agent):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris')
        self.current_tetrimino = None
        self.next_tetrimino = None
        self.game_over = False

        # self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.agent = agent
        self.game_counter = 0
        self.reset_game()
        self.previous_state = None
        self.current_state = None
        self.current_reward = 0
        self.current_action = None

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.speed = 200  # Milliseconds per fall
        # self.last_fall_time = pygame.time.get_ticks()
        self.agent.train()
        self.game_counter = self.game_counter + 1
        self.next_tetrimino = self.get_random_tetrimino()
        self.spawn_tetrimino()
        self.current_state = self.get_current_state()  # Save the current state

    def get_current_state(self):
        return State(self.grid, self.current_tetrimino)

    def get_random_tetrimino(self):
        shape = random.choice(list(TETRIMINOS.keys()))
        return {
            'shape': shape,
            'matrix': TETRIMINOS[shape],
            'color': COLORS[shape],
            'x': GRID_WIDTH // 2 - len(TETRIMINOS[shape][0]) // 2,
            'y': 0
        }

    def spawn_tetrimino(self):
        self.current_tetrimino = self.next_tetrimino
        self.next_tetrimino = self.get_random_tetrimino()
        if self.check_collision(self.current_tetrimino['x'], self.current_tetrimino['y'],
                                self.current_tetrimino['matrix']):
            self.reset_game()
        else:
            self.current_state = self.get_current_state()  # Update the current state

    def check_collision(self, x_offset, y_offset, matrix):
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    x_pos = x + x_offset
                    y_pos = y + y_offset
                    if x_pos < 0 or x_pos >= GRID_WIDTH or y_pos >= GRID_HEIGHT or (
                            y_pos >= 0 and self.grid[y_pos][x_pos]):
                        return True
        return False

    def lock_tetrimino(self):
        for y, row in enumerate(self.current_tetrimino['matrix']):
            for x, cell in enumerate(row):
                if cell:
                    x_pos = x + self.current_tetrimino['x']
                    y_pos = y + self.current_tetrimino['y']
                    if y_pos >= 0:
                        self.grid[y_pos][x_pos] = self.current_tetrimino['color']
        self.clear_lines()
        self.spawn_tetrimino()

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if 0 not in row]
        for i in lines_to_clear:
            del self.grid[i]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
        lines_cleared = len(lines_to_clear)
        if lines_cleared > 0:
            self.lines_cleared += lines_cleared
            self.score += (lines_cleared ** 2) * 100
            self.level = self.lines_cleared // 10 + 1
            self.speed = max(50, 500 - (self.level - 1) * 20)

    def rotate_tetrimino(self):
        matrix = self.current_tetrimino['matrix']
        rotated_matrix = [list(row) for row in zip(*matrix[::-1])]
        if not self.check_collision(self.current_tetrimino['x'], self.current_tetrimino['y'], rotated_matrix):
            self.current_tetrimino['matrix'] = rotated_matrix
        self.current_state = self.get_current_state()  # Update the current state

    def move_tetrimino(self, dx, dy):
        new_x = self.current_tetrimino['x'] + dx
        new_y = self.current_tetrimino['y'] + dy
        if not self.check_collision(new_x, new_y, self.current_tetrimino['matrix']):
            self.current_tetrimino['x'] = new_x
            self.current_tetrimino['y'] = new_y
            self.current_state = self.get_current_state()  # Update the current state
            return True
        elif dy == 1:
            self.lock_tetrimino()
            self.current_state = self.get_current_state()  # Update the current state
            return False
        self.current_state = self.get_current_state()  # Update the current state
        return True

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect_x = FRAME_WIDTH + INTERNAL_PADDING + x * BLOCK_SIZE + CUBE_OFFSET
                rect_y = FRAME_WIDTH + INTERNAL_PADDING + y * BLOCK_SIZE + CUBE_OFFSET
                rect = pygame.Rect(rect_x, rect_y, CUBE_SIZE, CUBE_SIZE)
                color = self.grid[y][x]
                if color:
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 2)  # Add white outline around the placed blocks

    def draw_tetrimino(self):
        matrix = self.current_tetrimino['matrix']
        color = self.current_tetrimino['color']
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    rect_x = FRAME_WIDTH + INTERNAL_PADDING + (
                            self.current_tetrimino['x'] + x) * BLOCK_SIZE + CUBE_OFFSET
                    rect_y = FRAME_WIDTH + INTERNAL_PADDING + (
                            self.current_tetrimino['y'] + y) * BLOCK_SIZE + CUBE_OFFSET
                    rect = pygame.Rect(rect_x, rect_y, CUBE_SIZE, CUBE_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 2)  # Add white outline around the active blocks

    def draw_next_tetrimino(self):
        matrix = self.next_tetrimino['matrix']
        color = self.next_tetrimino['color']
        start_x = GRID_PIXEL_WIDTH + FRAME_WIDTH + 20
        start_y = 150
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    rect_x = start_x + x * BLOCK_SIZE + CUBE_OFFSET
                    rect_y = start_y + y * BLOCK_SIZE + CUBE_OFFSET
                    rect = pygame.Rect(rect_x, rect_y, CUBE_SIZE, CUBE_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 2)  # Add white outline around the next piece blocks

    def draw_frame(self):
        rect = pygame.Rect(0, 0, GRID_PIXEL_WIDTH + 2 * FRAME_WIDTH, GRID_PIXEL_HEIGHT + 2 * FRAME_WIDTH)
        pygame.draw.rect(self.screen, WHITE, rect)
        inner_rect = pygame.Rect(FRAME_WIDTH, FRAME_WIDTH, GRID_PIXEL_WIDTH, GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, inner_rect)

    def draw_ui(self):
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 50))

        level_text = self.font.render(f'Level: {self.level}', True, WHITE)
        self.screen.blit(level_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 90))

        next_text = self.font.render('Next:', True, WHITE)
        self.screen.blit(next_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 120))

        game_counter = self.font.render(f'Games:{self.game_counter}', True, WHITE)
        self.screen.blit(game_counter, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 200))

        lines_text = self.font.render(f'Lines: {self.lines_cleared}', True, WHITE)
        self.screen.blit(lines_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 300))

    def handle_agent_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        lock_state = self.agent.choose_best_final_state(self.current_state, self.get_all_lock_states())
        self.previous_state = self.current_state
        self.set_tetrimino_to_state(lock_state)
        self.hard_drop()
        self.current_state = self.get_current_state()

        return False

    def hard_drop(self):
        # Drops the current tetrimino straight down to its lowest possible position, updating the display on each drop.
        moved = True
        while moved:
            moved = self.move_tetrimino(0, 1)
            # Refresh the game display to show the tetrimino moving down
            self.refresh_game()
            # Add a short delay to make the movement visually smooth
            pygame.time.delay(5)

    def rotate_matrix(self, matrix, times=1):
        # Rotate the tetrimino matrix 90 degrees clockwise `times` number of times
        for _ in range(times):
            matrix = [list(row) for row in zip(*matrix[::-1])]
        return matrix

    def update_agent_thread(self):
        self.agent.update_agent(self.previous_state, self.current_state,
                                self.calculate_reward(), self.game_over)

    def refresh_game(self):
        # self.updateGame()
        self.screen.fill(BLACK)
        self.draw_frame()
        self.draw_grid()
        self.draw_tetrimino()
        self.draw_ui()
        self.draw_next_tetrimino()
        pygame.display.flip()
        # self.clock.tick(60)

    def run(self):
        while not self.game_over:
            self.current_state = self.get_current_state()
            self.game_over = self.handle_agent_events()
            self.refresh_game()
            self.update_agent_thread()
        pygame.quit()

    def calculate_reward(self):
        # Calculate the reward for the agent's action based on the placement of the Tetrimino.
        temp_grid, (final_x, final_y) = self.simulate_drop()

        # Calculate penalties for holes
        hole_penalty = 0
        for col in range(GRID_WIDTH):
            column = [temp_grid[row][col] for row in range(GRID_HEIGHT)]
            first_block = GRID_HEIGHT - 1  # Start from the bottom and go upwards
            for row in range(GRID_HEIGHT - 1, -1, -1):
                if column[row] != 0:
                    first_block = row
                elif row < first_block:
                    hole_penalty += 1  # Count each empty space below the topmost block in each column as a hole

        # Reward for lines cleared
        lines_cleared = self.check_lines(temp_grid)  # Method to check how many lines would be cleared
        line_clear_reward = (lines_cleared ** 2) * 100  # Quadratically increase reward for multiple lines

        # Calculate height penalty to discourage stacking tetriminos too high
        height_penalty = 0
        for col in range(GRID_WIDTH):
            for row in range(GRID_HEIGHT):
                if temp_grid[row][col] != 0:
                    height_penalty += (GRID_HEIGHT - row)  # Penalize based on height of blocks
                    break  # Only count the topmost block in each column

        total_reward = line_clear_reward - (hole_penalty * 50) - (height_penalty * 2)
        return -total_reward

    def check_lines(self, grid):
        #Check how many lines can be cleared in the given grid.
        return sum(1 for row in grid if 0 not in row)

    def simulate_drop(self):
        # Create a temporary grid by copying the current grid
        temp_grid = [row[:] for row in self.grid]
        temp_tetrimino = self.current_tetrimino.copy()

        # Simulate the drop
        while True:
            new_y = temp_tetrimino['y'] + 1  # Move one row down
            if self.is_valid_position(temp_grid, temp_tetrimino['x'], new_y, temp_tetrimino['matrix']):
                temp_tetrimino['y'] = new_y
            else:
                break  # Stop if the new position is not valid

        # Find the highest block position in the Tetrimino after the drop
        highest_point = None  # To store (x, y) of the highest block
        for y, row in enumerate(temp_tetrimino['matrix']):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = temp_tetrimino['y'] + y
                    if highest_point is None or grid_y < highest_point[1]:
                        highest_point = (temp_tetrimino['x'] + x, grid_y)

        return temp_grid, highest_point

    def is_valid_position(self, grid, x, y, matrix):
        # Iterate over the matrix of the Tetrimino
        for row_index, row in enumerate(matrix):
            for col_index, cell in enumerate(row):
                if cell:  # Only consider non-empty cells of the Tetrimino
                    grid_x = x + col_index
                    grid_y = y + row_index

                    # Check if the cell is out of the left or right bounds of the grid
                    if grid_x < 0 or grid_x >= GRID_WIDTH:
                        return False

                    # Check if the cell is below the bottom of the grid
                    if grid_y >= GRID_HEIGHT:
                        return False

                    # Check if the cell collides with an already placed block in the grid
                    if grid_y >= 0 and grid[grid_y][grid_x]:
                        return False

        return True

    def get_all_lock_states(self):
        # Generates all possible lock states for the current tetrimino from the current state.
        lock_states = []
        initial_tetrimino = self.current_tetrimino.copy()
        original_matrix = initial_tetrimino['matrix']

        # Try all rotations of the tetrimino
        for rotation in range(4):
            rotated_matrix = self.rotate_matrix(original_matrix, times=rotation)
            temp_tetrimino = initial_tetrimino.copy()
            temp_tetrimino['matrix'] = rotated_matrix

            # Try all horizontal positions
            for x in range(0, GRID_WIDTH - len(rotated_matrix[0]) + 1):
                temp_tetrimino['x'] = x
                temp_tetrimino['y'] = 0  # Start at the top of the grid

                # Find the lowest valid y position for the current x and rotation
                while not self.check_collision(temp_tetrimino['x'], temp_tetrimino['y'] + 1, rotated_matrix):
                    temp_tetrimino['y'] += 1

                if self.check_collision(temp_tetrimino['x'], temp_tetrimino['y'] + len(rotated_matrix) - 1,
                                        rotated_matrix):
                    # Create a grid copy and lock the tetrimino in place
                    grid_copy = [row[:] for row in self.grid]
                    self.lock_tetrimino_in_grid(grid_copy, temp_tetrimino)
                    lock_states.append(State(grid_copy, temp_tetrimino.copy()))

        return lock_states

    def lock_tetrimino_in_grid(self, grid, tetrimino):
        # Locks the tetrimino into the provided grid.
        matrix = tetrimino['matrix']
        x_pos = tetrimino['x']
        y_pos = tetrimino['y']
        color = tetrimino['color']

        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = x_pos + x
                    grid_y = y_pos + y
                    if 0 <= grid_y < GRID_HEIGHT and 0 <= grid_x < GRID_WIDTH:
                        grid[grid_y][grid_x] = color

    def set_tetrimino_to_state(self, lock_state):

        # Adjusts the current tetrimino's orientation and position to match the given lock state.
        # Moves the tetrimino horizontally to match the x position specified in the lock state.

        desired_x = lock_state.current_tetrimino['x']
        desired_matrix = lock_state.current_tetrimino['matrix']

        # First, rotate the tetrimino to the desired orientation
        rotation_attempts = 0
        while self.current_tetrimino['matrix'] != desired_matrix and rotation_attempts < 4:
            self.rotate_tetrimino()
            rotation_attempts += 1
            self.refresh_game()  # Visual feedback
            pygame.time.delay(20)  # Delay to visualize rotation

        if rotation_attempts >= 4:
            print("Warning: Unable to align rotation with desired state.")
            return  # Exit if cannot align rotation to avoid infinite loop

        # Move horizontally to the desired position
        move_attempts = 0
        max_moves = abs(self.current_tetrimino['x'] - desired_x) + 2  # Allow some extra attempts
        while self.current_tetrimino['x'] != desired_x and move_attempts < max_moves:
            move_direction = 1 if self.current_tetrimino['x'] < desired_x else -1
            if not self.check_collision(self.current_tetrimino['x'] + move_direction, self.current_tetrimino['y'],
                                        self.current_tetrimino['matrix']):
                self.current_tetrimino['x'] += move_direction
                self.refresh_game()  # Visual feedback
                pygame.time.delay(20)  # Delay to visualize movement
            else:
                print("Collision detected, stopping movement.")
                break
            move_attempts += 1


class State:
    def __init__(self, grid, current_tetrimino):
        # Copy the grid to avoid altering the original one
        self.grid = [row[:] for row in grid]
        self.current_tetrimino = current_tetrimino


if __name__ == '__main__':
    game = Tetris(DQLAgent())
    game.run()
