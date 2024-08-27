import threading

import pygame
import random
import time
from DQLAgent import DQLAgent
pygame.init()

# Configuration
GRID_WIDTH = 10  # Number of columns
GRID_HEIGHT = 20  # Number of rows
BLOCK_SIZE = 25  # Size of each block in pixels (grid size)
CUBE_SIZE = 20  # Actual size of the cube inside the grid
FRAME_WIDTH = 4  # Width of the grid frame in pixels
INTERNAL_PADDING = 2  # Padding between frame and blocks in pixels (adjusted for consistency)
UI_WIDTH = 200  # Width of the UI area in pixels

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

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.agent = agent
        self.reset_game()
        self.previous_state = None
        self.current_state = None
        self.current_reward = 0
        self.current_action = None
        self.game_over = False
    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.speed = 200  # Milliseconds per fall
        self.last_fall_time = pygame.time.get_ticks()

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
        if self.check_collision(self.current_tetrimino['x'], self.current_tetrimino['y'], self.current_tetrimino['matrix']):
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
        self.current_state = self.get_current_state() # Update the current state

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

    def hard_drop(self):
        while self.move_tetrimino(0, 1):
            pass

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
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 50))

        # Draw level
        level_text = self.font.render(f'Level: {self.level}', True, WHITE)
        self.screen.blit(level_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 90))

        # Draw next piece label
        next_text = self.font.render('Next:', True, WHITE)
        self.screen.blit(next_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 130))

        # Draw lines cleared
        lines_text = self.font.render(f'Lines: {self.lines_cleared}', True, WHITE)
        self.screen.blit(lines_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 300))

    def updateGame(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fall_time > self.speed:
            self.move_tetrimino(0, 1)
            self.last_fall_time = current_time


    # def handle_events(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             return False
    #
    #         elif event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_LEFT:
    #                 self.move_tetrimino(-1, 0)
    #             elif event.key == pygame.K_RIGHT:
    #                 self.move_tetrimino(1, 0)
    #             # elif event.key == pygame.K_DOWN:
    #             #     self.move_tetrimino(0, 1)
    #             elif event.key == pygame.K_UP:
    #                 self.rotate_tetrimino()
    #             # elif event.key == pygame.K_SPACE:
    #             #     self.hard_drop()
    #     return True

    def handle_agent_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        self.current_action = self.agent.choose_action(self.current_state)
        self.previous_state = self.current_state
        if self.current_action == 0:
            self.move_tetrimino(-1, 0)
        elif self.current_action == 1:
            self.move_tetrimino(1, 0)
        elif self.current_action == 2:
            self.rotate_tetrimino()
        elif self.current_action == 3:
            pass
        return False

    # def get_next_states(self):
    #     next_states = []
    #     current_shape = self.current_tetrimino['shape']
    #     current_matrix = self.current_tetrimino['matrix']
    #
    #     # Explore all possible rotations
    #     for rotation in range(4):
    #         rotated_matrix = self.rotate_matrix(current_matrix, rotation)
    #
    #         # Explore all possible horizontal positions
    #         for x_pos in range(-len(rotated_matrix[0]) + 1, GRID_WIDTH):
    #             if not self.check_collision(x_pos, self.current_tetrimino['y'], rotated_matrix):
    #                 # Create a copy of the grid
    #                 new_grid = [row[:] for row in self.grid]
    #
    #                 # Simulate the tetromino placement in this state
    #                 for y, row in enumerate(rotated_matrix):
    #                     for x, cell in enumerate(row):
    #                         if cell:
    #                             new_grid[self.current_tetrimino['y'] + y][x_pos + x] = self.current_tetrimino['color']
    #
    #                 # Create a new state and add it to the list
    #                 next_states.append(State(new_grid, current_shape, x_pos, self.current_tetrimino['y'], rotation))
    #
    #     return next_states

    def rotate_matrix(self, matrix, times=1):
        # Rotate the matrix 90 degrees clockwise `times` number of times
        for _ in range(times):
            matrix = [list(row) for row in zip(*matrix[::-1])]
        return matrix

    def update_agent_thread(self):
        self.agent.update_agent(self.previous_state, self.current_state, self.current_action,
                                self.calculate_reward(), self.game_over)

    def refresh_game(self):
        self.updateGame()
        self.screen.fill(BLACK)
        self.draw_frame()
        self.draw_grid()
        self.draw_tetrimino()
        self.draw_ui()
        self.draw_next_tetrimino()
        pygame.display.flip()
        self.clock.tick(60)

    def run(self):
        while not self.game_over:
            # Update game state before threading
            self.current_state = self.get_current_state()  # Update the current state
            self.game_over = self.handle_agent_events()

            # Create threads
            agent_thread = threading.Thread(target=self.update_agent_thread)
            game_thread = threading.Thread(target=self.refresh_game())

            # Start threads
            agent_thread.start()
            game_thread.start()

            # Wait for both threads to complete
            agent_thread.join()
            game_thread.join()

        pygame.quit()
    def run(self):
        while not self.game_over:
            # 1. Update game state
            self.current_state = self.get_current_state()

            # 2. Handle agent's decision
            self.game_over = self.handle_agent_events()  # Agent decides action here

            # 3. Render immediately after handling agent events
            self.refresh_game()  # Direct call to refresh_game without threading
            self.update_agent_thread()

            # Thread handling may not be necessary if immediate rendering is required after every agent action.
            # If performance or responsiveness is an issue, consider optimizing handle_agent_events and refresh_game.

            # Delay here if needed to maintain consistent frame rate (optional)
            pygame.time.delay(20)  # Delay to simulate frame rate control

        pygame.quit()

    def calculate_reward(self):
        # Store the current state to avoid modifying the game
        temp_grid = [row[:] for row in self.grid]
        temp_tetrimino = self.current_tetrimino.copy()

        # Simulate placing the current tetromino in the grid without locking it
        simulated_grid = self.simulate_tetrimino_placement(temp_grid, temp_tetrimino)

        # Calculate penalties and rewards based on the simulated state
        empty_spaces_created = self.count_holes(simulated_grid) - self.count_holes(temp_grid)
        fit_penalty = self.calculate_fit_penalty(simulated_grid)
        max_tetromino_height = self.calculate_max_tetromino_height(temp_tetrimino)

        lines_cleared_now = self.lines_cleared  # Assume lines cleared remains the same for simplicity

        # Calculate penalties and rewards
        empty_space_penalty = -2 * empty_spaces_created  # Penalty for creating more empty spaces
        height_penalty = -10 * max_tetromino_height
        line_clear_reward = self.calculate_line_clear_reward(lines_cleared_now)

        # Combine the rewards and penalties
        reward = fit_penalty + empty_space_penalty + height_penalty + line_clear_reward
        return reward


    def calculate_max_tetromino_height(self, tetrimino):
        """Calculate the height of the topmost block of the tetromino."""
        # Determine the highest y-value (row) that the tetromino occupies after being placed
        matrix = tetrimino['matrix']
        y_pos = tetrimino['y']

        max_height = 0
        for dy, row in enumerate(matrix):
            for dx, cell in enumerate(row):
                if cell:
                    max_height = max(max_height, y_pos + dy)

        # The distance from this maximum height to the ground (bottom of the grid)
        distance_to_ground = GRID_HEIGHT - max_height
        return distance_to_ground

    def simulate_tetrimino_placement(self, grid, tetrimino):
        """Simulates placing a tetromino on the grid without locking it."""
        matrix = tetrimino['matrix']
        x = tetrimino['x']
        y = tetrimino['y']

        for dy, row in enumerate(matrix):
            for dx, cell in enumerate(row):
                if cell:
                    grid[y + dy][x + dx] = tetrimino['color']

        return grid

    def calculate_fit_penalty(self, grid):
        """Penalize based on how well the tetromino fits into the current grid."""
        gaps = 0
        for y in range(1, GRID_HEIGHT):  # Start from 1 to avoid counting the top row
            for x in range(GRID_WIDTH):
                if grid[y][x] == 0 and grid[y - 1][x] != 0:
                    gaps += 1
        # Penalize for every gap that appears in the grid
        return -5 * gaps

    def calculate_line_clear_reward(self, lines_cleared):
        """Calculates the reward based on the number of lines cleared."""
        if lines_cleared == 1:
            return 100
        elif lines_cleared == 2:
            return 300
        elif lines_cleared == 3:
            return 500
        elif lines_cleared == 4:
            return 800
        else:
            return 0

    def calculate_placement_height(self, grid, tetrimino):
        """Calculate the height penalty based on how high the tetromino was placed."""
        min_height = GRID_HEIGHT
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[y][x] != 0:
                    min_height = min(min_height, y)
                    break

        # Height penalty based on the distance from the lowest empty row to the ground
        placement_height = GRID_HEIGHT - min_height
        return placement_height

    def count_holes(self, grid=None):
        """Calculate the number of empty spaces (holes) created by the placement of the tetromino."""
        if grid is None:
            grid = self.grid  # Use the current game grid if no grid is provided

        total_empty_spaces = 0

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y][x] == 0:
                    total_empty_spaces += 1

        return total_empty_spaces

    def calculate_terrain_features(self, grid=None):
        if grid is None:
            grid = self.grid  # Use the current game grid if no grid is provided

        heights = [0] * GRID_WIDTH
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[y][x]:
                    heights[x] = GRID_HEIGHT - y
                    break

        total_height = sum(heights)
        max_height = max(heights)
        min_height = min(heights)
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(GRID_WIDTH - 1))
        max_bumpiness = max(abs(heights[i] - heights[i + 1]) for i in range(GRID_WIDTH - 1))

        return bumpiness, total_height, max_height, min_height,max_bumpiness

class State:
    def __init__(self, grid, current_tetrimino):
        # Copy the grid to avoid altering the original one
        self.grid = [row[:] for row in grid]
        self.current_tetrimino = current_tetrimino

    # def __repr__(self):
    #     return f"State(tetrimino={self.current_tetrimino}, x={self.tetrimino_x}, y={self.tetrimino_y}, rotation={self.tetrimino_rotation})"


if __name__ == '__main__':
    game = Tetris(DQLAgent())
    game.run()
