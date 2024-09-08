import pygame
import random
from DQLAgent import DQLAgent
from GeneticAgent import GeneticAgent

pygame.init()
# agents
DQL_AGENT = "DQL"
GENETIC_AGENT = "GENETIC"

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

SCORES = {
    1: 40,
    2: 100,
    3: 300,
    4: 1200
}


class Tetris:
    def __init__(self, agent):
        self.chosen_agent = agent
        if agent == GENETIC_AGENT:
            self.agent = GeneticAgent()
        else:
            self.agent = DQLAgent()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris')
        self.current_tetrimino = None
        self.next_tetrimino = None
        self.continue_playing = False
        self.game_over = False
        self.font = pygame.font.Font(None, 36)
        self.game_counter = 0
        self.previous_state = None
        self.current_state = None
        self.current_reward = 0
        self.current_action = None
        self.high_score = 0
        self.score = 0
        self.level = 0
        self.last_level = 0
        self.reset_game()


    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        if self.high_score < self.score:
            self.high_score = self.score
            self.last_level = self.level
        if self.high_score == self.score:
            self.high_score = self.score
            self.last_level = min(self.last_level, self.level)
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.speed = 200  # Milliseconds per fall
        self.game_counter = self.game_counter + 1
        self.next_tetrimino = self.get_random_tetrimino()
        self.spawn_tetrimino()
        self.current_state = self.get_current_state()  # Save the current state
        self.refresh_game()

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
        self.get_next_tetrimino_place_by_agent()
        self.current_state = self.get_current_state()  # Update the current state

    def check_collision(self, x_offset, y_offset, matrix):
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    x_pos = x + x_offset
                    y_pos = y + y_offset
                    if x_pos < 0 or x_pos >= GRID_WIDTH or y_pos < 0 or y_pos >= GRID_HEIGHT or (
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

    def renew_and_check_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if 0 not in row]
        for i in lines_to_clear:
            del self.grid[i]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
        lines_cleared = len(lines_to_clear)
        if lines_cleared > 0:
            self.lines_cleared += lines_cleared
            self.score += SCORES[lines_cleared] * self.level
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
            self.current_state = self.get_current_state()
            return True
        elif dy == 1:
            self.lock_tetrimino()
            self.current_state = self.get_current_state()
            return False
        self.current_state = self.get_current_state()
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

        high_score_text = self.font.render(f'High Score:', True, WHITE)
        self.screen.blit(high_score_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 350))

        high_score_text = self.font.render(f'{self.high_score}', True, WHITE)
        self.screen.blit(high_score_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 380))

        last_level_text = self.font.render(f'Last Level:', True, WHITE)
        self.screen.blit(last_level_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 420))

        last_level = self.font.render(f'{self.last_level}', True, WHITE)
        self.screen.blit(last_level, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 450))

    def get_next_tetrimino_place_by_agent(self):
        if self.chosen_agent == DQL_AGENT:
            lock_state = self.agent.choose_best_final_state(self.current_state, self.get_all_successor_states())
        else:
            lock_state = self.agent.choose_best_final_state(self.get_all_successor_states())
        self.previous_state = self.current_state
        self.set_tetrimino_to_state(lock_state)

    def handle_events_and_move(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        self.hard_drop()
        self.current_state = self.get_current_state()

    def hard_drop(self):
        # Drops the current tetrimino straight down to its lowest possible position, updating the display on each drop.
        moved = True
        while moved:
            moved = self.move_tetrimino(0, 1)
            # Refresh the game display to show the tetrimino moving down
            self.refresh_game()
            # Add a short delay to make the movement visually smooth
            pygame.time.delay(5)

    def update_agent_thread(self):
        if self.previous_state is not None:
            self.agent.update_agent(self.previous_state, self.current_state, False, None)

    def refresh_game(self):
        self.screen.fill(BLACK)
        self.draw_frame()
        self.draw_grid()
        self.draw_tetrimino()
        self.draw_ui()
        self.draw_next_tetrimino()
        pygame.display.flip()

    def run(self):
        while not self.continue_playing:
            if not self.game_over and self.score <= 999999:
                self.current_state = self.get_current_state()
                self.continue_playing = self.handle_events_and_move()
                self.game_over = self.is_game_over()
                self.refresh_game()
                if self.chosen_agent is DQL_AGENT:
                    self.update_agent_thread()
                self.finish_turn_and_prepere_to_next_one()
            else:
                self.game_over = False
                self.score = min(self.score, 999999)
                if self.chosen_agent is DQL_AGENT:
                    self.agent.train()
                else:
                    self.agent.train(self.current_state, self.lines_cleared)
                self.previous_state = None
                self.reset_game()
        pygame.quit()

    def finish_turn_and_prepere_to_next_one(self):
        self.renew_and_check_lines()
        self.spawn_tetrimino()

    def is_game_over(self):
        for x in range(len(self.grid[0])):
            if self.grid[0][x] != 0:
                return True

    # def calculate_reward(self):
    #     if self.is_game_over():
    #         return -100
    #     # Constants for easy tuning
    #     a = -1  # Aggregate height
    #     b = 0.5  # Complete lines
    #     c = -0.8  # Holes
    #     d = -0.3  # Bumpiness
    #     e = -1  # New holes created
    #     f = -0.3  # Increase in bumpiness
    #     g = -0.5  # Height of the highest block
    #
    #     # Current state metrics
    #     aggregate_height = self.calculate_aggregate_height(self.grid)
    #     complete_lines = sum(1 for row in self.grid if 0 not in row)
    #     current_holes = self.calculate_holes(self.grid)
    #     current_bumpiness = self.calculate_bumpiness(self.grid)
    #     highest_point = self.calculate_highest_point(self.grid)
    #
    #     # Calculate changes from the previous state if available
    #     if self.previous_state:
    #         previous_holes = self.calculate_holes(self.previous_state.grid)
    #         previous_bumpiness = self.calculate_bumpiness(self.previous_state.grid)
    #         previous_highest_point = self.calculate_highest_point(self.previous_state.grid)
    #
    #         new_holes = max(0, current_holes - previous_holes)
    #         bumpiness_increase = max(0, current_bumpiness - previous_bumpiness)
    #         height_change = highest_point - previous_highest_point
    #     else:
    #         new_holes = 0
    #         bumpiness_increase = 0
    #         height_change = 0
    #
    #     # Total Reward Calculation
    #     total_reward = (a * aggregate_height) + (b * complete_lines) + (c * current_holes) + (d * current_bumpiness) + (
    #             e * new_holes) + (f * bumpiness_increase) + (g * height_change)
    #     return total_reward
    #
    # def calculate_aggregate_height(self, grid):
    #     return sum(GRID_HEIGHT - next((y for y, cell in enumerate(col) if cell), GRID_HEIGHT) for col in zip(*grid))
    #
    # def calculate_holes(self, grid):
    #     holes = 0
    #     for x in range(GRID_WIDTH):
    #         block_found = False
    #         for y in range(GRID_HEIGHT):
    #             if grid[y][x] != 0:
    #                 block_found = True
    #             elif block_found and grid[y][x] == 0:
    #                 holes += 1
    #     return holes
    #
    # def calculate_bumpiness(self, grid):
    #     column_heights = [GRID_HEIGHT - next((y for y, cell in enumerate(col) if cell), GRID_HEIGHT) for col in
    #                       zip(*grid)]
    #     return sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(len(column_heights) - 1))
    #
    # def calculate_highest_point(self, grid):
    #     for y in range(GRID_HEIGHT):
    #         if any(grid[y][x] != 0 for x in range(GRID_WIDTH)):
    #             return GRID_HEIGHT - y
    #     return GRID_HEIGHT  # Return max height if no blocks found

    def rotate_matrix(self, matrix, times=1):
        # Rotate the tetrimino matrix 90 degrees clockwise `times` number of times
        for _ in range(times):
            matrix = [list(row) for row in zip(*matrix[::-1])]
        return matrix

    def get_all_successor_states(self):
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
                    self.fake_lock_tetrimino_in_grid(grid_copy, temp_tetrimino)
                    lock_states.append(State(grid_copy, temp_tetrimino.copy()))

        return lock_states

    def fake_lock_tetrimino_in_grid(self, grid, tetrimino):
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

        desired_x = lock_state.current_tetrimino['x']
        desired_matrix = lock_state.current_tetrimino['matrix']

        # First, rotate the tetrimino to the desired orientation
        rotation_attempts = 0
        while self.current_tetrimino['matrix'] != desired_matrix and rotation_attempts < 4:
            self.rotate_tetrimino()
            rotation_attempts += 1

        if rotation_attempts >= 4:
            return  # Exit if cannot align rotation to avoid infinite loop

        # Move horizontally to the desired position
        move_attempts = 0
        max_moves = abs(self.current_tetrimino['x'] - desired_x) + 2  # Allow some extra attempts
        while self.current_tetrimino['x'] != desired_x and move_attempts < max_moves:
            move_direction = 1 if self.current_tetrimino['x'] < desired_x else -1
            if not self.check_collision(self.current_tetrimino['x'] + move_direction, self.current_tetrimino['y'],
                                        self.current_tetrimino['matrix']):
                self.current_tetrimino['x'] += move_direction
            else:
                break
            move_attempts += 1


class State:
    def __init__(self, grid, current_tetrimino):
        # Copy the grid to avoid altering the original one
        self.grid = [row[:] for row in grid]
        self.current_tetrimino = current_tetrimino


if __name__ == '__main__':
    game = Tetris(GENETIC_AGENT)
    game.run()
