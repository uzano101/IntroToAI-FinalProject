import pygame
import random
import time

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
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris')

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.reset_game()

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.speed = 500  # Milliseconds per fall
        self.last_fall_time = pygame.time.get_ticks()

        self.next_tetrimino = self.get_random_tetrimino()
        self.spawn_tetrimino()
        self.current_state = self.get_current_state()  # Save the current state

    def get_current_state(self):
        return State(self.grid, self.current_tetrimino['shape'], self.current_tetrimino['x'],
                     self.current_tetrimino['y'], 0)

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

    def move_tetrimino(self, dx, dy):
        new_x = self.current_tetrimino['x'] + dx
        new_y = self.current_tetrimino['y'] + dy
        if not self.check_collision(new_x, new_y, self.current_tetrimino['matrix']):
            self.current_tetrimino['x'] = new_x
            self.current_tetrimino['y'] = new_y
            return True
        elif dy == 1:
            self.lock_tetrimino()
            return False
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

    def update(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fall_time > self.speed:
            self.move_tetrimino(0, 1)
            self.last_fall_time = current_time

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.move_tetrimino(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    self.move_tetrimino(1, 0)
                # elif event.key == pygame.K_DOWN:
                #     self.move_tetrimino(0, 1)
                elif event.key == pygame.K_UP:
                    self.rotate_tetrimino()
                # elif event.key == pygame.K_SPACE:
                #     self.hard_drop()
        return True

    def get_next_states(self):
        next_states = []
        current_shape = self.current_tetrimino['shape']
        current_matrix = self.current_tetrimino['matrix']

        # Explore all possible rotations
        for rotation in range(4):
            rotated_matrix = self.rotate_matrix(current_matrix, rotation)

            # Explore all possible horizontal positions
            for x_pos in range(-len(rotated_matrix[0]) + 1, GRID_WIDTH):
                if not self.check_collision(x_pos, self.current_tetrimino['y'], rotated_matrix):
                    # Create a copy of the grid
                    new_grid = [row[:] for row in self.grid]

                    # Simulate the tetromino placement in this state
                    for y, row in enumerate(rotated_matrix):
                        for x, cell in enumerate(row):
                            if cell:
                                new_grid[self.current_tetrimino['y'] + y][x_pos + x] = self.current_tetrimino['color']

                    # Create a new state and add it to the list
                    next_states.append(State(new_grid, current_shape, x_pos, self.current_tetrimino['y'], rotation))

        return next_states

    def rotate_matrix(self, matrix, times=1):
        # Rotate the matrix 90 degrees clockwise `times` number of times
        for _ in range(times):
            matrix = [list(row) for row in zip(*matrix[::-1])]
        return matrix

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.screen.fill(BLACK)
            self.draw_frame()
            self.draw_grid()
            self.draw_tetrimino()
            self.draw_ui()
            self.draw_next_tetrimino()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

class State:
    def __init__(self, grid, current_tetrimino, tetrimino_x, tetrimino_y, tetrimino_rotation):
        # Copy the grid to avoid altering the original one
        self.grid = [row[:] for row in grid]
        self.current_tetrimino = current_tetrimino
        self.tetrimino_x = tetrimino_x
        self.tetrimino_y = tetrimino_y
        self.tetrimino_rotation = tetrimino_rotation

    def __repr__(self):
        return f"State(tetrimino={self.current_tetrimino}, x={self.tetrimino_x}, y={self.tetrimino_y}, rotation={self.tetrimino_rotation})"


if __name__ == '__main__':
    game = Tetris()
    game.run()
