import pygame
import random
import time

pygame.init()

grid_width = 16
grid_height = 20

screen_width = 400
screen_height = 500

block_size = screen_width // grid_width

screen = pygame.display.set_mode((screen_width, screen_height))

black = (0, 0, 0)
white = (255, 255, 255)

colors = {
    'T': (255, 0, 0),
    'S': (0, 255, 0),
    'Z': (0, 0, 255),
    'I': (255, 255, 0),
    'O': (0, 255, 255),
    'L': (255, 0, 255),
    'J': (255, 165, 0)
}

tetriminos = {
    'T': [[1, 1, 1], [0, 1, 0]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'Z': [[1, 1, 0], [0, 1, 1]],
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'L': [[1, 1, 1], [1, 0, 0]],
    'J': [[1, 1, 1], [0, 0, 1]]
}


class Tetris:
    def __init__(self):
        self.falling_rate = 0.3
        self.tetrimino_y = None
        self.tetrimino_x = None
        self.tetrimino_color = None
        self.current_tetrimino = None
        self.current_shape = None
        self.grid = [[0] * grid_width for _ in range(grid_height)]
        self.new_tetrimino()
        self.last_fall_time = time.time()
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.font = pygame.font.Font(None, 36)

    def new_tetrimino(self):
        self.current_shape = random.choice(list(tetriminos.keys()))
        self.current_tetrimino = tetriminos[self.current_shape]
        self.tetrimino_color = colors[self.current_shape]
        self.tetrimino_x = grid_width // 2 - len(self.current_tetrimino[0]) // 2
        self.tetrimino_y = 0

    def rotate_tetrimino(self):
        rotated = [list(row) for row in zip(*self.current_tetrimino[::-1])]
        if not self.check_collision(0, 0, rotated):
            self.current_tetrimino = rotated

    def move(self, dx, dy):
        if not self.check_collision(dx, dy):
            self.tetrimino_x += dx
            self.tetrimino_y += dy
        elif dy == 1:
            self.stack_tetrimino()
            self.new_tetrimino()

    def fall(self):
        self.move(0, 1)

    def check_collision(self, dx, dy, shape=None):
        if shape is None:
            shape = self.current_tetrimino
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.tetrimino_x + x + dx
                    new_y = self.tetrimino_y + y + dy
                    if new_x < 0 or new_x >= grid_width or new_y >= grid_height or self.grid[new_y][new_x]:
                        return True
        return False

    def stack_tetrimino(self):
        for y, row in enumerate(self.current_tetrimino):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[self.tetrimino_y + y][self.tetrimino_x + x] = self.tetrimino_color
        self.clear_lines()

    def clear_lines(self):
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        cleared_lines = grid_height - len(new_grid)
        self.lines_cleared += cleared_lines
        self.update_score(cleared_lines)
        if cleared_lines > 0:
            self.update_level()
        new_grid = [[0] * grid_width for _ in range(cleared_lines)] + new_grid
        self.grid = new_grid

    def update_score(self, lines):
        if lines == 1:
            self.score += 40 * self.level
        elif lines == 2:
            self.score += 100 * self.level
        elif lines == 3:
            self.score += 300 * self.level
        elif lines == 4:
            self.score += 1200 * self.level

    def update_level(self):
        self.level = (self.lines_cleared // 10) + 1

    def draw_grid(self):
        for y in range(grid_height):
            for x in range(grid_width):
                color = self.grid[y][x] if self.grid[y][x] else black
                pygame.draw.rect(screen, color, (x * block_size, y * block_size, block_size, block_size), 0)
                if self.grid[y][x]:
                    pygame.draw.rect(screen, white, (x * block_size, y * block_size, block_size, block_size), 1)

    def draw_tetrimino(self):
        for y, row in enumerate(self.current_tetrimino):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, white,
                                     ((self.tetrimino_x + x) * block_size, (self.tetrimino_y + y) * block_size,
                                      block_size, block_size), 1)
                    pygame.draw.rect(screen, self.tetrimino_color,
                                     ((self.tetrimino_x + x) * block_size + 1, (self.tetrimino_y + y) * block_size + 1,
                                      block_size - 2, block_size - 2), 0)

    def draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, white)
        level_text = self.font.render(f"Level: {self.level}", True, white)
        screen.blit(score_text, (screen_width - score_text.get_width() - 10, 10))
        screen.blit(level_text, (screen_width - level_text.get_width() - 10, 40))

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            current_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move(1, 0)
                    # elif event.key == pygame.K_DOWN:
                    #     self.fall()
                    elif event.key == pygame.K_UP:
                        self.rotate_tetrimino()

            if current_time - self.last_fall_time >= self.falling_rate:
                self.fall()
                self.last_fall_time = current_time

            screen.fill(black)
            self.draw_grid()
            self.draw_tetrimino()
            self.draw_score()
            pygame.display.flip()
            clock.tick(30)


game = Tetris()
game.run()
pygame.quit()
