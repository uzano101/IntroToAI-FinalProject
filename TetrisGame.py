import csv
import time
import pygame
import random
from DQLAgent import DQLAgent
from GeneticAgent import GeneticAgent

pygame.init()

DQL_AGENT = "DQL"
GENETIC_AGENT = "GENETIC"
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 25
CUBE_SIZE = 24
FRAME_WIDTH = 4
INTERNAL_PADDING = 1
UI_WIDTH = 260
GRID_PIXEL_WIDTH = GRID_WIDTH * BLOCK_SIZE + 2 * INTERNAL_PADDING
GRID_PIXEL_HEIGHT = GRID_HEIGHT * BLOCK_SIZE + 2 * INTERNAL_PADDING
SCREEN_WIDTH = GRID_PIXEL_WIDTH + UI_WIDTH + 2 * FRAME_WIDTH
SCREEN_HEIGHT = GRID_PIXEL_HEIGHT + 2 * FRAME_WIDTH
CUBE_OFFSET = (BLOCK_SIZE - CUBE_SIZE) // 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (50, 50, 50)
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
LEGO_COLORS = [(233, 30, 99), (33, 150, 243), (255, 193, 7), (76, 175, 80), (255, 87, 34), (156, 39, 176),
               (255, 152, 0)]
BACKGROUND_COLOR = (240, 240, 240)
TEXT_COLOR = (0, 0, 0)
OUTLINE_COLOR = (0, 0, 0)
COLORS = {
    'I': LEGO_COLORS[0],
    'J': LEGO_COLORS[1],
    'L': LEGO_COLORS[2],
    'O': LEGO_COLORS[3],
    'S': LEGO_COLORS[4],
    'T': LEGO_COLORS[5],
    'Z': LEGO_COLORS[6],
}
headline_font = pygame.font.SysFont('Bebas Neue', 80)
small_font = pygame.font.SysFont('Bebas Neue', 35, bold=True)
lego_background_image = pygame.image.load('backgroung.jpg')


def draw_background(screen):
    screen.blit(pygame.transform.scale(lego_background_image, (SCREEN_WIDTH, SCREEN_HEIGHT)), (0, 0))


class Button:
    """Class for creating buttons in the game."""
    def __init__(self, text, x, y, width, height, color, hover_color, action=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.action = action

    def draw(self, screen):
        """ Draws the button on the screen. """
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, OUTLINE_COLOR, self.rect, 3, border_radius=5)
        text_surf = small_font.render(self.text, True, TEXT_COLOR)
        screen.blit(text_surf, (self.rect.x + (self.rect.width - text_surf.get_width()) // 2,
                                self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

    def is_hovered(self, pos):
        """ Checks if the mouse is hovering over the button. """
        return self.rect.collidepoint(pos)

    def update(self, pos):
        """ Updates the button's color based on the mouse position. """
        if self.is_hovered(pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color

    def check_click(self, pos):
        """ Checks if the button was clicked and performs the action if it was. """
        if self.is_hovered(pos) and self.action:
            self.action()


def exit_game():
    """ Exits the game, for the exit button. """
    global exit_game_flag
    exit_game_flag = True


exit_game_flag = False
exit_button = Button('Exit', SCREEN_WIDTH - 180, SCREEN_HEIGHT - 45, 80, 40, LEGO_COLORS[0],
                     (244, 67, 54), exit_game)


class Tetris:
    """ Main class for the Tetris game. """
    def __init__(self, agent):
        self.chosen_agent = agent
        if agent == GENETIC_AGENT:
            self.agent = GeneticAgent()
        else:
            self.agent = DQLAgent()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris Lego Theme')
        self.current_tetrimino = None
        self.next_tetrimino = None
        self.continue_playing = False
        self.game_over = False
        self.font = small_font
        self.game_counter = 0
        self.previous_state = None
        self.current_state = None
        self.current_reward = 0
        self.current_action = None
        self.high_score = 0
        self.score = 0
        self.level = 0
        self.last_level = 0
        self.level_at_999999 = 0
        self.statistics = []
        self.start_time = time.time()
        self.num_tetriminoes_dropped = 0
        self.num_moves = 0
        self.reset_game()

    def export_statistics_to_csv(self):
        """Exports the current game statistics to a CSV file."""
        filename = f"tetris_game_statistics.csv"
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Game Number", "Score", "Lines Cleared", "Level", "Reward", "Total Time Played (s)",
                                 "Tetriminoes Dropped", "Moves Made", "Generation", "Weights", "LevelAt999999"])
                for stat in self.statistics:
                    writer.writerow(stat)
            print(f"Game statistics exported to {filename}")
        except PermissionError:
            print("Please close the file before continuing.")

    def reset_game(self):
        """ Resets the game to its initial state. """
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
        self.speed = 200
        self.game_counter += 1
        self.next_tetrimino = self.get_random_tetrimino()
        self.spawn_tetrimino()
        self.current_state = self.get_current_state()
        self.refresh_game()

    def get_current_state(self):
        """ Returns the current state of the game. """
        return State(self.grid, self.current_tetrimino)

    def get_random_tetrimino(self):
        """ Returns a random tetrimino. """
        shape = random.choice(list(TETRIMINOS.keys()))
        return {
            'shape': shape,
            'matrix': TETRIMINOS[shape],
            'color': COLORS[shape],
            'x': GRID_WIDTH // 2 - len(TETRIMINOS[shape][0]) // 2,
            'y': 0
        }

    def spawn_tetrimino(self):
        """ Spawns a new tetrimino. """
        self.current_tetrimino = self.next_tetrimino
        self.next_tetrimino = self.get_random_tetrimino()
        if self.check_collision(self.current_tetrimino['x'], self.current_tetrimino['y'],
                                self.current_tetrimino['matrix']):
            self.game_over = True
            return
        self.get_next_tetrimino_place_by_agent()
        self.current_state = self.get_current_state()

    def check_collision(self, x_offset, y_offset, matrix):
        """ Checks if the tetrimino collides with the grid or other tetriminoes. """
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
        """ Lock the current tetrimino onto the grid and check its isolation properties. """
        for y, row in enumerate(self.current_tetrimino['matrix']):
            for x, cell in enumerate(row):
                if cell:
                    x_pos = x + self.current_tetrimino['x']
                    y_pos = y + self.current_tetrimino['y']
                    if y_pos >= 0:
                        self.grid[y_pos][x_pos] = self.current_tetrimino['color']

    def renew_and_check_lines(self):
        """ Check if there are any lines to clear and update the score and level accordingly. """
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
        """ Rotates the current tetrimino. """
        matrix = self.current_tetrimino['matrix']
        rotated_matrix = [list(row) for row in zip(*matrix[::-1])]
        if not self.check_collision(self.current_tetrimino['x'], self.current_tetrimino['y'], rotated_matrix):
            self.current_tetrimino['matrix'] = rotated_matrix
        self.current_state = self.get_current_state()

    def move_tetrimino(self, dx, dy):
        """ Moves the current tetrimino. """
        new_x = self.current_tetrimino['x'] + dx
        new_y = self.current_tetrimino['y'] + dy
        if not self.check_collision(new_x, new_y, self.current_tetrimino['matrix']):
            self.current_tetrimino['x'] = new_x
            self.current_tetrimino['y'] = new_y
            self.current_state = self.get_current_state()
            self.num_moves += 1
            return True
        elif dy == 1:
            self.lock_tetrimino()
        self.current_state = self.get_current_state()
        return False

    def draw_grid(self):
        """ Draws the grid on the screen. """
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect_x = FRAME_WIDTH + INTERNAL_PADDING + x * BLOCK_SIZE + CUBE_OFFSET
                rect_y = FRAME_WIDTH + INTERNAL_PADDING + y * BLOCK_SIZE + CUBE_OFFSET
                rect = pygame.Rect(rect_x, rect_y, CUBE_SIZE, CUBE_SIZE)
                color = self.grid[y][x]
                if color:
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, OUTLINE_COLOR, rect, 2)

    def draw_tetrimino(self):
        """ Draws the current tetrimino on the screen. """
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
                    pygame.draw.rect(self.screen, OUTLINE_COLOR, rect, 2)

    def draw_next_tetrimino(self):
        """ Draws the next tetrimino on the screen. """
        matrix = self.next_tetrimino['matrix']
        color = self.next_tetrimino['color']
        start_x = GRID_PIXEL_WIDTH + FRAME_WIDTH + 20
        start_y = 165
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    rect_x = start_x + x * BLOCK_SIZE + CUBE_OFFSET
                    rect_y = start_y + y * BLOCK_SIZE + CUBE_OFFSET
                    rect = pygame.Rect(rect_x, rect_y, CUBE_SIZE, CUBE_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, OUTLINE_COLOR, rect,
                                     2)

    def draw_frame(self):
        """ Draws the frame around the game grid. """
        draw_background(self.screen)
        rect = pygame.Rect(0, 0, GRID_PIXEL_WIDTH + 2 * FRAME_WIDTH, GRID_PIXEL_HEIGHT + 2 * FRAME_WIDTH)
        pygame.draw.rect(self.screen, OUTLINE_COLOR, rect, 5)
        inner_rect = pygame.Rect(FRAME_WIDTH, FRAME_WIDTH, GRID_PIXEL_WIDTH, GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, inner_rect)

    def draw_ui(self):
        """ Draws the user interface on the screen. """
        score_text = self.font.render('Score:', True, TEXT_COLOR)
        self.screen.blit(score_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 50))
        score_value = self.font.render(f'{self.score}', True, TEXT_COLOR)
        self.screen.blit(score_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 100, 50))

        level_text = self.font.render('Level:', True, TEXT_COLOR)
        self.screen.blit(level_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 90))
        level_value = self.font.render(f'{self.level}', True, TEXT_COLOR)
        self.screen.blit(level_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 100, 90))

        next_text = self.font.render('Next:', True, TEXT_COLOR)
        self.screen.blit(next_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 130))

        game_counter_text = self.font.render('Games:', True, TEXT_COLOR)
        self.screen.blit(game_counter_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 230))

        game_counter_value = self.font.render(f'{self.game_counter}', True, TEXT_COLOR)
        self.screen.blit(game_counter_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 120, 230))

        lines_text = self.font.render('Lines:', True, TEXT_COLOR)
        self.screen.blit(lines_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 280))

        lines_value = self.font.render(f'{self.lines_cleared}', True, TEXT_COLOR)
        self.screen.blit(lines_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 100, 280))

        high_score_text = self.font.render('High Score:', True, TEXT_COLOR)
        self.screen.blit(high_score_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 330))

        high_score_value = self.font.render(f'{self.high_score}', True, TEXT_COLOR)
        self.screen.blit(high_score_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 360))

        highest_level_text = self.font.render('Highest Level:', True, TEXT_COLOR)
        self.screen.blit(highest_level_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 390))

        highest_level_value = self.font.render(f'{self.last_level}', True, TEXT_COLOR)
        self.screen.blit(highest_level_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 190, 390))

        if isinstance(self.agent, GeneticAgent):
            generation_text = self.font.render('Generation:', True, TEXT_COLOR)
            self.screen.blit(generation_text, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 20, 440))
            generation_value = self.font.render(f'{self.agent.generation}', True, TEXT_COLOR)
            self.screen.blit(generation_value, (GRID_PIXEL_WIDTH + FRAME_WIDTH + 170, 440))

    def get_next_tetrimino_place_by_agent(self):
        """ Gets the best place for the next tetrimino using the agent. """
        lock_state = self.agent.choose_best_final_state(self.get_all_successor_states())
        self.previous_state = self.current_state
        self.set_tetrimino_to_state(lock_state)

    def handle_events_and_move(self):
        """ Handles the game events and moves the tetrimino. """
        global exit_game_flag
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.export_statistics_to_csv()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                exit_button.check_click(mouse_pos)
        if exit_game_flag:
            pygame.quit()
        self.hard_drop()
        self.current_state = self.get_current_state()

    def hard_drop(self):
        """ Drops the tetrimino to the bottom of the grid. """
        moved = True
        while moved:
            moved = self.move_tetrimino(0, 1)
            self.refresh_game()
            pygame.time.delay(5)

    def update_agent(self):
        """ Updates the agent """
        if self.previous_state is not None:
            self.agent.update_agent(self.previous_state, self.current_state, self.game_over)

    def refresh_game(self):
        """ Refreshes the game screen. """
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_frame()
        self.draw_grid()
        self.draw_tetrimino()
        self.draw_ui()
        self.draw_next_tetrimino()
        mouse_pos = pygame.mouse.get_pos()
        exit_button.update(mouse_pos)
        exit_button.draw(self.screen)
        pygame.display.flip()

    def run(self):
        """ Main game loop. """
        while not self.continue_playing and not exit_game_flag:
            if not self.game_over and self.score <= 99999999:
                if self.level_at_999999 == 0 and self.score >= 99999999:
                    self.level_at_999999 = self.level
                self.current_state = self.get_current_state()
                self.continue_playing = self.handle_events_and_move()
                self.game_over = self.is_game_over()
                self.refresh_game()
                if self.chosen_agent == DQL_AGENT:
                    self.update_agent()
                self.finish_turn_and_prepere_to_next_one()
            else:
                self.record_game()
                self.game_over = False
                self.score = min(self.score, 99999999)
                if self.chosen_agent == DQL_AGENT:
                    self.agent.train()
                else:
                    self.agent.train(self.score)
                self.previous_state = None
                self.reset_game()
        pygame.quit()

    def finish_turn_and_prepere_to_next_one(self):
        """ Finishes the current turn and prepares for the next one. """
        self.renew_and_check_lines()
        self.spawn_tetrimino()
        self.num_tetriminoes_dropped += 1

    def record_game(self):
        """ Records the game statistics. """
        elapsed_time = time.time() - self.start_time
        weights = self.agent.current_weights if self.chosen_agent == GENETIC_AGENT else []
        self.statistics.append([
            self.game_counter,
            self.score,
            self.lines_cleared,
            self.level,
            round(elapsed_time, 2),
            self.num_tetriminoes_dropped,
            self.num_moves,
            self.agent.generation,
            weights,
            self.level_at_999999
        ])
        self.num_tetriminoes_dropped = 0
        self.num_moves = 0
        self.start_time = time.time()

    def is_game_over(self):
        """ Checks if the game is over. """
        for x in range(len(self.grid[0])):
            if self.grid[0][x] != 0:
                return True

    def rotate_matrix(self, matrix, times=1):
        """ Rotates the matrix. """
        for _ in range(times):
            matrix = [list(row) for row in zip(*matrix[::-1])]
        return matrix

    def get_all_successor_states(self):
        """ Returns all possible successor states of the current state. """
        lock_states = []
        initial_tetrimino = self.current_tetrimino.copy()
        original_matrix = initial_tetrimino['matrix']
        for rotation in range(4):
            rotated_matrix = self.rotate_matrix(original_matrix, times=rotation)
            temp_tetrimino = initial_tetrimino.copy()
            temp_tetrimino['matrix'] = rotated_matrix
            for x in range(0, GRID_WIDTH - len(rotated_matrix[0]) + 1):
                temp_tetrimino['x'] = x
                temp_tetrimino['y'] = 0
                while not self.check_collision(temp_tetrimino['x'], temp_tetrimino['y'] + 1, rotated_matrix):
                    temp_tetrimino['y'] += 1
                if self.check_collision(temp_tetrimino['x'], temp_tetrimino['y'] + len(rotated_matrix) - 1,
                                        rotated_matrix):
                    grid_copy = [row[:] for row in self.grid]
                    self.fake_lock_tetrimino_in_grid(grid_copy, temp_tetrimino)
                    lock_states.append(State(grid_copy, temp_tetrimino.copy()))
        return lock_states

    def fake_lock_tetrimino_in_grid(self, grid, tetrimino):
        """ Locks the tetrimino in the grid without actually changing the grid. """
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
        """ Sets the tetrimino to the desired state. """
        desired_x = lock_state.current_tetrimino['x']
        desired_matrix = lock_state.current_tetrimino['matrix']
        rotation_attempts = 0
        while self.current_tetrimino['matrix'] != desired_matrix and rotation_attempts < 4:
            self.rotate_tetrimino()
            rotation_attempts += 1
        if rotation_attempts >= 4:
            return
        move_attempts = 0
        max_moves = abs(self.current_tetrimino['x'] - desired_x) + 2
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
        self.grid = [row[:] for row in grid]
        self.current_tetrimino = current_tetrimino

