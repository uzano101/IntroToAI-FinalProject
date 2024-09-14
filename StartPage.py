import sys
import pygame
import random

from DQLAgent import DQLAgent
from GeneticAgent import GeneticAgent
from TetrisGame import Tetris

pygame.init()

DQL_AGENT = "DQL"
GENETIC_AGENT = "GENETIC"

SCREEN_WIDTH = 261 * 2
SCREEN_HEIGHT = 365 * 1.5

LEGO_COLORS = [(233, 30, 99), (33, 150, 243), (255, 193, 7), (76, 175, 80), (255, 87, 34)]
LEGO_HOVER_COLORS = [(244, 143, 177), (100, 181, 246), (255, 213, 79), (129, 199, 132), (255, 138, 101)]
BUTTON_COLOR = LEGO_COLORS[0]
BUTTON_HOVER_COLOR = (255, 64, 129)
SLIDER_COLOR = LEGO_COLORS[1]
SLIDER_HANDLE_COLOR = LEGO_COLORS[2]
TEXT_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (240, 240, 240)

headline_font = pygame.font.SysFont('Bebas Neue', 100)
button_font = pygame.font.SysFont('Bebas Neue', 35)
message_font = pygame.font.SysFont('Bebas Neue', 50)
small_font = pygame.font.SysFont('Bebas Neue', 30, bold=True)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Tetris Lego Theme')

lego_background_image = pygame.image.load('backgroung.jpg')


def draw_background():
    screen.blit(pygame.transform.scale(lego_background_image, (SCREEN_WIDTH, SCREEN_HEIGHT)), (0, 0))


class Button:
    def __init__(self, text, x, y, width, height, color, hover_color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.current_color = color

    def draw(self):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 4, border_radius=5)
        text_surf = button_font.render(self.text, True, TEXT_COLOR)
        screen.blit(text_surf, (self.rect.x + (self.rect.width - text_surf.get_width()) // 2,
                                self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

    def is_hovered(self, pos):
        return self.rect.collidepoint(pos)

    def update(self, pos):
        if self.is_hovered(pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color


class Slider:
    def __init__(self, x, y, width, min_val, max_val, initial_val, integer=False):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.integer = integer
        self.slider_rect = pygame.Rect(x + (initial_val - min_val) / (max_val - min_val) * width - 10, y - 5, 20,
                                       20)
        self.dragging = False

    def draw(self):
        pygame.draw.rect(screen, SLIDER_COLOR, self.rect, border_radius=10)
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, self.slider_rect, border_radius=5)
        pygame.draw.rect(screen, (0, 0, 0), self.slider_rect, 2, border_radius=5)
        value_display = int(self.value) if self.integer else f'{self.value:.2f}'
        value_text = small_font.render(str(value_display), True, TEXT_COLOR)
        screen.blit(value_text, (self.rect.x + 170, self.rect.y - 5))

    def move(self, pos):
        if self.dragging:
            self.slider_rect.x = max(self.rect.x - 10, min(pos[0], self.rect.x - 10 + self.rect.width))
            raw_value = self.min_val + (self.slider_rect.x - self.rect.x + 10) / self.rect.width * (
                    self.max_val - self.min_val)
            self.value = int(raw_value) if self.integer else raw_value

    def check_for_input(self, pos):
        if self.slider_rect.collidepoint(pos):
            self.dragging = True

    def release(self):
        self.dragging = False


play_button = Button('Start', SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT - 50, 150, 40, LEGO_COLORS[3],
                     LEGO_HOVER_COLORS[3])
quit_button = Button('Quit', SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 50, 150, 40, BUTTON_COLOR, BUTTON_HOVER_COLOR)

dql_button = Button('Deep Q-Learning', SCREEN_WIDTH // 2 - 240, 200, 230, 40, LEGO_COLORS[1], LEGO_HOVER_COLORS[1])
genetic_button = Button('Genetic', SCREEN_WIDTH // 2 + 10, 200, 230, 40, LEGO_COLORS[2], LEGO_HOVER_COLORS[2])

alpha_slider = Slider(SCREEN_WIDTH // 2, 300, 150, 0, 1.0, 0.5)
gamma_slider = Slider(SCREEN_WIDTH // 2, 350, 150, 0, 1.0, 0.5)
alpha_decay_slider = Slider(SCREEN_WIDTH // 2, 400, 150, 0.001, 0.999, 0.5)
batch_size_slider = Slider(SCREEN_WIDTH // 2, 450, 150, 1, 128, 64, integer=True)  # Batch size slider as integer

generations_slider = Slider(SCREEN_WIDTH // 2 - 20, 300, 150, 10, 100, 50, integer=True)

selected_agent = None


def draw_text(text, font, color, x, y):
    text_surf = font.render(text, True, color)
    screen.blit(text_surf, (x, y))


def start_page():
    running = True
    global selected_agent
    colors = [LEGO_COLORS[random.randint(0, 4)] for i in range(6)]

    while running:
        draw_background()
        draw_text('T', headline_font, colors[0], SCREEN_WIDTH // 2 - 130, 50)
        draw_text('E', headline_font, colors[1], SCREEN_WIDTH // 2 - 83, 50)
        draw_text('T', headline_font, colors[2], SCREEN_WIDTH // 2 - 30, 50)
        draw_text('R', headline_font, colors[3], SCREEN_WIDTH // 2 + 20, 50)
        draw_text('I', headline_font, colors[4], SCREEN_WIDTH // 2 + 70, 50)
        draw_text('S', headline_font, colors[5], SCREEN_WIDTH // 2 + 90, 50)

        if not selected_agent:
            message_colors = [LEGO_COLORS[random.randint(0, 4)] for i in range(6)]
            draw_text('Select an Agent', message_font, message_colors[random.randint(0, 5)], SCREEN_WIDTH // 2 - 130,
                      150)
        mouse_pos = pygame.mouse.get_pos()
        play_button.update(mouse_pos)
        quit_button.update(mouse_pos)
        dql_button.update(mouse_pos)
        genetic_button.update(mouse_pos)

        play_button.draw()
        quit_button.draw()
        dql_button.draw()
        genetic_button.draw()

        if selected_agent == 'DQL':
            draw_text('Epsilon:', small_font, TEXT_COLOR, SCREEN_WIDTH // 2 - 230, 295)
            alpha_slider.draw()
            draw_text('Gamma:', small_font, TEXT_COLOR, SCREEN_WIDTH // 2 - 230, 345)
            gamma_slider.draw()
            draw_text('Epsilon Decay Factor:', small_font, TEXT_COLOR, SCREEN_WIDTH // 2 - 230, 395)
            alpha_decay_slider.draw()
            draw_text('Batch Size:', small_font, TEXT_COLOR, SCREEN_WIDTH // 2 - 230, 445)
            batch_size_slider.draw()

        elif selected_agent == 'Genetic':
            draw_text('Population:', small_font, TEXT_COLOR, SCREEN_WIDTH // 2 - 200, 295)
            generations_slider.draw()

        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.is_hovered(mouse_pos):
                    start_game()
                elif quit_button.is_hovered(mouse_pos):
                    pygame.quit()
                    sys.exit()
                elif dql_button.is_hovered(mouse_pos):
                    selected_agent = 'DQL'
                elif genetic_button.is_hovered(mouse_pos):
                    selected_agent = 'Genetic'
                elif selected_agent == 'DQL':
                    alpha_slider.check_for_input(mouse_pos)
                    gamma_slider.check_for_input(mouse_pos)
                    alpha_decay_slider.check_for_input(mouse_pos)
                    batch_size_slider.check_for_input(mouse_pos)
                elif selected_agent == 'Genetic':
                    generations_slider.check_for_input(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                alpha_slider.release()
                gamma_slider.release()
                alpha_decay_slider.release()
                batch_size_slider.release()
                generations_slider.release()

        if selected_agent == 'DQL':
            alpha_slider.move(mouse_pos)
            gamma_slider.move(mouse_pos)
            alpha_decay_slider.move(mouse_pos)
            batch_size_slider.move(mouse_pos)
        elif selected_agent == 'Genetic':
            generations_slider.move(mouse_pos)

        pygame.display.flip()


def start_game():
    global selected_agent

    if selected_agent == 'DQL':
        alpha = alpha_slider.value
        gamma = gamma_slider.value
        alpha_decay = alpha_decay_slider.value
        batch_size = int(batch_size_slider.value)
        agent = DQLAgent(gamma=gamma, epsilon=alpha, epsilon_decay=alpha_decay, batch_size=batch_size)
        game = Tetris(DQL_AGENT)
        game.agent = agent
    elif selected_agent == 'Genetic':
        generations = int(generations_slider.value)
        agent = GeneticAgent(population_size=generations)
        game = Tetris(GENETIC_AGENT)
        game.agent = agent
    else:
        return

    game.run()


if __name__ == '__main__':
    start_page()
