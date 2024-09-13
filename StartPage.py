import csv
import sys
import time
import pygame
import random
from DQLAgent import DQLAgent
from GeneticAgent import GeneticAgent
from TetrisGame import Tetris

pygame.init()

# Agents
DQL_AGENT = "DQL"
GENETIC_AGENT = "GENETIC"

# Configuration
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 25
CUBE_SIZE = 20
FRAME_WIDTH = 4
INTERNAL_PADDING = 2
UI_WIDTH = 260

# Calculated dimensions
GRID_PIXEL_WIDTH = GRID_WIDTH * BLOCK_SIZE + 2 * INTERNAL_PADDING
GRID_PIXEL_HEIGHT = GRID_HEIGHT * BLOCK_SIZE + 2 * INTERNAL_PADDING
SCREEN_WIDTH = GRID_PIXEL_WIDTH + UI_WIDTH + 2 * FRAME_WIDTH
SCREEN_HEIGHT = GRID_PIXEL_HEIGHT + 2 * FRAME_WIDTH
CUBE_OFFSET = (BLOCK_SIZE - CUBE_SIZE) // 2  # Offset to center the cube in the block

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Fonts
font = pygame.font.SysFont('Arial', 32)
small_font = pygame.font.SysFont('Arial', 24)

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Tetris')

# Button class to create interactive buttons
class Button:
    def __init__(self, text, x, y, width, height, color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = font.render(self.text, True, WHITE)
        screen.blit(text_surf, (self.rect.x + (self.rect.width - text_surf.get_width()) // 2,
                                self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Slider class to create interactive sliders
class Slider:
    def __init__(self, x, y, width, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.slider_rect = pygame.Rect(x + (initial_val - min_val) / (max_val - min_val) * width, y - 10, 10, 30)

    def draw(self):
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, RED, self.slider_rect)
        # Draw current value of the slider
        value_text = small_font.render(f'{self.value:.2f}' if isinstance(self.value, float) else str(int(self.value)), True, WHITE)
        screen.blit(value_text, (self.slider_rect.x + 15, self.slider_rect.y - 10))

    def move(self, pos):
        if self.rect.collidepoint(pos):
            self.slider_rect.x = max(self.rect.x, min(pos[0], self.rect.x + self.rect.width))
            self.value = self.min_val + (self.slider_rect.x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)

# Initialize buttons and sliders
play_button = Button('Start', SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT - 100, 100, 50, GREEN)
quit_button = Button('Quit', SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 100, 100, 50, RED)

# Agent selection buttons
dql_button = Button('Deep Q-Learning', SCREEN_WIDTH // 2 - 230, 100, 220, 50, BLUE)
genetic_button = Button('Genetic', SCREEN_WIDTH // 2 + 20, 100, 200, 50, BLUE)

# Sliders for Deep Q-Learning parameters
alpha_slider = Slider(SCREEN_WIDTH // 2, 220, 200, 0, 1.0, 0.5)
gamma_slider = Slider(SCREEN_WIDTH // 2, 270, 200, 0, 1.0, 0.5)
alpha_decay_slider = Slider(SCREEN_WIDTH // 2, 320, 200, 0.001, 0.999, 0.5)
batch_size_slider = Slider(SCREEN_WIDTH // 2, 370, 200, 1, 128, 64)

# Slider for Genetic agent parameter
generations_slider = Slider(SCREEN_WIDTH // 2, 200, 200, 10, 100, 50)

# Flags to control the display of sections
selected_agent = None  # 'DQL' or 'Genetic'

def draw_text(text, font, color, x, y):
    text_surf = font.render(text, True, color)
    screen.blit(text_surf, (x, y))

def start_page():
    running = True
    global selected_agent

    while running:
        screen.fill(BLACK)
        draw_text('Tetris Game', font, WHITE, SCREEN_WIDTH // 2 - 100, 30)

        # Draw buttons
        play_button.draw()
        quit_button.draw()
        dql_button.draw()
        genetic_button.draw()

        # Draw configuration sections based on selected agent
        if selected_agent == 'DQL':
            draw_text('Alpha (Learning Rate):', small_font, WHITE, SCREEN_WIDTH // 2 - 250, 200)
            alpha_slider.draw()
            draw_text('Gamma :', small_font, WHITE, SCREEN_WIDTH // 2 - 250, 250)
            gamma_slider.draw()
            draw_text('Alpha Decay Factor:', small_font, WHITE, SCREEN_WIDTH // 2 - 250, 300)
            alpha_decay_slider.draw()
            draw_text('Batch Size:', small_font, WHITE, SCREEN_WIDTH // 2 - 250, 350)
            batch_size_slider.draw()

        elif selected_agent == 'Genetic':
            draw_text('Population:', small_font, WHITE, SCREEN_WIDTH // 2 - 250, 200)
            generations_slider.draw()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if play_button.is_clicked(mouse_pos):
                    start_game()
                elif quit_button.is_clicked(mouse_pos):
                    pygame.quit()
                    sys.exit()
                elif dql_button.is_clicked(mouse_pos):
                    selected_agent = 'DQL'
                elif genetic_button.is_clicked(mouse_pos):
                    selected_agent = 'Genetic'
                elif selected_agent == 'DQL':
                    # Controll the sliders using dragging the mouse
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
        return  # No agent selected

    game.run()

if __name__ == '__main__':
    start_page()
