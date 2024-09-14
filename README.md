# IntroToAI - FinalProject - Tetris AI

## Overview
This project implements AI agents for playing the classic game of Tetris. Our goal is to develop a system that can arrange falling blocks in the best possible way to maximize the score by clearing as many rows as possible. We implemented two models to solve this problem: a Genetic Algorithm and a Deep Q-Learning (DQL) Agent.

## Demo

## Requirements
- Numpy
- Tensorflow
- Pygame
  
## Installation

To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/uzano101/tetris-ai-project.git

2. Install requirements:
   ```bash
   pip install -r requirements.txt

4. Run the project:
   ```bash
   python Main.py

## Models
1. **Genetic Algorithm**:
   - Uses evolution principles like selection, crossover, and mutation to improve decision-making over generations.
   - Heuristics include: Aggregate Height, Holes, Complete Lines, Bumpiness, Highest Point, and Neighbors.

2. **Deep Q-Learning (DQL) Agent**:
   - Uses a neural network to predict the best next state, balancing exploration and exploitation.
   - The agent learns from experience by storing game states and updating the Q-values using the Bellman Equation.

### Implemantation
- **Genetic Algorithm**: Optimizes weights for heuristics over multiple generations.
- **DQL Agent**: Learns from game experience and predicts the optimal move for each state.

### Results


