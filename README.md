# IntroToAI-FinalProject

## 1. Introduction
Tetris is one of those classic games that everyone knows, but it’s also surprisingly challenging when you dig into it. The game’s simple concept—fitting falling blocks together—hides a lot of complexity. This complexity, combined with the need to make quick decisions as the game speeds up, makes Tetris a really interesting problem for AI.

In this project, we’re tackling the challenge of creating an AI that can play Tetris effectively. What makes this problem particularly interesting is that Tetris isn’t just a straightforward puzzle. The sequence of blocks (or tetrominoes) that appear is random, which means that our AI needs to be able to handle unpredictability and still perform well. Additionally, the sheer number of possible board configurations is massive, so finding the best move at any given time isn’t easy. Because of these factors, we decided Tetris would be a great way to test out different AI strategies.

Another layer of complexity comes from the scoring system in Tetris. The more rows you clear at once, the higher your score. This is something human players naturally aim for—stacking blocks in a way that lets them clear multiple rows at once. For our AI, learning this strategy is far from trivial. It requires the agent not just to avoid immediate game-over scenarios but also to plan ahead and make moves that will set up future multi-row clears. Getting the AI to think like a human in this way adds a significant challenge and makes the problem even more interesting.

We’re starting with deep Q-learning, a method from reinforcement learning that’s been successful in other games. The idea here is that the AI will learn from playing the game, gradually improving its understanding of what moves are likely to lead to better outcomes. Deep Q-learning uses a neural network to estimate the best actions to take, which is helpful in a game like Tetris where the number of possible moves and outcomes is huge.

-- TO ADD : another model --