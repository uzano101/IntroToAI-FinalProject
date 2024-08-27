# IntroToAI-FinalProject

## 1. Introduction
Tetris is one of those classic games that everyone knows, but it’s also surprisingly challenging when you dig into it. The game’s simple concept—fitting falling blocks together—hides a lot of complexity. This complexity, combined with the need to make quick decisions as the game speeds up, makes Tetris a really interesting problem for AI.

Our interest in this problem was sparked by a recent event. On December 21, 2023, a 13-year-old named Willis Gibson made headlines when he won the game by playing the old version of Tetris on Nintendo for about 38 minutes, eventually causing the game to crash. This impressive feat got us thinking: could we design an AI that could not only match but possibly outperform a human player like Willis? That challenge, combined with the desire to explore AI techniques, led us to choose Tetris as the focus of our project.

Another layer of complexity comes from the scoring system in Tetris. The more rows you clear at once, the higher your score. This is something human players naturally aim for—stacking blocks in a way that lets them clear multiple rows at once. For our AI, learning this strategy is far from trivial. It requires the agent not just to avoid immediate game-over scenarios but also to plan ahead and make moves that will set up future multi-row clears. Getting the AI to think like a human in this way adds a significant challenge and makes the problem even more interesting.

We’re starting with deep Q-learning, a method from reinforcement learning that’s been successful in other games. The idea here is that the AI will learn from playing the game, gradually improving its understanding of what moves are likely to lead to better outcomes. Deep Q-learning uses a neural network to estimate the best actions to take, which is helpful in a game like Tetris where the number of possible moves and outcomes is huge.
-- TO ADD : another model --