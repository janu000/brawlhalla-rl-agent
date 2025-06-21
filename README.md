# Brawlhalla RL Agent

A project to train an artificial intelligence (AI) agent to play the video game **Brawlhalla** using deep reinforcement learning. This project captures the game screen, interprets what is happening, and controls the game using simulated keyboard and mouse input—just like a human player would!

---

## What is This?

This project is a research and hobbyist framework for teaching computers to play Brawlhalla, a popular platform fighting game. It uses modern AI techniques to learn how to play by watching the screen and trying different actions, improving over time based on its performance.

- **No game files are modified.**
- **Works by watching the screen and pressing keys.**
- **Can be used for AI research, fun experiments, or learning about reinforcement learning.**

---

## Features

- **Custom Game Environment**: The AI sees the game just like a player—via screen capture.
- **Deep RL with Memory**: Uses advanced neural networks (with LSTM memory) to handle the fast-paced, partially observable nature of Brawlhalla.
- **Flexible Actions**: Supports a wide range of moves, from simple jumps to complex attack combos.
- **Automatic Health Tracking**: Reads the health bars from the screen to know how well it's doing.
- **Progress Saving**: Regularly saves its progress so you can pause and resume training.
- **Visual Logging**: Supports TensorBoard for visualizing training progress.

---

## Who is This For?

- **AI/ML Enthusiasts**: Learn about reinforcement learning in a real-world, visual setting.
- **Gamers & Modders**: Curious how bots can learn to play games from scratch.
- **Researchers & Students**: A hands-on project for computer vision, RL, and automation.
- **Anyone** who wants to see AI learn to play a game by itself!

---

## Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **Brawlhalla** installed and running on your PC (Fullscreen, optimally on a secondary monitor)

### 2. Install Dependencies

Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```
You may also need:
```bash
pip install torch stable-baselines3 sb3-contrib tqdm
```

### 3. Run the AI Trainer

Open Brawlhalla and start a training match with a bot, then in your terminal:
```bash
python main.py
```
- The script will prompt you to switch to Brawlhalla.
- The AI will start learning to play, saving its progress as it goes.

---

## How Does It Work?

1. **Screen Capture**: The AI takes screenshots of the game window.
2. **Observation**: It processes these images to "see" the game state.
3. **Action**: It decides what keys to press (move, jump, attack, etc.) using a neural network.
4. **Feedback**: It checks the health bars to see if its actions were good or bad.
5. **Learning**: Over time, it gets better at choosing actions that lead to winning.

---

## Project Structure

```
brawlhalla-rl-agent/
├── BrawlhallaEnv.py         # The game environment for the AI
├── Policy.py                # The AI's brain (neural network)
├── controller.py            # Simulates keyboard/mouse input
├── ScreenRecorder.py        # Captures the game screen
├── config.py                # Settings and parameters
├── main.py                  # Where training starts
├── checkpoints/             # Saved AI models
├── ppo_brawlhalla_logs/     # Training logs for visualization
├── healthcolors.txt         # Color data for health bar tracking
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Customization & Advanced Use

- **Change Actions**: Edit `config.py` to add or remove moves.
- **Tune AI**: Adjust learning rates, batch sizes, and more in `config.py`.
- **Health Bar Positions**: If your game window is in a different spot, update `HEALTH1_POS` and `HEALTH2_POS` in `config.py`.
- **TensorBoard**: Run `tensorboard --logdir ppo_brawlhalla_logs/` to visualize training progress in your browser.

---

## Troubleshooting

- **Low FPS?** Close other apps and make sure Brawlhalla is not minimized. Cosider a smaller Feature Extractor Network defined in `Policy.py`
- **Health not detected?** Adjust health bar positions. You can use the `temp/mouse_tracker.py` to find the positions on your screen
- **Keyboard/mouse not working?** Try running as administrator.

---

## License

MIT License. Use, modify, and share freely!

---

## Credits & Acknowledgements

- Built with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), [sb3_contrib](https://github.com/Stable-Baselines-Team/sb3_contrib), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and [Brawlhalla](https://www.brawlhalla.com/).
- Inspired by the open-source AI and gaming communities.

---

*Have fun watching your AI learn to brawl!*