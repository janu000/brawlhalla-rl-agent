# ⚠️ DISCONTINUED: Brawlhalla RL Agent

> **Notice:** This project is no longer actively maintained or updated. The code is provided "as-is" for educational and archival purposes. It may require updates to work with the latest versions of Brawlhalla or Python dependencies.

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
- **Expert Data Collection & Pretraining**: Collect your own gameplay data and pretrain the agent using behavior cloning.
- **Efficient Data Loading**: Expert data is lazily loaded from disk to avoid memory issues.

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

### 3. Collect Expert Data (Optional but recommended)

Play the game yourself and record expert demonstrations:
```bash
python collect_expert_data.py
```
- This will record your gameplay (actions + screen images) as episodes in the `data/` folder.
- The script prints the FPS and saves each episode automatically.
- Data is stored efficiently and loaded lazily to avoid memory issues.

### 4. Pretrain the Agent with Behavior Cloning (Optional)

Use your collected expert data to pretrain the agent:
```bash
python bc_pretrain.py
```
- This will train the agent to imitate your actions using the expert data.
- Progress and losses (including log-likelihood and entropy) are logged to TensorBoard.
- You can monitor training with:
  ```bash
  tensorboard --logdir=training_logs/bc_pretrain
  ```
- The pretrained model is saved in `checkpoints/`.

### 5. Train the RL Agent

Open Brawlhalla and start a training match with a bot, then in your terminal:
```bash
python main.py
```
- The script will prompt you to switch to Brawlhalla.
- The AI will start learning to play, saving its progress as it goes.
- If a pretrained model exists, it will be loaded automatically.

### 6. View and Inspect Data

To view your collected expert data visually:
```bash
python data_viewer.py
```
- This will show the last two recorded episodes, with frame-by-frame navigation and action labels.

### 7. Run the Pretrained Model (No Training)

To run the agent using a pretrained model without further training, use `run.py`:
```bash
python run.py
```
- The agent will play the game using the loaded model.

---

## How Does It Work?

1. **Screen Capture**: The AI takes screenshots of the game window.
2. **Observation**: It processes these images to "see" the game state (now as image tensors only).
3. **Action**: It decides what keys to press (move, jump, attack, etc.) using a neural network.
4. **Feedback**: It checks the health bars to see if its actions were good or bad.
5. **Learning**: Over time, it gets better at choosing actions that lead to winning.
6. **Expert Data**: Optionally, it can learn from your own gameplay before RL training.

---

## Project Structure

```
brawlhalla-rl-agent/
├── BrawlhallaEnv.py         # The game environment for the AI
├── BrawlhallaController.py  # Simulates keyboard/mouse input
├── ScreenRecorder.py        # Captures the game screen
├── config.py                # Settings and parameters
├── main.py                  # RL training entry point
├── bc_pretrain.py           # Behavior cloning (pretraining) script
├── collect_expert_data.py   # Script to record expert demonstrations
├── data_viewer.py           # Visualize collected expert data
├── run.py                   # Run the agent without training
├── checkpoints/             # Saved AI models
├── data/                    # Collected expert data
├── training_logs/           # TensorBoard logs
├── utils/
│   ├── healthcolors.txt     # Color data for health bar tracking
│   └── mouse_tracker.py     # Tool for finding health bar positions
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Customization & Advanced Use

- **Change Actions**: Edit `config.py` to add or remove moves.
- **Tune AI**: Adjust learning rates, batch sizes, and more in `config.py`.
- **Health Bar Positions**: If your game window is in a different spot, update `HEALTH1_POS` and `HEALTH2_POS` in `config.py`.
- **TensorBoard**: Run `tensorboard --logdir training_logs/` to visualize training progress in your browser.
- **Data Loading**: Expert data is now loaded lazily for memory efficiency.

---

## Troubleshooting

- **Out of Memory?** The dataset now uses lazy loading, so you can use large expert datasets without RAM issues.
- **Low FPS?** Close other apps and make sure Brawlhalla is not minimized. Consider a smaller feature extractor network defined in `Policy.py`.
- **Health not detected?** Adjust health bar positions. You can use the `utils/mouse_tracker.py` to find the positions on your screen.
- **Keyboard/mouse not working?** Try running as administrator.
- **Data Viewer Issues?** Make sure your expert data is in the new format (image-only observations).

---

## License

MIT License. Use, modify, and share freely!

---

## Credits & Acknowledgements

- Built with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), [sb3_contrib](https://github.com/Stable-Baselines-Team/sb3_contrib), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and [Brawlhalla](https://www.brawlhalla.com/).
- Inspired by the open-source AI and gaming communities.

---

*Have fun watching your AI learn to brawl!*
