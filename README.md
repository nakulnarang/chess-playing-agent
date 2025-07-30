# Chess-Playing Agent: Deep Reinforcement Learning & Imitation Learning

A research-driven project implementing deep reinforcement learning and imitation learning agents to play chess, developed for a Reinforcement Learning class. The agents are evaluated against top engines and random agents, with all code and training environments based on open-source Python libraries.

## 🚀 Project Highlights

* **Deep Q-Network Agent**: Implements a DQN agent with Lc0-style board encoding, using PyTorch and python-chess.
* **Imitation Learning**: Trains a supervised model to mimic Magnus Carlsen’s blitz games, implementing custom PyTorch architectures.
* **Environment**: Chess gameplay environment compatible with [gym-chess](https://github.com/not-patar/gym-chess) and `python-chess`.
* **Evaluation**: Includes automated benchmarking against Stockfish (skill level 1) and random-move agents, with comprehensive win/draw/loss tracking.
* **Reproducibility**: All dependencies tracked in `requirements.txt` and designed for easy Colab GPU training.

## 📊 Key Results

* Strong move prediction and strategy using DQN, verified with automated evaluations.
* Grandmaster-level move imitation via supervised learning with PyTorch.
* Benchmarking against Stockfish and random agents for objective performance metrics.

## 🛠️ Technologies & Libraries

* [PyTorch](https://pytorch.org/)
* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [python-chess](https://python-chess.readthedocs.io/)
* [gym-chess](https://github.com/not-patar/gym-chess)
* [numpy](https://numpy.org/), [tqdm](https://tqdm.github.io/)
* Google Colab (recommended for GPU training support)
* All dependencies: see [requirements.txt](https://github.com/nakulnarang/chess-playing-agent/blob/main/requirements.txt)

## ⚡ Getting Started

To run or further train the agents:

1. **Clone the repository & install dependencies:**
    ```
    pip install -r requirements.txt
    ```
2. **Run on Google Colab for best performance:**
    - Upload both notebooks to Colab and ensure GPU is enabled.
    - Execute all cells to train; resulting model files (`.pth`) will be saved in the Colab session.
3. **Evaluation:**
    - Download your trained model files.
    - Place DQN and imitation models into the respective directories:
      ```
      models/dqn/
      models/imitation_learning/
      ```
    - Run `evaluate_chess_models.py` to benchmark against Stockfish and random agents.

## 🧩 Files

* `src/DQN_Chess_Agent_with_Lc0_style_Encoding.ipynb` — DQN agent & training pipeline.
* `src/chess_imitation_rl.ipynb` — Imitation learning pipeline.
* `src/evaluate_chess_models.py` — Evaluation and benchmarking script.

## ✨ Contributors

* [@nakulnarang](https://github.com/nakulnarang)
* [@farzanmrz](https://github.com/farzanmrz)

## 📄 License

MIT License - see [LICENSE](https://github.com/nakulnarang/chess-playing-agent/blob/main
