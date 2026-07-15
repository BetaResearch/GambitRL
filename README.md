# ChessBot-RL

ChessBot-RL is a small final project for a Reinforcement Learning course. It builds a playable chess bot that uses a simple Double DQN agent as the learning component, then blends the learned move score with a readable heuristic evaluator and a lightweight minimax search.

The goal is not to build AlphaZero. The goal is a working, explainable, two-week student project that demonstrates state encoding, legal-action filtering, epsilon-greedy exploration, replay memory, target networks, shaped rewards, reward logging, and a simple human-vs-bot UI.

## Project Structure

```text
.
├── app.py                 # Streamlit UI
├── train.py               # Double DQN training loop
├── requirements.txt
├── README.md
├── models/                # Saved checkpoints
├── results/               # Training plots
└── src/
    ├── environment.py     # python-chess environment wrapper
    ├── agent.py           # Double DQN agent, replay memory, action selection
    ├── model.py           # PyTorch neural network
    ├── minimax.py         # Depth-limited minimax with alpha-beta pruning
    ├── heuristic.py       # Material, check, center, development scoring
    ├── utils.py           # Board encoding, move indexing, paths
    └── streamlit_chessboard_component/
        ├── index.html     # Local drag-and-drop chessboard component
        └── img/           # Local piece images
```

## RL Formulation

**State:** The board is encoded as a numeric vector with 12 piece planes, one plane for each piece type and color, plus small metadata such as side to move, castling rights, and normalized move number.

**Action:** Every chess move is mapped to an integer index using from-square, to-square, and promotion type. The network outputs scores for the full move space, but action selection and training targets only consider legal moves from `python-chess`.

**Reward:** The environment uses deterministic reward shaping from the bot's perspective: wins and delivered checkmates are `+100`, losses and getting checkmated are `-100`, draws are `-5`, captures use material values, giving check is `+0.5`, castling is `+2`, developing knights/bishops is `+0.3`, occupying the center is `+0.5`, and simple no-progress/repetition penalties are `-0.2`.

**Policy:** During training, the bot uses epsilon-greedy exploration. It picks a random legal move with probability epsilon and otherwise picks the legal move with the highest DQN Q-value.

**Learning:** Experiences are stored in replay memory. Training samples random batches and uses Double DQN by default (`use_double_dqn = True`): the online policy network selects the best legal next action, and the target network evaluates that selected action. Terminal next states use reward only.

## Role of Minimax

The trained Double DQN alone will be weak with limited training time. To make gameplay more reasonable, the bot decision in the UI combines:

```text
final_score = 0.5 * rl_score + 0.3 * heuristic_score + 0.2 * minimax_score
```

The minimax depth is intentionally small, usually 1 or 2 plies, so the project stays lightweight and easy to explain.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 12.4 PyTorch wheel (`torch==2.6.0+cu124`) so training can use NVIDIA VRAM. If you do not have an NVIDIA GPU, replace it with the CPU wheel from PyPI before installing.

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Train

Run a quick smoke test:

```bash
python train.py --smoke
```

Run a longer training session:

```bash
python train.py --episodes 200 --opponent random
```

Or train against a simple heuristic opponent:

```bash
python train.py --episodes 200 --opponent heuristic
```

The checkpoint is saved to `models/chessbot_dqn.pt`. Plots are saved to `results/reward_plot.png`, `results/loss_plot.png`, and `results/outcome_plot.png`.

## Run UI

```bash
streamlit run app.py
```

The app runs in headless mode (see `.streamlit/config.toml`) so it will not auto-open a browser — this avoids a native crash on Windows when Streamlit's browser auto-launch interacts with the CUDA-enabled torch wheel. Open the printed `Local URL` (e.g. `http://localhost:8501`) manually in your browser.

The human plays White and enters moves in UCI format:

```text
e2e4
g1f3
e7e8q
```

The board supports drag-and-drop moves with local piece images, and the UCI input remains available as a fallback.

If a trained checkpoint exists, the UI loads it automatically. If no checkpoint exists, the bot still works using heuristic and minimax scores with the RL score disabled.

## Limitations

- The Double DQN architecture is small and does not understand chess like a strong engine.
- The action space is large, so training is slow and sample inefficient.
- The reward function is simple and may not teach long-term strategy well.
- The minimax search is shallow by design.
- There is no opening book, no Stockfish integration, and no AlphaZero-style Monte Carlo tree search.
- This is best viewed as an educational RL project, not a competitive chess engine.
