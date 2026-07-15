from __future__ import annotations

import argparse
import math
import random
from collections import deque
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import torch

from src.agent import DQNAgent
from src.environment import shaped_move_reward
from src.heuristic import PIECE_VALUES, evaluate_board, pick_heuristic_move
from src.utils import (
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    choose_random_move,
    encode_board,
    ensure_dirs,
    legal_move_indices,
    mean_or_zero,
    move_to_index,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChessBot-RL with CNN Dueling-DQN.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument(
        "--opponent",
        choices=["random", "weak-heuristic", "heuristic", "self-play"],
        default="random",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train against random → weak-heuristic → heuristic → self-play opponents.",
    )
    parser.add_argument("--random-opponent-ratio", type=float, default=0.20)
    parser.add_argument("--weak-heuristic-ratio", type=float, default=0.30)
    parser.add_argument("--heuristic-ratio", type=float, default=0.25)
    # remaining ratio (0.25) goes to self-play
    parser.add_argument("--use-per", action="store_true", help="Use Prioritized Experience Replay.")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--target-update", type=int, default=10, help="Hard target-net sync interval (episodes). Soft update happens every train step.")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Run one tiny episode to verify the training path.")
    # Epsilon schedule
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=0.992, help="Multiplicative decay per episode.")
    # Self-play snapshot interval
    parser.add_argument("--selfplay-snapshot-every", type=int, default=100, help="Save a self-play opponent snapshot every N episodes.")
    return parser.parse_args()


def validate_curriculum_args(args: argparse.Namespace) -> None:
    if args.random_opponent_ratio < 0 or args.weak_heuristic_ratio < 0 or args.heuristic_ratio < 0:
        raise ValueError("Curriculum ratios must be non-negative.")
    total = args.random_opponent_ratio + args.weak_heuristic_ratio + args.heuristic_ratio
    if total > 1.0:
        raise ValueError("Sum of curriculum ratios must be <= 1.0")


def curriculum_opponent_for_episode(episode: int, total_episodes: int, args: argparse.Namespace) -> str:
    """Return the opponent type for this curriculum episode.

    Example with defaults (20/30/25/25):
      first 20%  -> random
      next  30%  -> weak-heuristic
      next  25%  -> heuristic
      final 25%  -> self-play
    """
    progress = episode / max(total_episodes, 1)
    r_cutoff = args.random_opponent_ratio
    w_cutoff = r_cutoff + args.weak_heuristic_ratio
    h_cutoff = w_cutoff + args.heuristic_ratio
    if progress <= r_cutoff:
        return "random"
    if progress <= w_cutoff:
        return "weak-heuristic"
    if progress <= h_cutoff:
        return "heuristic"
    return "self-play"


def captured_piece_value(board: chess.Board, move: chess.Move) -> float:
    captured_piece = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured_square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        captured_piece = board.piece_at(captured_square)
    return PIECE_VALUES[captured_piece.piece_type] if captured_piece is not None else 0.0


def pick_weak_heuristic_move(board: chess.Board, perspective: chess.Color) -> chess.Move:
    """A deliberately imperfect opponent for curriculum learning."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")
    if random.random() < 0.20:
        return random.choice(legal_moves)
    captures = [move for move in legal_moves if board.is_capture(move)]
    if captures and random.random() < 0.60:
        return max(captures, key=lambda move: captured_piece_value(board, move))

    def shallow_score(move: chess.Move) -> float:
        board.push(move)
        score = evaluate_board(board, perspective)
        board.pop()
        return score

    scored_moves = sorted(legal_moves, key=shallow_score, reverse=True)
    top_count = max(1, min(3, len(scored_moves)))
    return random.choice(scored_moves[:top_count])


def choose_opponent_move(
    board: chess.Board,
    opponent: str,
    opponent_color: chess.Color,
    selfplay_agent: DQNAgent | None = None,
) -> chess.Move:
    """Pick a move for the opponent side."""
    if opponent == "self-play" and selfplay_agent is not None:
        # Self-play opponent uses the snapshot agent with a small epsilon for
        # diversity.  This is the key training signal: the bot improves by
        # playing against older versions of itself.
        return selfplay_agent.select_action(board, epsilon=0.1)
    if opponent == "weak-heuristic":
        return pick_weak_heuristic_move(board, opponent_color)
    if opponent == "heuristic":
        return pick_heuristic_move(board, opponent_color)
    return choose_random_move(board)


def terminal_reward(board: chess.Board, bot_color: chess.Color) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == bot_color else -1.0


def rolling_average(values: list[float], window: int) -> list[float]:
    return [
        mean_or_zero(values[max(0, i - window + 1) : i + 1])
        for i in range(len(values))
    ]


def rolling_win_rate(outcomes: list[str], window: int) -> list[float]:
    rates = []
    for i in range(len(outcomes)):
        recent = outcomes[max(0, i - window + 1) : i + 1]
        rates.append(recent.count("win") / max(len(recent), 1))
    return rates


def plot_metrics(
    rewards: list[float],
    losses: list[float],
    outcomes: list[str],
    epsilons: list[float],
    win_rates: list[float],
) -> None:
    ensure_dirs()
    window = 10
    rolling_rewards = rolling_average(rewards, window)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="episode reward", alpha=0.45)
    plt.plot(rolling_rewards, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reward_plot.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Huber loss")
    plt.title("DQN Loss")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_plot.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "epsilon_plot.png")
    plt.close()

    counts = [outcomes.count("win"), outcomes.count("draw"), outcomes.count("loss")]
    plt.figure(figsize=(5, 4))
    plt.bar(["win", "draw", "loss"], counts, color=["#2c7a7b", "#718096", "#c53030"])
    plt.title("Training Outcomes")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "outcome_plot.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(win_rates, label=f"{window}-episode win rate")
    plt.ylim(0, 1)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Win Rate Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "win_rate_plot.png")
    plt.close()


def train() -> None:
    args = parse_args()
    validate_curriculum_args(args)
    if args.smoke:
        args.episodes = 1
        args.max_plies = 8

    ensure_dirs()
    set_seed(args.seed)

    bot_color = chess.BLACK
    opponent_color = chess.WHITE
    agent = DQNAgent(
        color=bot_color,
        use_per=args.use_per,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
    )
    board = chess.Board()

    # ---- self-play opponent ----
    selfplay_agent: DQNAgent | None = None
    selfplay_snapshot_path = MODELS_DIR / "selfplay_snapshot.pt"

    rewards: list[float] = []
    losses: list[float] = []
    epsilons: list[float] = []
    episode_losses: list[float] = []
    episode_avg_q_values: list[float] = []
    outcomes: list[str] = []
    recent_outcomes: deque[str] = deque(maxlen=20)
    interval_outcomes: deque[str] = deque(maxlen=10)

    # Exponential epsilon schedule
    epsilon = args.eps_start

    for episode in range(1, args.episodes + 1):
        opponent_type = (
            curriculum_opponent_for_episode(episode, args.episodes, args)
            if args.curriculum
            else args.opponent
        )

        # Lazily create / refresh self-play opponent when needed
        if opponent_type == "self-play":
            if selfplay_agent is None:
                selfplay_agent = DQNAgent(color=opponent_color)
                # Start with current weights
                selfplay_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
                selfplay_agent.policy_net.eval()
            if episode % args.selfplay_snapshot_every == 0:
                # Refresh snapshot so the opponent slowly improves too
                selfplay_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
                selfplay_agent.policy_net.eval()
                agent.save(selfplay_snapshot_path)

        board.reset()
        episode_reward = 0.0
        current_episode_losses: list[float] = []
        current_episode_q_values: list[float] = []
        pending_opponent_reward = 0.0

        # Exponential decay
        epsilon = max(args.eps_end, epsilon * args.eps_decay)

        for _ply in range(args.max_plies):
            if board.is_game_over(claim_draw=True):
                break

            if board.turn == opponent_color:
                before_board = board.copy()
                opponent_move = choose_opponent_move(board, opponent_type, opponent_color, selfplay_agent)
                board.push(opponent_move)
                pending_opponent_reward += shaped_move_reward(before_board, opponent_move, board, bot_color)
                if board.is_game_over(claim_draw=True):
                    episode_reward += pending_opponent_reward
                    pending_opponent_reward = 0.0
                continue

            state = encode_board(board)
            move = agent.select_action(board, epsilon=epsilon)
            with torch.no_grad():
                q_value = float(agent.q_values(board)[move_to_index(move)].item())
            current_episode_q_values.append(q_value)
            action = move_to_index(move)
            before_board = board.copy()
            board.push(move)

            reward = shaped_move_reward(before_board, move, board, bot_color) + pending_opponent_reward
            pending_opponent_reward = 0.0
            done = board.is_game_over(claim_draw=True)
            next_state = encode_board(board)
            next_legal_actions = [] if done else legal_move_indices(board)

            # If this is the final ply and the game is still undecided, treat it
            # as a terminal timeout with a small penalty so the agent gets a
            # learning signal instead of an open-ended bootstrap. Without this,
            # long drawn games never terminate in the replay buffer and the bot
            # has no incentive to make progress.
            timeout = _ply == args.max_plies - 1 and not done
            if timeout:
                done = True
                reward -= 2.0
                next_legal_actions = []

            agent.remember(state, action, reward, next_state, done, next_legal_actions)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                current_episode_losses.append(loss)
            episode_reward += reward

            if done:
                break

        if not board.is_game_over(claim_draw=True):
            outcome = "draw"
        else:
            result = terminal_reward(board, bot_color)
            outcome = "win" if result > 0 else "loss" if result < 0 else "draw"

        rewards.append(episode_reward)
        outcomes.append(outcome)
        recent_outcomes.append(outcome)
        interval_outcomes.append(outcome)
        epsilons.append(epsilon)
        episode_loss = mean_or_zero(current_episode_losses)
        episode_avg_q = mean_or_zero(current_episode_q_values)
        episode_losses.append(episode_loss)
        episode_avg_q_values.append(episode_avg_q)
        moving_reward = mean_or_zero(rewards[max(0, len(rewards) - 10) :])
        win_rates = rolling_win_rate(outcomes, 10)
        current_win_rate = win_rates[-1]

        if episode % args.target_update == 0:
            agent.update_target_network()
        if episode % args.save_every == 0 or episode == args.episodes:
            agent.save(DEFAULT_MODEL_PATH)

        opponent_log = f" | opponent={opponent_type}" if args.curriculum else ""
        print(
            f"Episode {episode:04d} | reward={episode_reward:+.3f} | "
            f"moving_reward={moving_reward:+.3f} | loss={episode_loss:.4f} | "
            f"epsilon={epsilon:.3f} | avg_q={episode_avg_q:+.3f}{opponent_log} | "
            f"outcome={outcome} | interval W/D/L={interval_outcomes.count('win')}/"
            f"{interval_outcomes.count('draw')}/{interval_outcomes.count('loss')} | "
            f"recent_win_rate={current_win_rate:.2f}"
        )

    agent.save(DEFAULT_MODEL_PATH)
    plot_metrics(rewards, losses, outcomes, epsilons, rolling_win_rate(outcomes, 10))
    print(f"Saved model to {DEFAULT_MODEL_PATH}")
    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == "__main__":
    train()
