from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn, optim

from src.heuristic import evaluate_board
from src.minimax import minimax_score
from src.model import DQN, DQNFlat
from src.utils import (
    DEFAULT_MODEL_PATH,
    INPUT_DIM_FLAT,
    encode_board,
    encode_board_flat,
    legal_move_indices,
    move_to_index,
)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_actions: list[int] | None = None


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class PrioritizedReplayMemory:
    """Simple proportional Prioritized Experience Replay buffer.

    Priority is updated after learning as abs(TD error) + epsilon. Sampling uses
    priority ** alpha, and beta controls importance-sampling correction.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.memory: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, transition: Transition) -> None:
        max_priority = float(self.priorities.max()) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        active_priorities = self.priorities[: len(self.memory)]
        scaled_priorities = active_priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[index] for index in indices]

        # Importance-sampling weights reduce bias introduced by prioritized
        # sampling. Normalizing by max keeps the loss scale stable.
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for index, td_error in zip(indices, td_errors):
            self.priorities[index] = abs(float(td_error)) + self.epsilon

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        color: chess.Color = chess.BLACK,
        gamma: float = 0.99,
        lr: float = 3e-4,
        batch_size: int = 256,
        memory_size: int = 100_000,
        use_double_dqn: bool = True,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_epsilon: float = 1e-5,
        tau: float = 0.005,
        train_every: int = 4,
        device: str | None = None,
    ):
        self.color = color
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.use_per = use_per
        self.tau = tau
        self.train_every = train_every
        self._step_count = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # New CNN-based model
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.5)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.memory = (
            PrioritizedReplayMemory(memory_size, alpha=per_alpha, beta=per_beta, epsilon=per_epsilon)
            if use_per
            else ReplayMemory(memory_size)
        )

        # Flag to track whether we loaded a legacy flat-model checkpoint
        self._legacy_mode = False

    # ---- Q-value helpers --------------------------------------------------

    def q_values(self, board: chess.Board) -> torch.Tensor:
        if self._legacy_mode:
            state = torch.tensor(encode_board_flat(board), dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state = torch.tensor(encode_board(board), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).squeeze(0)

    def select_action(self, board: chess.Board, epsilon: float = 0.1) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        if random.random() < epsilon:
            return random.choice(legal_moves)

        q_values = self.q_values(board)
        best_move = max(legal_moves, key=lambda move: float(q_values[move_to_index(move)].item()))
        return best_move

    def select_combined_action(
        self,
        board: chess.Board,
        epsilon: float = 0.0,
        minimax_depth: int = 1,
        rl_weight: float = 0.5,
        heuristic_weight: float = 0.3,
        minimax_weight: float = 0.2,
    ) -> tuple[chess.Move, dict[str, float]]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        if random.random() < epsilon:
            move = random.choice(legal_moves)
            return move, {"rl": 0.0, "heuristic": 0.0, "minimax": 0.0, "final": 0.0}

        q_values = self.q_values(board) if rl_weight != 0 else None
        best_move = legal_moves[0]
        best_parts = {"rl": float("-inf"), "heuristic": 0.0, "minimax": 0.0, "final": float("-inf")}

        for move in legal_moves:
            rl_score = float(q_values[move_to_index(move)].item()) if q_values is not None else 0.0
            board.push(move)
            heuristic_score = evaluate_board(board, self.color)
            search_score = minimax_score(board, max(minimax_depth - 1, 0), self.color)
            board.pop()

            final_score = (
                rl_weight * rl_score
                + heuristic_weight * heuristic_score
                + minimax_weight * search_score
            )
            if final_score > best_parts["final"]:
                best_move = move
                best_parts = {
                    "rl": rl_score,
                    "heuristic": heuristic_score,
                    "minimax": search_score,
                    "final": final_score,
                }
        return best_move, best_parts

    # ---- Experience storage -----------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: list[int] | None = None,
    ) -> None:
        self.memory.push(Transition(state, action, reward, next_state, done, next_legal_actions))
        self._step_count += 1

    # ---- Training ---------------------------------------------------------

    def should_train(self) -> bool:
        """Return True when enough steps have passed for a training update."""
        return len(self.memory) >= self.batch_size and self._step_count % self.train_every == 0

    def train_step(self) -> float | None:
        if not self.should_train():
            return None

        if self.use_per:
            transitions, sample_indices, sample_weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(sample_weights, dtype=torch.float32, device=self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            sample_indices = None
            weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)

        states = torch.tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.bool, device=self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            policy_next_q = self.policy_net(next_states)
            target_next_q = self.target_net(next_states)
            next_q = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

            for row, transition in enumerate(transitions):
                if transition.done:
                    continue

                # Legal-action masking keeps impossible chess moves out of the
                # target. Older transitions may not have this field, so they
                # fall back to the full action space for compatibility.
                legal_actions = transition.next_legal_actions
                if legal_actions is None:
                    legal_actions = list(range(target_next_q.shape[1]))
                if not legal_actions:
                    continue

                legal_tensor = torch.tensor(legal_actions, dtype=torch.long, device=self.device)
                if self.use_double_dqn:
                    best_legal_index = policy_next_q[row, legal_tensor].argmax()
                    best_action = legal_tensor[best_legal_index]
                    next_q[row] = target_next_q[row, best_action]
                else:
                    next_q[row] = target_next_q[row, legal_tensor].max()

            target_q = rewards + self.gamma * next_q * (~dones).float()

        td_errors = target_q - current_q
        per_sample_loss = self.loss_fn(current_q, target_q)
        loss = (per_sample_loss * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()

        # ---- soft target update (Polyak averaging) ----
        self.soft_update_target()

        if self.use_per and sample_indices is not None:
            self.memory.update_priorities(sample_indices, td_errors.detach().abs().cpu().numpy())

        return float(loss.item())

    # ---- Target network ---------------------------------------------------

    def soft_update_target(self) -> None:
        """Polyak averaging: θ_target ← τ·θ_policy + (1−τ)·θ_target."""
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def update_target_network(self) -> None:
        """Hard copy (kept for backward compatibility but rarely needed now)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---- Persistence ------------------------------------------------------

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "color": self.color,
                "use_double_dqn": self.use_double_dqn,
                "use_per": self.use_per,
                "model_type": "cnn",  # mark as new architecture
            },
            path,
        )

    def load(self, path: str | Path = DEFAULT_MODEL_PATH) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model_type = checkpoint.get("model_type", "flat")

        if model_type == "cnn":
            # New CNN checkpoint
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
            self._legacy_mode = False
        else:
            # Legacy flat MLP checkpoint — load into DQNFlat so inference works
            self.policy_net = DQNFlat().to(self.device)
            self.target_net = DQNFlat().to(self.device)
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
            self._legacy_mode = True

        self.color = checkpoint.get("color", self.color)
        self.use_double_dqn = checkpoint.get("use_double_dqn", self.use_double_dqn)
        self.use_per = checkpoint.get("use_per", self.use_per)
        self.policy_net.eval()
        self.target_net.eval()
        return True
