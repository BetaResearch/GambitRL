from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import chess
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_PATH = MODELS_DIR / "chessbot_dqn.pt"


MOVE_PLANES = 64 * 64 * 5
PROMOTION_TO_INDEX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
INDEX_TO_PROMOTION = {
    0: None,
    1: chess.KNIGHT,
    2: chess.BISHOP,
    3: chess.ROOK,
    4: chess.QUEEN,
}


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a python-chess board as a (18, 8, 8) tensor for CNN input.

    Planes 0-11:  piece placement (6 piece types × 2 colors)
    Plane 12:     squares attacked by the side to move
    Plane 13:     squares attacked by the opponent
    Plane 14:     side to move (all 1s if White, all 0s if Black)
    Plane 15:     castling rights — 4 quadrants encode KQkq
    Plane 16:     en-passant square (single 1 on the target square)
    Plane 17:     normalized move number (broadcast)
    """
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # --- piece planes (0-11) ---
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)  # 0-7
        file = chess.square_file(square)  # 0-7
        color_offset = 0 if piece.color == chess.WHITE else 6
        planes[color_offset + piece_to_plane[piece.piece_type], rank, file] = 1.0

    # --- attack maps (12-13) ---
    for sq in range(64):
        r, f = chess.square_rank(sq), chess.square_file(sq)
        if board.is_attacked_by(board.turn, sq):
            planes[12, r, f] = 1.0
        if board.is_attacked_by(not board.turn, sq):
            planes[13, r, f] = 1.0

    # --- side to move (14) ---
    if board.turn == chess.WHITE:
        planes[14, :, :] = 1.0

    # --- castling rights (15) ---
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[15, :4, 4:] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[15, :4, :4] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, 4:, 4:] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, 4:, :4] = 1.0

    # --- en-passant (16) ---
    if board.ep_square is not None:
        r = chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        planes[16, r, f] = 1.0

    # --- normalized move number (17) ---
    planes[17, :, :] = min(board.fullmove_number / 100.0, 1.0)

    return planes


# Keep the old flat encoder available for backward compatibility / checkpoints.
INPUT_DIM_FLAT = 12 * 64 + 6


def encode_board_flat(board: chess.Board) -> np.ndarray:
    """Legacy flat encoder (774-dim) for loading old checkpoints."""
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    planes = np.zeros((12, 64), dtype=np.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        planes[color_offset + piece_to_plane[piece.piece_type], square] = 1.0

    extras = np.array(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            min(board.fullmove_number / 100.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([planes.reshape(-1), extras])


def move_to_index(move: chess.Move) -> int:
    promotion_index = PROMOTION_TO_INDEX.get(move.promotion, 0)
    return ((move.from_square * 64) + move.to_square) * 5 + promotion_index


def index_to_move(index: int) -> chess.Move:
    base, promotion_index = divmod(index, 5)
    from_square, to_square = divmod(base, 64)
    return chess.Move(from_square, to_square, promotion=INDEX_TO_PROMOTION[promotion_index])


def legal_move_indices(board: chess.Board) -> list[int]:
    return [move_to_index(move) for move in board.legal_moves]


def board_result_value(board: chess.Board, bot_color: chess.Color) -> float:
    if not board.is_game_over():
        return 0.0
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == bot_color else -1.0


def choose_random_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0

