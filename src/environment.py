from __future__ import annotations

import chess

from src.heuristic import evaluate_board


REWARD_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}
EXTENDED_CENTER_SQUARES = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}
KNIGHT_BISHOP_START_SQUARES = {
    chess.B1,
    chess.G1,
    chess.C1,
    chess.F1,
    chess.B8,
    chess.G8,
    chess.C8,
    chess.F8,
}

# Piece-square bonus tables (from White's perspective; flipped for Black).
# Small positional nudges that teach the network basic positional concepts.
PAWN_TABLE = [
     0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
     0.05,  0.05,  0.05, -0.05, -0.05,  0.05,  0.05,  0.05,
     0.01,  0.01,  0.02,  0.06,  0.06,  0.02,  0.01,  0.01,
     0.00,  0.00,  0.00,  0.05,  0.05,  0.00,  0.00,  0.00,
     0.01,  0.01,  0.02,  0.06,  0.06,  0.02,  0.01,  0.01,
     0.02,  0.02,  0.04,  0.04,  0.04,  0.04,  0.02,  0.02,
     0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,
     0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
]

KNIGHT_TABLE = [
    -0.10, -0.04, -0.03, -0.03, -0.03, -0.03, -0.04, -0.10,
    -0.04, -0.02,  0.00,  0.01,  0.01,  0.00, -0.02, -0.04,
    -0.03,  0.01,  0.02,  0.03,  0.03,  0.02,  0.01, -0.03,
    -0.03,  0.00,  0.03,  0.04,  0.04,  0.03,  0.00, -0.03,
    -0.03,  0.01,  0.03,  0.04,  0.04,  0.03,  0.01, -0.03,
    -0.03,  0.00,  0.02,  0.03,  0.03,  0.02,  0.00, -0.03,
    -0.04, -0.02,  0.00,  0.00,  0.00,  0.00, -0.02, -0.04,
    -0.10, -0.04, -0.03, -0.03, -0.03, -0.03, -0.04, -0.10,
]


def _piece_square_bonus(piece: chess.Piece, square: int) -> float:
    """Return a small positional bonus for *piece* sitting on *square*."""
    if piece.piece_type == chess.PAWN:
        table = PAWN_TABLE
    elif piece.piece_type == chess.KNIGHT:
        table = KNIGHT_TABLE
    else:
        return 0.0
    idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
    return table[idx]


def shaped_move_reward(
    before_board: chess.Board,
    move: chess.Move,
    after_board: chess.Board,
    bot_color: chess.Color,
) -> float:
    """Deterministic reward shaping from the bot's perspective.

    Positive values help the bot; negative values hurt the bot. The same helper
    can score bot moves and opponent moves, so opponent captures or checks become
    penalties when used by training.
    """
    mover_color = before_board.turn
    mover_sign = 1.0 if mover_color == bot_color else -1.0
    reward = 0.0

    # ---- terminal outcomes (dominate all shaping) ----
    if after_board.is_checkmate():
        winner = not after_board.turn
        return 100.0 if winner == bot_color else -100.0
    if after_board.is_game_over(claim_draw=True):
        return -5.0

    # ---- material: captures ----
    captured_piece = before_board.piece_at(move.to_square)
    if before_board.is_en_passant(move):
        captured_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square),
        )
        captured_piece = before_board.piece_at(captured_square)
    if captured_piece is not None:
        reward += mover_sign * REWARD_PIECE_VALUES[captured_piece.piece_type]

    # ---- tactics: giving check ----
    if after_board.is_check():
        reward += mover_sign * 0.5

    # ---- king safety: castling ----
    if before_board.is_castling(move):
        reward += mover_sign * 2.0

    # ---- king safety: penalize an exposed king ----
    bot_king_sq = after_board.king(bot_color)
    if bot_king_sq is not None:
        n_attackers = len(after_board.attackers(not bot_color, bot_king_sq))
        if n_attackers > 0:
            reward -= 0.3 * n_attackers  # always negative for the bot

    # ---- positional: piece-square tables (pawn / knight) ----
    moved_piece = before_board.piece_at(move.from_square)
    if moved_piece is not None:
        psq_after = _piece_square_bonus(moved_piece, move.to_square)
        psq_before = _piece_square_bonus(moved_piece, move.from_square)
        reward += mover_sign * (psq_after - psq_before)

        developed_minor_piece = False
        # Developing a knight or bishop from its initial square.
        if (
            moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and move.from_square in KNIGHT_BISHOP_START_SQUARES
        ):
            reward += mover_sign * 0.3
            developed_minor_piece = True

        # Occupying the four central squares.
        moved_to_center = move.to_square in CENTER_SQUARES
        moved_to_ext_center = move.to_square in EXTENDED_CENTER_SQUARES
        if moved_to_center:
            reward += mover_sign * 0.5
        elif moved_to_ext_center:
            reward += mover_sign * 0.15

        no_capture = captured_piece is None
        no_progress_piece = moved_piece.piece_type != chess.PAWN
        no_tactical_gain = (
            not after_board.is_check()
            and not moved_to_center
            and not moved_to_ext_center
            and not developed_minor_piece
        )
        if mover_color == bot_color:
            # Quiet, reversible-looking moves that do nothing useful.
            if (
                no_capture
                and no_progress_piece
                and no_tactical_gain
                and not before_board.is_castling(move)
            ):
                reward -= 0.3

            # Repeating a position.
            if after_board.is_repetition(2):
                reward -= 0.3

    # ---- mobility: reward having more legal moves ----
    if not after_board.is_game_over(claim_draw=True):
        mobility = len(list(after_board.legal_moves))
        # The side to move in after_board is the *opponent* of the mover.
        # High opponent mobility is bad for the mover, low is good.
        # We normalize: average mobility ≈ 30 moves.
        reward += mover_sign * (30 - mobility) * 0.01

    return reward


class ChessEnvironment:
    """Thin python-chess wrapper used by training and the UI."""

    def __init__(self, bot_color: chess.Color = chess.BLACK):
        self.bot_color = bot_color
        self.board = chess.Board()
        self.previous_eval = evaluate_board(self.board, self.bot_color)

    def reset(self) -> chess.Board:
        self.board.reset()
        self.previous_eval = evaluate_board(self.board, self.bot_color)
        return self.board.copy()

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> tuple[chess.Board, float, bool]:
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")

        before_board = self.board.copy()
        self.board.push(move)
        reward = self._reward(before_board, move)
        done = self.board.is_game_over(claim_draw=True)
        self.previous_eval = evaluate_board(self.board, self.bot_color)
        return self.board.copy(), reward, done

    def _reward(self, before_board: chess.Board, move: chess.Move) -> float:
        return shaped_move_reward(before_board, move, self.board, self.bot_color)

    def status(self) -> str:
        if self.board.is_checkmate():
            return f"Checkmate. {'White' if not self.board.turn else 'Black'} wins."
        if self.board.is_stalemate():
            return "Draw by stalemate."
        if self.board.is_insufficient_material():
            return "Draw by insufficient material."
        if self.board.can_claim_fifty_moves():
            return "Draw can be claimed by fifty-move rule."
        if self.board.can_claim_threefold_repetition():
            return "Draw can be claimed by threefold repetition."
        if self.board.is_check():
            return f"{'White' if self.board.turn else 'Black'} to move, in check."
        return f"{'White' if self.board.turn else 'Black'} to move."
