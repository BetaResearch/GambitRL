from __future__ import annotations

import chess


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
NEAR_CENTER_SQUARES = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}


def evaluate_board(board: chess.Board, perspective: chess.Color) -> float:
    """Positive means good for perspective, negative means bad."""
    if board.is_checkmate():
        winner = not board.turn
        return 1000.0 if winner == perspective else -1000.0
    if board.is_game_over(claim_draw=True):
        return 0.0

    score = 0.0
    for square, piece in board.piece_map().items():
        sign = 1.0 if piece.color == perspective else -1.0
        score += sign * PIECE_VALUES[piece.piece_type]

        if square in CENTER_SQUARES:
            score += sign * 0.08
        elif square in NEAR_CENTER_SQUARES:
            score += sign * 0.04

        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            home_rank = 0 if piece.color == chess.WHITE else 7
            if chess.square_rank(square) != home_rank:
                score += sign * 0.06

    if board.is_check():
        checked_color = board.turn
        score += -0.25 if checked_color == perspective else 0.25

    return score


def pick_heuristic_move(board: chess.Board, perspective: chess.Color) -> chess.Move:
    best_move = None
    best_score = float("-inf")
    for move in board.legal_moves:
        board.push(move)
        score = evaluate_board(board, perspective)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    if best_move is None:
        raise ValueError("No legal moves available")
    return best_move

