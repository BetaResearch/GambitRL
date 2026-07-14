from __future__ import annotations

import chess

from src.heuristic import evaluate_board


def minimax_score(
    board: chess.Board,
    depth: int,
    perspective: chess.Color,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
) -> float:
    if depth == 0 or board.is_game_over(claim_draw=True):
        return evaluate_board(board, perspective)

    maximizing = board.turn == perspective
    if maximizing:
        value = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            value = max(value, minimax_score(board, depth - 1, perspective, alpha, beta))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = float("inf")
    for move in board.legal_moves:
        board.push(move)
        value = min(value, minimax_score(board, depth - 1, perspective, alpha, beta))
        board.pop()
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value

