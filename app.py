from __future__ import annotations

from pathlib import Path

import chess
import streamlit as st
import streamlit.components.v1 as components

from src.agent import DQNAgent
from src.environment import ChessEnvironment
from src.utils import DEFAULT_MODEL_PATH


COMPONENT_DIR = Path(__file__).parent / "src" / "streamlit_chessboard_component"
drag_board = components.declare_component("drag_board", path=str(COMPONENT_DIR))


def init_state() -> None:
    if "env" not in st.session_state:
        st.session_state.env = ChessEnvironment(bot_color=chess.BLACK)
    if "agent" not in st.session_state:
        agent = DQNAgent(color=chess.BLACK, device="cpu")
        st.session_state.model_loaded = agent.load(DEFAULT_MODEL_PATH)
        st.session_state.agent = agent
    if "last_bot_move" not in st.session_state:
        st.session_state.last_bot_move = None
    if "last_scores" not in st.session_state:
        st.session_state.last_scores = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "processed_drag_event" not in st.session_state:
        st.session_state.processed_drag_event = None


def make_bot_move() -> None:
    env: ChessEnvironment = st.session_state.env
    if env.board.is_game_over(claim_draw=True) or env.board.turn != chess.BLACK:
        return
    if st.session_state.model_loaded:
        weights = {"rl_weight": 0.5, "heuristic_weight": 0.3, "minimax_weight": 0.2}
    else:
        weights = {"rl_weight": 0.0, "heuristic_weight": 0.6, "minimax_weight": 0.4}
    move, scores = st.session_state.agent.select_combined_action(
        env.board,
        epsilon=0.0,
        minimax_depth=1,
        **weights,
    )
    env.push(move)
    st.session_state.last_bot_move = move.uci()
    st.session_state.last_scores = scores


def reset_game() -> None:
    st.session_state.env = ChessEnvironment(bot_color=chess.BLACK)
    st.session_state.last_bot_move = None
    st.session_state.last_scores = None
    st.session_state.last_error = None
    st.session_state.processed_drag_event = None


def apply_human_move(move_text: str) -> bool:
    env: ChessEnvironment = st.session_state.env
    ok, move, error = validate_human_move(env.board, move_text)
    if not ok:
        st.session_state.last_error = error
        return False

    env.push(move)
    st.session_state.last_error = None
    make_bot_move()
    return True


def validate_human_move(board: chess.Board, move_text: str) -> tuple[bool, chess.Move | None, str | None]:
    clean_move = move_text.strip().lower()
    try:
        move = chess.Move.from_uci(clean_move)
    except ValueError:
        return False, None, f"Invalid move format: {move_text}"

    if board.turn != chess.WHITE:
        return False, None, "It is not White's turn."
    if move not in board.legal_moves:
        return False, None, f"Illegal move: {move_text}"

    return True, move, None


def render_drag_board(board: chess.Board) -> dict | None:
    """Render the drag-and-drop board and return the latest move event, if any."""
    return drag_board(
        fen=board.fen(),
        orientation="white",
        disabled=board.is_game_over(claim_draw=True) or board.turn != chess.WHITE,
        key="drag-board",
        default=None,
    )


def main() -> None:
    st.set_page_config(page_title="ChessBot-RL", page_icon="♟", layout="centered")
    init_state()
    env: ChessEnvironment = st.session_state.env

    st.title("ChessBot-RL")
    st.caption("Human plays White. Drag a piece on the board, or use UCI input as fallback.")

    left, right = st.columns([2, 1])
    with left:
        drag_event = render_drag_board(env.board)
    with right:
        st.metric("Turn", "White" if env.board.turn == chess.WHITE else "Black")
        st.write(env.status())
        st.write("Model:", "loaded checkpoint" if st.session_state.model_loaded else "no checkpoint found")
        if st.session_state.last_bot_move:
            st.write("Bot move:", st.session_state.last_bot_move)
        if st.session_state.last_scores:
            st.write("Bot score blend")
            st.json({key: round(value, 3) for key, value in st.session_state.last_scores.items()})
        st.button("Reset game", on_click=reset_game)

    if drag_event and drag_event.get("event_id") != st.session_state.processed_drag_event:
        st.session_state.processed_drag_event = drag_event.get("event_id")
        if apply_human_move(drag_event.get("move", "")):
            st.rerun()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    with st.form("move_form", clear_on_submit=True):
        human_move = st.text_input("Your move", placeholder="e2e4", disabled=env.board.is_game_over(claim_draw=True))
        submitted = st.form_submit_button("Play move")

    if submitted:
        if apply_human_move(human_move):
            st.rerun()

    if env.board.is_game_over(claim_draw=True):
        st.success(env.status())


if __name__ == "__main__":
    main()
