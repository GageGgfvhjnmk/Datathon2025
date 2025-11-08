# gage_logic.py
from typing import Tuple, List, Optional
import agent
from case_closed_game import Direction, EMPTY

# Order also serves as the tie-break priority
DIRS: List[Tuple[str, Tuple[int, int], Direction]] = [
    ("UP",    ( 0, -1), Direction.UP),
    ("RIGHT", ( 1,  0), Direction.RIGHT),
    ("DOWN",  ( 0,  1), Direction.DOWN),
    ("LEFT",  (-1,  0), Direction.LEFT),
]

def _wrap(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    return x % w, y % h

def _is_wall(board, x: int, y: int) -> bool:
    # Board uses torus semantics internally; get_cell_state normalizes
    return board.get_cell_state((x, y)) != EMPTY

def _head_xy(my_trail) -> Tuple[int, int]:
    # trail is a deque/list of (x, y); head is last entry
    hx, hy = my_trail[-1]
    return hx, hy

def _window_avg_ahead(board, hx: int, hy: int, vec: Tuple[int, int], size: int = 7) -> float:
    """
    Average 'walliness' in a size×size window pushed forward along vec.
    We center the window ~half the window (3) ahead to bias lookahead.
    """
    w, h = board.width, board.height
    dx, dy = vec
    push = size // 2  # 3 for 7x7

    cx, cy = _wrap(hx + dx * push, hy + dy * push, w, h)
    half = size // 2

    total = 0
    for oy in range(-half, half + 1):
        for ox in range(-half, half + 1):
            x, y = _wrap(cx + ox, cy + oy, w, h)
            total += 1 if _is_wall(board, x, y) else 0

    return total / float(size * size)

def gage_logic() -> str:
    """
    Returns one of: 'UP', 'RIGHT', 'DOWN', 'LEFT'
    Uses agent.GLOBAL_GAME + agent.LAST_POSTED_STATE to decide.
    """
    game = agent.GLOBAL_GAME
    board = game.board

    # Prefer player_number from the last posted state; default to 1 if absent
    pnum = (agent.LAST_POSTED_STATE or {}).get("player_number", 1)

    my_agent = game.agent1 if pnum == 1 else game.agent2
    hx, hy = _head_xy(my_agent.trail)

    w, h = board.width, board.height

    # 1) Filter: immediate neighbor must be free
    candidates: List[Tuple[str, Tuple[int, int], Direction]] = []
    for name, (dx, dy), enum_dir in DIRS:
        nx, ny = _wrap(hx + dx, hy + dy, w, h)
        if not _is_wall(board, nx, ny):
            candidates.append((name, (dx, dy), enum_dir))

    if not candidates:
        # Boxed in: pick something deterministic
        return "UP"

    # 2) For each candidate, compute 7×7 'ahead' average and pick the minimum
    best_name: Optional[str] = None
    best_score: Optional[float] = None
    for name, vec, _enum_dir in candidates:
        avg = _window_avg_ahead(board, hx, hy, vec, size=7)
        if best_score is None or avg < best_score:
            best_name = name
            best_score = avg

    return best_name or "UP"
