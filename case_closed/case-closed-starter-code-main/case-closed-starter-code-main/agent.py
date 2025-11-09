import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import random
from collections import deque
from generate_report import generate_report

from case_closed_game import Game, Direction, GameResult, EMPTY

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "DA_AGENT"
AGENT_NAME = "AGENT"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server. 

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
        report = {
            "my_position": my_agent.trail[-1]
        }
        generate_report(state, report)
    
    def pick_move(state):

        DIRECTIONS = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        def in_bounds(x, y, width, height):
            return 0 <= x < width and 0 <= y < height

        def get_head_position(state, agent_id):
            trail = state[f"agent{agent_id}_trail"]
            return tuple(trail[-1])  # last cell is the current head

        def flood_fill_score(board, start):
            """Estimate reachable area for a given starting position."""
            width, height = len(board[0]), len(board)
            visited = set()
            queue = deque([start])
            visited.add(start)
            score = 0

            while queue:
                x, y = queue.popleft()
                score += 1
                for dx, dy in DIRECTIONS.values():
                    nx, ny = x + dx, y + dy
                    if not in_bounds(nx, ny, width, height):
                        continue
                    if board[ny][nx] != 0:
                        continue
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            return score

        def simulate_move(board, start, direction, boost=False):
            """Simulate moving 1 or 2 cells in a direction.
            Returns new position or None if collision."""
            width, height = len(board[0]), len(board)
            dx, dy = DIRECTIONS[direction]
            x, y = start
            steps = 2 if boost else 1

            for _ in range(steps):
                x += dx
                y += dy
                if not in_bounds(x, y, width, height):
                    return None  # out of bounds
                if board[y][x] != 0:
                    return None  # collision
                board[y][x] = 1  # mark new trail
            return (x, y)

        def choose_best_move(state):
            board = [row[:] for row in state["board"]]
            width, height = len(board[0]), len(board)
            player = state["player_number"]
            opponent = 1 if player == 2 else 2

            you = get_head_position(state, player)
            opp = get_head_position(state, opponent)
            boosts = state[f"agent{player}_boosts"]

            best_move = None
            best_score = -float("inf")
            use_boost = False

            for direction in DIRECTIONS.keys():
                # simulate normal move
                for boosted in [False, True] if boosts > 0 else [False]:
                    board_copy = [row[:] for row in board]
                    new_pos = simulate_move(board_copy, you, direction, boost=boosted)
                    if new_pos is None:
                        continue

                    # Compute space control
                    area_score = flood_fill_score(board_copy, new_pos)
                    # Slight penalty for getting close to opponent
                    dist_penalty = abs(new_pos[0] - opp[0]) + abs(new_pos[1] - opp[1])
                    # Boost cost (prefer saving for critical moments)
                    boost_penalty = 4 if boosted else 0

                    total_score = area_score - 0.6 * dist_penalty - boost_penalty

                    if total_score > best_score:
                        best_score = total_score
                        best_move = direction
                        use_boost = boosted

            # fallback if nothing found
            if not best_move:
                valid_moves = []
                for direction, (dx, dy) in DIRECTIONS.items():
                    nx, ny = you[0] + dx, you[1] + dy
                    if in_bounds(nx, ny, width, height) and board[ny][nx] == 0:
                        valid_moves.append(direction)
                best_move = random.choice(valid_moves) if valid_moves else "UP"
                use_boost = False

            return best_move, use_boost
        best_move, use_boost = choose_best_move(state) 

        if use_boost:
            move = f"{best_move}:BOOST"
        else:
            move = best_move
        return move
    move = pick_move(state)
    return jsonify({"move": move}), 200

#


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
