import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "AvoidParticipant"
AGENT_NAME = "AvoidAgent"


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

    The agent's strategy: avoid the opponent at all costs.
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opponent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        boosts_remaining = my_agent.boosts_remaining
        board = GLOBAL_GAME.board

    # helper: torus shortest delta
    def torus_delta(a_coord, b_coord, size):
        d = b_coord - a_coord
        if abs(d) > size // 2:
            if d > 0:
                d = d - size
            else:
                d = d + size
        return d

    my_head = tuple(my_agent.trail[-1])
    try:
        opp_head = tuple(opponent.trail[-1])
    except Exception:
        opp_head = None

    width = board.width
    height = board.height

    # compute torus manhattan distance
    def torus_manhattan(a, b):
        if b is None:
            return 0
        dx = abs(torus_delta(a[0], b[0], width))
        dy = abs(torus_delta(a[1], b[1], height))
        return dx + dy

    # avoid reversing
    cur_dx, cur_dy = my_agent.direction.value
    reverse_dir = (-cur_dx, -cur_dy)

    best = None
    best_score = None
    best_use_boost = False

    for d in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
        if d.value == reverse_dir:
            continue

        # simulate one step
        t1x = my_head[0] + d.value[0]
        t1y = my_head[1] + d.value[1]
        t1x, t1y = board._torus_check((t1x, t1y))
        t1 = (t1x, t1y)

        # base score: larger distance is better (we want to maximize distance)
        score = torus_manhattan(t1, opp_head)

        # penalize landing on occupied cells (strong negative)
        if board.get_cell_state(t1) == 1:
            score -= 1000

        # consider boost: if boost increases distance after the second step, prefer it
        will_boost_help = False
        if boosts_remaining > 0:
            t2x = t1[0] + d.value[0]
            t2y = t1[1] + d.value[1]
            t2x, t2y = board._torus_check((t2x, t2y))
            t2 = (t2x, t2y)
            if torus_manhattan(t2, opp_head) > torus_manhattan(t1, opp_head):
                will_boost_help = True

        # prefer larger score; if boost helps, slightly prefer it
        if best_score is None or score > best_score or (score == best_score and will_boost_help):
            best_score = score
            best = d
            best_use_boost = will_boost_help

    if not best:
        best = Direction.RIGHT

    move = best.name
    if best_use_boost:
        move = f"{move}:BOOST"

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
