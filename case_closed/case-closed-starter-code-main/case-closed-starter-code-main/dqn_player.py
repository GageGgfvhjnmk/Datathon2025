import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import torch
import numpy as np

from case_closed_game import Game, Direction, EMPTY
from dqn_agent import SimpleDQNAgent

app = Flask(__name__)
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "DQNMaster"
AGENT_NAME = "TrainedDQN"

# Load your trained agent
print("Loading trained DQN agent...")
dqn_agent = SimpleDQNAgent(state_size=15, action_size=8)
dqn_agent.epsilon = 0.01  # Minimal exploration for actual play
print("DQN Agent ready!")

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

def _update_local_game_from_post(data: dict):
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
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opponent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1

    try:
        # Get state and valid actions
        state = dqn_agent.get_state_representation(GLOBAL_GAME, my_agent, opponent)
        valid_actions = dqn_agent.get_valid_actions(my_agent, my_agent.boosts_remaining)
        
        # Choose action using trained model
        action = dqn_agent.act(state, valid_actions)
        
        # Convert to move string
        direction_idx = action // 2
        use_boost = action % 2
        direction_names = ["RIGHT", "LEFT", "DOWN", "UP"]
        move = direction_names[direction_idx]
        
        # Add boost if chosen and available
        if use_boost and my_agent.boosts_remaining > 0:
            move += ":BOOST"

        print(f"DQN playing {move} (boosts: {my_agent.boosts_remaining})")
        return jsonify({"move": move}), 200
        
    except Exception as e:
        print(f"Error in DQN: {e}")
        # Fallback
        return jsonify({"move": "RIGHT"}), 200

@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5010"))
    print(f"DQN Agent Server starting on port {port}")
    print(f"Participant: {PARTICIPANT}, Agent: {AGENT_NAME}")
    app.run(host="0.0.0.0", port=port, debug=False)