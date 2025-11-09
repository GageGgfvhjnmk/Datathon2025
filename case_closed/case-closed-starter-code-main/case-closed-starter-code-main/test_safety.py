"""Quick test to verify RL agent never crosses itself"""
from local_judge import LocalJudge

print("Testing RL Agent Self-Collision Prevention...")
print("Running 20 games to check if RL ever crosses itself\n")

self_collision_count = 0
opponent_collision_count = 0
total_games = 20

for i in range(total_games):
    judge = LocalJudge()
    result = judge.run_game(delay=0.0, visualize=False)
    
    # Check result
    # POSITIVE score = Agent 1 wins (RL loses)
    # NEGATIVE score = Agent 2 (RL) wins
    
    if result == 10:  # Agent 1 won because Agent 2 crossed itself
        self_collision_count += 1
        print(f"Game {i+1}: ❌ RL CROSSED ITSELF!")
    elif result == 25:  # Agent 1 won because Agent 2 hit Agent 1's trail (backwards or opponent trail)
        print(f"Game {i+1}: ✅ RL won (opponent hit RL)")
    elif result == -10:  # Agent 2 (RL) won because Agent 1 crossed RL's trail
        opponent_collision_count += 1
        print(f"Game {i+1}: ~ RL hit opponent (RL won)")
    elif result == -25:  # Agent 2 (RL) won because Agent 1 crossed its own trail
        print(f"Game {i+1}: ✅ RL won (opponent crossed itself)")
    else:
        print(f"Game {i+1}: Draw or other result")

print("\n" + "="*50)
print(f"RESULTS:")
print(f"  Self-collisions: {self_collision_count}/{total_games}")
print(f"  Opponent-collisions: {opponent_collision_count}/{total_games}")

if self_collision_count == 0:
    print("  ✅ SUCCESS! RL agent NEVER crossed itself!")
else:
    print(f"  ❌ FAILURE! RL crossed itself {self_collision_count} times")
print("="*50)
