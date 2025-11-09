"""Run multiple games to test torus-aware algorithm performance"""
from local_judge import LocalJudge

if __name__ == "__main__":
    num_games = 20
    
    wins_agent1 = 0
    wins_agent2 = 0
    total_trail_agent1 = 0
    total_trail_agent2 = 0
    scores = []
    
    print(f"Testing Torus-Aware Algorithm: Running {num_games} games...\n")
    
    for i in range(num_games):
        judge = LocalJudge()
        score = judge.run_game(delay=0.0, visualize=False)
        scores.append(score)
        
        # Track results
        trail_len_agent1 = judge.game.agent1.length
        trail_len_agent2 = judge.game.agent2.length
        total_trail_agent1 += trail_len_agent1
        total_trail_agent2 += trail_len_agent2
        
        if score > 0:
            wins_agent1 += 1
            print(f"Game {i+1}: Agent 1 wins (score: {score})")
        elif score < 0:
            wins_agent2 += 1
            print(f"Game {i+1}: Agent 2 wins (score: {score})")
        else:
            print(f"Game {i+1}: Draw (score: 0)")
    
    print(f"\n{'='*50}")
    print(f"RESULTS AFTER {num_games} GAMES")
    print(f"{'='*50}")
    print(f"Agent 1 wins: {wins_agent1} ({wins_agent1/num_games*100:.1f}%)")
    print(f"Agent 2 wins: {wins_agent2} ({wins_agent2/num_games*100:.1f}%)")
    print(f"Average trail length - Agent 1: {total_trail_agent1/num_games:.1f}")
    print(f"Average trail length - Agent 2: {total_trail_agent2/num_games:.1f}")
    print(f"Average score: {sum(scores)/len(scores):.1f}")
    print(f"Score distribution: {min(scores)} to {max(scores)}")
    print(f"{'='*50}")
