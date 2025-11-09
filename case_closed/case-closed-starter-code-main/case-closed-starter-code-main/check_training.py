import torch
import numpy as np

def check_training_progress():
    try:
        checkpoint = torch.load('simple_dqn_model.pth', map_location='cpu')
        epsilon = checkpoint['epsilon']
        print(f"Current epsilon: {epsilon:.3f}")
        print(f"Exploration rate: {epsilon*100:.1f}%")
        
        if epsilon <= 0.05:
            print("✅ Agent is mostly exploiting (good!)")
        elif epsilon <= 0.2:
            print("🔄 Agent is balancing exploration/exploitation")
        else:
            print("🔍 Agent is still exploring heavily")
            
    except FileNotFoundError:
        print("No model found yet")

if __name__ == "__main__":
    check_training_progress()