import torch
import numpy as np
import os

def inspect_model():
    """Inspect the contents of your trained model"""
    model_path = r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\simple_dqn_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"🔍 Inspecting: {model_path}")
    print("=" * 50)
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("📁 CHECKPOINT KEYS:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        print(f"\n📊 MODEL INFO:")
        if 'epsilon' in checkpoint:
            print(f"  Epsilon: {checkpoint['epsilon']:.4f}")
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"  Model layers: {len(model_state)}")
            
            print(f"\n🧠 NEURAL NETWORK ARCHITECTURE:")
            total_params = 0
            for name, param in model_state.items():
                print(f"  {name}: {tuple(param.shape)}")
                total_params += param.numel()
            
            print(f"\n📈 NETWORK STATS:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            print(f"\n⚙️ OPTIMIZER INFO:")
            print(f"  Optimizer type: {optimizer_state.get('name', 'Unknown')}")
            if 'param_groups' in optimizer_state:
                print(f"  Learning rate: {optimizer_state['param_groups'][0].get('lr', 'Unknown')}")
        
        # Sample some weight values to see if they're learning
        if 'model_state_dict' in checkpoint:
            print(f"\n🎯 WEIGHT ANALYSIS (first layer sample):")
            first_layer_weights = None
            for name, param in checkpoint['model_state_dict'].items():
                if 'weight' in name and param.dim() == 2:
                    first_layer_weights = param
                    break
            
            if first_layer_weights is not None:
                weights_sample = first_layer_weights.flatten()[:10]  # First 10 weights
                print(f"  Weight range: [{weights_sample.min():.4f}, {weights_sample.max():.4f}]")
                print(f"  Weight mean: {weights_sample.mean():.4f}")
                print(f"  Weight std: {weights_sample.std():.4f}")
                
                # Check if weights are learning (not stuck)
                if abs(weights_sample.mean()) > 0.01 and weights_sample.std() > 0.01:
                    print("  ✅ Weights show learning (good variance)")
                else:
                    print("  ⚠️ Weights might be stuck (low variance)")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def compare_models():
    """Compare multiple model files if they exist"""
    model_files = [
        r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\simple_dqn_model.pth",
        "dqn_model.pth", 
        r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\cnn_dqn_model.pth",
        r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\numpy_dqn_model.pkl"
    ]
    
    print("\n" + "=" * 50)
    print("🔄 MODEL COMPARISON")
    print("=" * 50)
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                if model_file.endswith('.pth'):
                    checkpoint = torch.load(model_file, map_location='cpu')
                    epsilon = checkpoint.get('epsilon', 'Unknown')
                    size_kb = os.path.getsize(model_file) / 1024
                    print(f"✅ {model_file}: epsilon={epsilon}, size={size_kb:.1f}KB")
                else:
                    size_kb = os.path.getsize(model_file) / 1024
                    print(f"✅ {model_file}: size={size_kb:.1f}KB")
            except:
                print(f"❌ {model_file}: corrupted")
        else:
            print(f"❌ {model_file}: not found")

def check_training_health():
    """Check if the model shows signs of healthy training"""
    model_path = r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\simple_dqn_model.pth"
    
    if not os.path.exists(model_path):
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("\n" + "=" * 50)
        print("🏥 TRAINING HEALTH CHECK")
        print("=" * 50)
        
        # Check epsilon
        epsilon = checkpoint.get('epsilon', 1.0)
        if epsilon <= 0.05:
            print("✅ Exploration: Well-converged (low epsilon)")
        elif epsilon <= 0.2:
            print("🔄 Exploration: Still balancing")
        else:
            print("🔍 Exploration: High exploration rate")
        
        # Check model weights
        if 'model_state_dict' in checkpoint:
            all_weights = []
            for name, param in checkpoint['model_state_dict'].items():
                if 'weight' in name:
                    all_weights.extend(param.flatten().tolist())
            
            if all_weights:
                weights_array = np.array(all_weights)
                print(f"✅ Weights: Mean={weights_array.mean():.4f}, Std={weights_array.std():.4f}")
                
                # Health indicators
                if abs(weights_array.mean()) < 0.1 and weights_array.std() > 0.01:
                    print("✅ Weight distribution: Healthy")
                elif weights_array.std() < 0.001:
                    print("⚠️ Weight distribution: Low variance (possible saturation)")
                else:
                    print("🔄 Weight distribution: Normal")
        
        # File info
        file_size = os.path.getsize(model_path) / 1024
        print(f"📁 File size: {file_size:.1f} KB")
        
        if file_size < 10:
            print("⚠️ Model file seems very small")
        elif file_size > 1000:
            print("✅ Model file has substantial learned knowledge")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    inspect_model()
    compare_models()
    check_training_health()