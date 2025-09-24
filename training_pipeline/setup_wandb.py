# setup_wandb.py - Setup script for wandb
import wandb
import os

def setup_wandb():
    """Setup wandb authentication"""
    
    # Check if wandb is logged in
    try:
        wandb.login()
        print("✅ Wandb already logged in")
    except:
        print("⚠️  Please login to wandb:")
        print("1. Get your API key from: https://wandb.ai/authorize")
        print("2. Run: wandb login")
        print("3. Or set environment variable: WANDB_API_KEY=your_api_key")
        
    # Test connection to your project
    try:
        run = wandb.init(
            project="finetune-my-diary", 
            mode="disabled"  # Don't actually log anything
        )
        run.finish()
        print("✅ Successfully connected to 'finetune-my-diary' project")
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

if __name__ == "__main__":
    setup_wandb()