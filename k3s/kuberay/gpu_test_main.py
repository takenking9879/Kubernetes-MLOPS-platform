import ray
import torch
import torch.nn as nn
import torch.optim as optim
import time

@ray.remote(num_gpus=1)
def train_gpu_task():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")
    
    if not torch.cuda.is_available():
        return "Error: CUDA not available on this worker"

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Minimal Neural Network
    model = nn.Linear(10, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Synthetic Data
    inputs = torch.randn(64, 10).to(device)
    targets = torch.randn(64, 1).to(device)

    # Minimal Training Loop
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return f"Training finished on GPU: {torch.cuda.get_device_name(0)}"

if __name__ == "__main__":
    # Connect to the existing Ray cluster
    ray.init(address="auto")
    
    print("Submitting GPU task to Ray...")
    result_ref = train_gpu_task.remote()
    
    try:
        result = ray.get(result_ref, timeout=300)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Task failed or timed out: {e}")
