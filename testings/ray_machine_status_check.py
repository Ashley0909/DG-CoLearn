import ray
import torch
import time
ray.init()

@ray.remote(num_gpus=1)
def continuous_gpu_compute():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Debug print
    
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [],
        "computation_count": 0,
        "total_time": 0
    }
    
    if gpu_info["cuda_available"]:
        for i in range(gpu_info["gpu_count"]):
            gpu_info["gpu_names"].append(torch.cuda.get_device_name(i))
            print(f"GPU {i}: {gpu_info['gpu_names'][-1]}")  # Debug print
    
    while True:
        try:
            start_time = time.time()
            # Create larger tensors and perform more intensive computation
            tensor_a = torch.randn(5000, 5000, device=device)
            tensor_b = torch.randn(5000, 5000, device=device)
            
            # Force GPU computation with multiple operations
            result = torch.matmul(tensor_a, tensor_b)
            result = torch.matmul(result, tensor_a)  # Additional computation
            result = torch.matmul(result, tensor_b)  # Additional computation
            
            # Ensure computation is complete
            torch.cuda.synchronize()
            
            # Verify tensor is on GPU
            if gpu_info["computation_count"] % 10 == 0:
                print(f"Tensor device: {result.device}")  # Debug print
            
            end_time = time.time()
            
            gpu_info["computation_count"] += 1
            gpu_info["total_time"] += (end_time - start_time)
            gpu_info["avg_time"] = gpu_info["total_time"] / gpu_info["computation_count"]
            
            if gpu_info["computation_count"] % 10 == 0:  # More frequent updates
                print(f"Completed {gpu_info['computation_count']} computations, avg time: {gpu_info['avg_time']:.4f}s")
                print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")  # Debug print
                
        except Exception as e:
            gpu_info["error"] = str(e)
            print(f"Error occurred: {e}")  # Debug print
            return gpu_info
            
        time.sleep(0.01)  # Reduced delay for more frequent computation

nodes = ray.nodes()
active_nodes = [node for node in nodes if node['Alive']]
print(f"Active nodes: {len(active_nodes)}")  # Debug print
tasks = []

# Start continuous computation on each node
for node in active_nodes:
    tasks.append(continuous_gpu_compute.remote())

# Monitor results periodically
try:
    while True:
        for i, task in enumerate(tasks):
            try:
                result = ray.get(task, timeout=1)
                if 'error' in result:
                    node_ip = active_nodes[i]['NodeManagerAddress']
                    print(f"Error on node {node_ip}: {result['error']}")
                    tasks[i] = continuous_gpu_compute.remote()
            except ray.exceptions.GetTimeoutError:
                continue
        time.sleep(2)  # More frequent checks
except KeyboardInterrupt:
    print("\nStopping GPU computations...")
    ray.shutdown()
