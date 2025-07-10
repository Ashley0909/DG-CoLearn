import ray
import torch
import time
ray.init()

@ray.remote(num_gpus=1)
def continuous_gpu_compute():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [],
        "computation_count": 0
    }

    if gpu_info["cuda_available"]:
        for i in range(gpu_info["gpu_count"]):
            gpu_info["gpu_names"].append(torch.cuda.get_device_name(i))

    while True:
        try:
            # Batch of matmul operations to stress GPU
            tensor_a = torch.randn(5000, 5000, device=device)
            tensor_b = torch.randn(5000, 5000, device=device)

            for _ in range(5):  # Repeat to make GPU stay busy longer
                result = torch.matmul(tensor_a, tensor_b)
                result = torch.matmul(result, tensor_a)
                result = torch.matmul(result, tensor_b)

            torch.cuda.synchronize()  # Force GPU to complete all work before moving on

            gpu_info["computation_count"] += 1

            if gpu_info["computation_count"] % 2 == 0:
                print(f"Completed {gpu_info['computation_count']} batches on {device}")

        except Exception as e:
            gpu_info["error"] = str(e)
            return gpu_info

        # Optional: remove sleep or reduce it
        # time.sleep(0.01)


nodes = ray.nodes()
active_nodes = [node for node in nodes if node['Alive']]
tasks = []

for node in active_nodes:
    tasks.append(continuous_gpu_compute.remote())

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
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopping GPU computations...")
    ray.shutdown()
