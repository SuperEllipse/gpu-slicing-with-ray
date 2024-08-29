#!pip install ray modin

import ray
import modin.pandas as pd
import numpy as np

# Initialize Ray with GPU resources
ray.init()

@ray.remote(num_gpus=0.5)
def process_large_dataset(task_id, num_rows):
    print(f"Task {task_id} using 0.5 GPU started.")
    
    # Create a large DataFrame using Modin
    df = pd.DataFrame(np.random.rand(num_rows, 10), columns=[f'col_{i}' for i in range(10)])
    
    # Perform some operations on the DataFrame
    result = df.apply(lambda x: x ** 2).sum()
    
    print(f"Task {task_id} using 0.5 GPU completed.")
    return f"Result of task {task_id} is:\n{result}"

@ray.remote(num_gpus=0.25)
def filter_large_dataset(task_id, num_rows):
    print(f"Task {task_id} using 0.25 GPU started.")
    
    # Create a large DataFrame using Modin
    df = pd.DataFrame(np.random.rand(num_rows, 10), columns=[f'col_{i}' for i in range(10)])
    
    # Filter the DataFrame
    filtered_df = df[df['col_0'] > 0.5]
    
    print(f"Task {task_id} using 0.25 GPU completed.")
    return f"Filtered DataFrame for task {task_id} has {len(filtered_df)} rows"

def main():
    num_rows = 10**6  # Example size of the dataset

    # Launch tasks with GPU slicing
    task1 = process_large_dataset.remote(1, num_rows)
    task2 = filter_large_dataset.remote(2, num_rows)
    task3 = process_large_dataset.remote(3, num_rows)

    # Gather results
    results = ray.get([task1, task2, task3])

    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
    ray.shutdown()
