import time
import statistics
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototype"))
from pipeline_model import pipeline

def run_benchmark(n_iterations=100):
    # Sample input (taken from the database)
    sample = {
        "ja4": "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "ja4s": "t130200_1302_a56c5b993250",
        "ja4ts": "64240_2-1-1-4-1-3_1250_7",
        "ja4_string": "t13d1516h2_002f,0035,009c,009d,1301,1302,1303,c013,c014,c02b,c02c,c02f,c030,cca8,cca9_0005,000a,000b,000d,0012,0017,001b,0023,002b,002d,0033,44cd,fe0d,ff01_0403,0804,0401,0503,0805,0501,0806,0601",
        "ja4s_string": "t130200_1302_002b,0033"
    }

    print(f"Benchmarking pipeline latency over {n_iterations} iterations...")
    
    # Warm up (load models/DB into memory)
    start_warm = time.perf_counter()
    pipeline.classify(**sample)
    warm_up_time = (time.perf_counter() - start_warm) * 1000
    print(f"Warm-up (first run + loading): {warm_up_time:.2f} ms\n")

    latencies = []
    
    for i in range(n_iterations):
        start = time.perf_counter()
        pipeline.classify(**sample)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # convert to ms
        
    avg_time = statistics.mean(latencies)
    median_time = statistics.median(latencies)
    min_time = min(latencies)
    max_time = max(latencies)
    
    print("--- Results ---")
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Median time:  {median_time:.2f} ms")
    print(f"Min time:     {min_time:.2f} ms")
    print(f"Max time:     {max_time:.2f} ms")
    print(f"Total time for 100 runs: {sum(latencies)/1000:.2f} seconds")

if __name__ == "__main__":
    run_benchmark()
