# After pipeline completes, in main process
import os
import pstats

for i in range(max(2, (os.cpu_count() // 2))):
    path = f"worker_{i}_profile.stats"
    if os.path.exists(path):
        print(f"\n--- Worker {i} ---")
        stats = pstats.Stats(path)
        stats.strip_dirs()
        stats.sort_stats("tottime")
        stats.print_stats(10)
        # os.remove(path)
