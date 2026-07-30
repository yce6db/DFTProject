print("STEP 1: script started", flush=True)

from pathlib import Path
print("STEP 2: imports done", flush=True)

import matplotlib
print(f"STEP 3: matplotlib backend = {matplotlib.get_backend()}", flush=True)

import matplotlib.pyplot as plt
print("STEP 4: pyplot imported", flush=True)

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
print("STEP 5: figure created", flush=True)

output_path = Path.cwd() / "test_output.png"
fig.savefig(output_path)
print(f"STEP 6: saved to {output_path}", flush=True)
print(f"STEP 7: file exists? {output_path.exists()}", flush=True)
