import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import csv, os, subprocess, sys


def run_and_save(cmd, outfile):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open(outfile, "w") as f:
        f.write(result.stdout)
    return outfile

def read_q1(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[int(row["K_millions"])] = float(row["time_ms"])
    return data

def read_q23(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["scenario"]
            k = int(row["K_millions"])
            t = float(row["time_ms"])
            data.setdefault(s, {})[k] = t
    return data


script_dir = os.path.dirname(os.path.abspath(__file__))

q1_csv = os.path.join(script_dir, "q1_results.csv")
q2_csv = os.path.join(script_dir, "q2_results.csv")
q3_csv = os.path.join(script_dir, "q3_results.csv")

if not os.path.exists(q1_csv):
    run_and_save(f"{script_dir}/q1", q1_csv)
if not os.path.exists(q2_csv):
    run_and_save(f"{script_dir}/q2", q2_csv)
if not os.path.exists(q3_csv):
    run_and_save(f"{script_dir}/q3", q3_csv)

cpu_data = read_q1(q1_csv)
gpu_data = read_q23(q2_csv)        
uni_data = read_q23(q3_csv)       

K_list = sorted(cpu_data.keys())   

scenario_labels = {
    "1block_1thread":      "GPU: 1 block / 1 thread",
    "1block_256threads":   "GPU: 1 block / 256 threads",
    "Nblocks_256threads":  "GPU: N blocks / 256 threads",
}
colors = {
    "1block_1thread":     "tab:orange",
    "1block_256threads":  "tab:green",
    "Nblocks_256threads": "tab:blue",
}


def make_plot(gpu_dict, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(K_list, [cpu_data[k] for k in K_list],
            marker="s", color="tab:red", linestyle="--", label="CPU (host only)")

    for s, label in scenario_labels.items():
        if s not in gpu_dict:
            continue
        times = [gpu_dict[s].get(k, float("nan")) for k in K_list]
        ax.plot(K_list, times, marker="o", color=colors[s], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (million elements)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title(title)
    ax.set_xticks(K_list)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)

import matplotlib.ticker

make_plot(gpu_data, "Vector Addition: GPU (explicit memory) vs CPU",
          os.path.join(script_dir, "q4_without_unified.jpg"))

make_plot(uni_data, "Vector Addition: GPU (unified memory) vs CPU",
          os.path.join(script_dir, "q4_with_unified.jpg"))

print("Done.")
