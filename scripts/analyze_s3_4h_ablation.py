"""
scripts/analyze_s3_4h_ablation.py - S3.4h IPPO vs MAPPO Ablation Analysis (Pure Standard Library).

Performs direct comparison between IPPO and MAPPO across all 5 jamming levels:
1. Loads new-reward granular data (logs/ippo_jamming_sweep_newreward_granular.csv,
   logs/mappo_jamming_sweep_newreward_granular.csv).
2. Computes mean and sample std (ddof=1) across all 5 seeds (0..4) for:
   - Success Rate (eval_full_success)
   - Collisions (eval_collisions)
   - Latency (eval_steps_to_last_delivery)
   - Energy (eval_energy)
3. Computes robustness check excluding Seed 1 (seeds 0, 2, 3, 4: N=4).
4. Loads old-reward dataset (combined_jamming_sweep_5seeds.csv) for quick sanity comparison.
5. Emits side-by-side comparison tables and analysis.
"""

import os
import csv
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_csv(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = ""
                elif v.lower() == "true":
                    parsed[k] = True
                elif v.lower() == "false":
                    parsed[k] = False
                else:
                    try:
                        parsed[k] = int(v) if "." not in v else float(v)
                    except ValueError:
                        parsed[k] = v
            records.append(parsed)
    return records

def calc_stats(records, metric_key):
    vals = [float(r[metric_key]) for r in records]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std

def run_analysis():
    ippo_gran_path = os.path.join(PROJECT_ROOT, "logs", "ippo_jamming_sweep_newreward_granular.csv")
    mappo_gran_path = os.path.join(PROJECT_ROOT, "logs", "mappo_jamming_sweep_newreward_granular.csv")
    old_comb_path = os.path.join(PROJECT_ROOT, "logs", "combined_jamming_sweep_5seeds.csv")

    ippo_recs = load_csv(ippo_gran_path)
    mappo_recs = load_csv(mappo_gran_path)
    old_comb_recs = load_csv(old_comb_path)

    jam_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
    metrics = [
        ("eval_full_success", "Full Success Rate", "{:.1%}", "{:.1%}"),
        ("eval_collisions", "Collisions", "{:.2f}", "{:.2f}"),
        ("eval_steps_to_last_delivery", "Latency (steps)", "{:.2f}", "{:.2f}"),
        ("eval_energy", "Energy", "{:.2f}", "{:.2f}"),
    ]

    print("====================================================================================================")
    print("S3.4h IPPO vs MAPPO ABLATION — NEW REWARD DATASET (S3.4g)")
    print("====================================================================================================\n")

    # 1. Full 5-seed comparison
    print("----------------------------------------------------------------------------------------------------")
    print("PART 1: FULL 5-SEED COMPARISON (ALL SEEDS 0-4)")
    print("----------------------------------------------------------------------------------------------------")
    for key, name, m_fmt, s_fmt in metrics:
        print(f"\n--- Metric: {name} ---")
        print(f"{'Jamming':<8} | {'IPPO (mean +/- std)':<25} | {'MAPPO (mean +/- std)':<25} | {'Delta (MAPPO-IPPO)':<20} | {'Winner':<12} | {'Overlap / Real'}")
        print("-" * 115)
        for lvl in jam_levels:
            i_sub = [r for r in ippo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            m_sub = [r for r in mappo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            
            i_mean, i_std = calc_stats(i_sub, key)
            m_mean, m_std = calc_stats(m_sub, key)
            delta = m_mean - i_mean

            # For success, higher is better; for collisions, latency, energy, lower is better.
            if key == "eval_full_success":
                better_algo = "MAPPO" if delta > 0 else ("IPPO" if delta < 0 else "TIE")
                d_str = f"{delta:+.1%}"
            else:
                better_algo = "MAPPO" if delta < 0 else ("IPPO" if delta > 0 else "TIE")
                d_str = f"{delta:+.2f}"

            # Noise / real assessment: do 1-std confidence intervals overlap?
            i_low, i_high = i_mean - i_std, i_mean + i_std
            m_low, m_high = m_mean - m_std, m_mean + m_std
            overlap = max(i_low, m_low) <= min(i_high, m_high)
            significance = "Within Noise (overlap)" if overlap else "Separated (Real gap)"

            i_str = f"{m_fmt.format(i_mean)} +/- {s_fmt.format(i_std)}"
            m_str = f"{m_fmt.format(m_mean)} +/- {s_fmt.format(m_std)}"

            print(f"{lvl:<8.0%} | {i_str:<25} | {m_str:<25} | {d_str:<20} | {better_algo:<12} | {significance}")

    # 2. Combined Single Summary Table (with & without Seed 1)
    print("\n----------------------------------------------------------------------------------------------------")
    print("PART 2: ROBUSTNESS CHECK — EXCLUDING SEED 1 (SEEDS 0, 2, 3, 4; N=4)")
    print("----------------------------------------------------------------------------------------------------")
    for key, name, m_fmt, s_fmt in metrics:
        print(f"\n--- Metric: {name} (Excluding Seed 1) ---")
        print(f"{'Jamming':<8} | {'IPPO (no S1)':<25} | {'MAPPO (no S1)':<25} | {'Delta (MAPPO-IPPO)':<20} | {'Winner':<12} | {'Change vs 5-seed'}")
        print("-" * 115)
        for lvl in jam_levels:
            i_sub = [r for r in ippo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4 and r["seed"] != 1]
            m_sub = [r for r in mappo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4 and r["seed"] != 1]
            
            i_mean, i_std = calc_stats(i_sub, key)
            m_mean, m_std = calc_stats(m_sub, key)
            delta = m_mean - i_mean

            i_sub5 = [r for r in ippo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            m_sub5 = [r for r in mappo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            i_m5, _ = calc_stats(i_sub5, key)
            m_m5, _ = calc_stats(m_sub5, key)
            d5 = m_m5 - i_m5

            if key == "eval_full_success":
                better_algo = "MAPPO" if delta > 0 else ("IPPO" if delta < 0 else "TIE")
                d_str = f"{delta:+.1%}"
                better5 = "MAPPO" if d5 > 0 else ("IPPO" if d5 < 0 else "TIE")
            else:
                better_algo = "MAPPO" if delta < 0 else ("IPPO" if delta > 0 else "TIE")
                d_str = f"{delta:+.2f}"
                better5 = "MAPPO" if d5 < 0 else ("IPPO" if d5 > 0 else "TIE")

            i_str = f"{m_fmt.format(i_mean)} +/- {s_fmt.format(i_std)}"
            m_str = f"{m_fmt.format(m_mean)} +/- {s_fmt.format(m_std)}"
            change = "Same Winner" if better_algo == better5 else f"FLIPPED (was {better5})"

            print(f"{lvl:<8.0%} | {i_str:<25} | {m_str:<25} | {d_str:<20} | {better_algo:<12} | {change}")

    # 3. Side-by-side With vs Without Seed 1 comparison
    print("\n----------------------------------------------------------------------------------------------------")
    print("PART 3: COMPREHENSIVE SIDE-BY-SIDE SUMMARY TABLE (WITH SEED 1 vs WITHOUT SEED 1)")
    print("----------------------------------------------------------------------------------------------------")
    for key, name, m_fmt, s_fmt in metrics:
        print(f"\nMetric: {name}")
        print(f"{'Jamming':<8} | {'5-Seed IPPO':<18} | {'5-Seed MAPPO':<18} | {'5-Seed Win':<11} | {'No-S1 IPPO':<18} | {'No-S1 MAPPO':<18} | {'No-S1 Win':<11} | {'Winner Flip?'}")
        print("-" * 120)
        for lvl in jam_levels:
            # 5-seed
            i_sub5 = [r for r in ippo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            m_sub5 = [r for r in mappo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
            i_m5, i_s5 = calc_stats(i_sub5, key)
            m_m5, m_s5 = calc_stats(m_sub5, key)
            d5 = m_m5 - i_m5
            if key == "eval_full_success":
                w5 = "MAPPO" if d5 > 0 else ("IPPO" if d5 < 0 else "TIE")
            else:
                w5 = "MAPPO" if d5 < 0 else ("IPPO" if d5 > 0 else "TIE")

            # no seed 1
            i_sub4 = [r for r in ippo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4 and r["seed"] != 1]
            m_sub4 = [r for r in mappo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4 and r["seed"] != 1]
            i_m4, i_s4 = calc_stats(i_sub4, key)
            m_m4, m_s4 = calc_stats(m_sub4, key)
            d4 = m_m4 - i_m4
            if key == "eval_full_success":
                w4 = "MAPPO" if d4 > 0 else ("IPPO" if d4 < 0 else "TIE")
            else:
                w4 = "MAPPO" if d4 < 0 else ("IPPO" if d4 > 0 else "TIE")

            flipped = "NO (Stable)" if w5 == w4 else f"YES ({w5} -> {w4})"
            i_str5 = f"{m_fmt.format(i_m5)}+/-{s_fmt.format(i_s5)}"
            m_str5 = f"{m_fmt.format(m_m5)}+/-{s_fmt.format(m_s5)}"
            i_str4 = f"{m_fmt.format(i_m4)}+/-{s_fmt.format(i_s4)}"
            m_str4 = f"{m_fmt.format(m_m4)}+/-{s_fmt.format(m_s4)}"

            print(f"{lvl:<8.0%} | {i_str5:<18} | {m_str5:<18} | {w5:<11} | {i_str4:<18} | {m_str4:<18} | {w4:<11} | {flipped}")

    # 4. Seed 1 specific values across all conditions
    print("\n----------------------------------------------------------------------------------------------------")
    print("PART 4: SEED 1 EXACT METRICS ACROSS BOTH ALGORITHMS")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Jamming':<8} | {'Algo':<8} | {'Final Iter':<11} | {'Train FullSucc':<15} | {'Eval FullSucc':<14} | {'Collisions':<12} | {'Latency':<10} | {'Energy':<10} | {'Flagged <=20%?'}")
    print("-" * 115)
    for lvl in jam_levels:
        for algo, recs in [("IPPO", ippo_recs), ("MAPPO", mappo_recs)]:
            s1 = [r for r in recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4 and r["seed"] == 1][0]
            efs = float(s1["eval_full_success"])
            flag = "YES (WEAK)" if efs <= 0.20 else "No"
            print(f"{lvl:<8.0%} | {algo:<8} | {s1['final_iteration']:<11} | {float(s1['full_success_rate']):<15.1%} | {efs:<14.1%} | {float(s1['eval_collisions']):<12.1f} | {float(s1['eval_steps_to_last_delivery']):<10.1f} | {float(s1['eval_energy']):<10.1f} | {flag}")

    # 5. Old reward sanity comparison
    print("\n----------------------------------------------------------------------------------------------------")
    print("PART 5: OLD REWARD (S3.4a) SANITY COMPARISON (FULL SUCCESS RATE)")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Jamming':<8} | {'IPPO Old (5s)':<18} | {'MAPPO Old (5s)':<18} | {'Old Winner':<12} | {'New Winner (5s)':<15} | {'Agreement'}")
    print("-" * 95)
    for lvl in jam_levels:
        i_old = [r for r in old_comb_recs if r.get("algorithm") == "ippo" and abs(float(r["jammer_intensity"]) - lvl) < 1e-4][0]
        m_old = [r for r in old_comb_recs if r.get("algorithm") == "mappo" and abs(float(r["jammer_intensity"]) - lvl) < 1e-4][0]

        i_old_m = float(i_old["eval_full_success_mean"])
        m_old_m = float(m_old["eval_full_success_mean"])
        old_win = "MAPPO" if m_old_m > i_old_m else ("IPPO" if i_old_m > m_old_m else "TIE")

        i_new = [r for r in ippo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
        m_new = [r for r in mappo_recs if abs(float(r["jammer_intensity"]) - lvl) < 1e-4]
        i_new_m, _ = calc_stats(i_new, "eval_full_success")
        m_new_m, _ = calc_stats(m_new, "eval_full_success")
        new_win = "MAPPO" if m_new_m > i_new_m else ("IPPO" if i_new_m > m_new_m else "TIE")

        agree = "Agrees" if old_win == new_win else f"Reversed (Old={old_win}, New={new_win})"
        i_str = f"{i_old_m:.1%} +/- {float(i_old['eval_full_success_std']):.1%}"
        m_str = f"{m_old_m:.1%} +/- {float(m_old['eval_full_success_std']):.1%}"
        print(f"{lvl:<8.0%} | {i_str:<18} | {m_str:<18} | {old_win:<12} | {new_win:<15} | {agree}")

if __name__ == "__main__":
    run_analysis()
