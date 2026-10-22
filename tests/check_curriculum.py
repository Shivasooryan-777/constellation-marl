"""
tests/check_curriculum.py - Verification script for Session S2.7: Curriculum Training over Jamming Intensity (Valid Re-Run).

Specifications:
1. Runs 5-stage curriculum training via src/curriculum_train.py:
   - Stage 1: 0.0 (Fresh untrained network)
   - Stage 2: 0.1 (Warm-start from Stage 1 checkpoint)
   - Stage 3: 0.3 (Warm-start from Stage 2 checkpoint)
   - Stage 4: 0.5 (Warm-start from Stage 3 checkpoint)
   - Stage 5: 0.7 (Warm-start from Stage 4 checkpoint)
2. Stability rule: Trailing 20-episode full-delivery success rate >= 70% maintained for 5 consecutive iterations.
3. Per-stage cap: 50 iterations OR 20 minutes.
4. Total session cap: 90 minutes.
5. Saves CSV log to logs/curriculum_run2.csv and plot to logs/curriculum_run2.png.
6. Saves per-stage checkpoints to checkpoints/curriculum_v2/stage_<intensity>_<status>.pt.
7. Logs hardware CPU/RAM baseline and post-stage readings.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.curriculum_train import train_curriculum
from src.ippo_train import get_system_memory, get_process_memory_mb


def main():
    print("================================================================================")
    print("CONSTELLATION — Session S2.7: Curriculum Training Verification (Valid Re-Run)")
    print("================================================================================\n")

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "curriculum_run2.csv"))
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "curriculum_run2.png"))
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "curriculum_v2"))

    t_start = time.time()

    # Execute Curriculum Training
    logs, summaries, agent = train_curriculum(
        seed=0,
        grid_size=20,
        n_drones=5,
        n_delivery_points=3,
        episode_max_steps=250,
        rollout_world_steps=410,
        stages=[0.0, 0.1, 0.3, 0.5, 0.7],
        stage_max_iterations=50,
        stage_time_limit_seconds=1200.0,   # 20 min per stage
        total_time_limit_seconds=5400.0,   # 90 min total hard stop
        target_success_rate=0.70,
        consecutive_stable_iterations=5,
        window_size=20,
        checkpoint_dir=checkpoint_dir,
        log_csv_path=csv_path,
        plot_png_path=plot_path,
    )

    t_end = time.time()
    total_duration_s = t_end - t_start

    # Final Resource Check
    sys_ram_total, sys_ram_used, sys_mem_pct = get_system_memory()
    proc_ram_mb = get_process_memory_mb()

    print("\n================================================================================")
    print("CURRICULUM TRAINING COMPLETE — FINAL SUMMARY")
    print("================================================================================")
    print(f"Total Wall-Clock Time:  {total_duration_s:.1f}s ({total_duration_s / 60:.2f} min)")
    print(f"Total Iterations Run:   {len(logs)}")
    print(f"Stages Attempted:       {len(summaries)}/5")
    print(f"Final System RAM:       {sys_ram_used:.2f} GB / {sys_ram_total:.2f} GB ({sys_mem_pct:.1f}%)")
    print(f"Final Process RAM:      {proc_ram_mb:.2f} MB")
    print("================================================================================\n")

    print("STAGE-BY-STAGE PERFORMANCE TABLE:")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Stage':<8} | {'Jammer':<8} | {'Status':<8} | {'Iters':<6} | {'Final Success':<14} | {'Duration (s)':<12} | {'Checkpoint'}")
    print("---------------------------------------------------------------------------------------------------------")
    for s in summaries:
        stg = s["stage"]
        intensity = f"{s['intensity']:.1f}"
        status = s["status"].upper()
        iters = f"{s['iterations']}"
        succ = f"{s['final_full_success']:.1%}"
        dur = f"{s['duration_s']:.1f}"
        ckpt = os.path.basename(s["checkpoint"])
        print(f"{stg:<8} | {intensity:<8} | {status:<8} | {iters:<6} | {succ:<14} | {dur:<12} | {ckpt}")
    print("---------------------------------------------------------------------------------------------------------\n")

    # Verification assertions
    print("Verifying artifacts...")
    assert os.path.exists(csv_path), f"CSV log {csv_path} does not exist!"
    assert os.path.getsize(csv_path) > 0, "CSV log is empty!"
    print(f"  [PASS] CSV log exists ({os.path.getsize(csv_path)} bytes): {csv_path}")

    assert os.path.exists(plot_path), f"PNG plot {plot_path} does not exist!"
    assert os.path.getsize(plot_path) > 0, "PNG plot is empty!"
    print(f"  [PASS] Plot PNG exists ({os.path.getsize(plot_path)} bytes): {plot_path}")

    for s in summaries:
        ckpt_path = s["checkpoint"]
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} missing!"
        assert os.path.getsize(ckpt_path) > 0, f"Checkpoint {ckpt_path} is empty!"
        print(f"  [PASS] Stage {s['stage']} Checkpoint exists ({os.path.getsize(ckpt_path)} bytes): {os.path.basename(ckpt_path)}")

    print("\nAll curriculum checks and artifact validations passed successfully!")


if __name__ == "__main__":
    main()
