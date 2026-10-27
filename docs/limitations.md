# Known Limitations (Phase 3)

## No-fly zones are not part of the learning problem

No-fly zones (NFZs) are simulated by `DroneWorld` (moving, deterministic, pattern-based) and tracked as a
metric (`no_fly_zone_collisions` in `MetricsTracker`), but they are neither observable nor penalized in
reward: the 7-float per-drone observation contains no zone information, and `rewards.py` has no NFZ term.
As a result, current policies (IPPO and MAPPO, all checkpoints) have no learned avoidance behavior, and no
meaningful generalization to unseen NFZ patterns can exist or be tested; any NFZ-collision differences seen
in S3.5/S3.6 reflect incidental trajectory overlap, not learned behavior. This is a documented scope
boundary decided at S3.9 (Phase 3 closeout), not a bug. Adding NFZ observation/reward coupling is a new
feature that would reopen settled design decisions and requires explicit owner sign-off; it is out of
scope for Phase 3 and should be stated as a limitation in the Phase 5 write-up.

Evidence: S3.5 (`logs/runs/s3_5_nofly_diagnostic_*.log`), S3.6 (`logs/runs/s3_6_nfz_baseline_*.log`).
