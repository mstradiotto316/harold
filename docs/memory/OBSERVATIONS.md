# Harold Observations & Insights

## 2026-03-18: Coordinate Frame & Quaternion Convention

### Isaac Lab Quaternion Convention
- **Isaac Lab uses (w, x, y, z) quaternion format.** Confirmed from official API docs.
- Identity quaternion = `(1.0, 0.0, 0.0, 0.0)` (w=1, no rotation)
- 180° Z rotation = `(0.0, 0.0, 0.0, 1.0)` (w=0, z=1)
- This applies to `ArticulationCfg.InitialStateCfg.rot`, `write_root_pose_to_sim()`, and all pose/state APIs.

### Coordinate Frame Investigation (RESOLVED 2026-03-19)
- **No bug found.** Identity quaternion is correct: body +X = world +X = visual forward.
- A 180° Z rotation was temporarily applied (EXP-429) based on incorrect URDF joint analysis. Reverted after diagnostic testing proved the original setup was correct.
- Diagnostic: pushing env 0 at +2.0 m/s world +X confirmed `vx_b ≈ vx_w` (positive) with identity quat. Forward rewards correctly tracked positive body-frame velocity.
- The performance improvement attributed to the rotation was likely coincidental.
- See `COORDINATE_FRAME_BUG.md` for full investigation details.

### Body-Frame Velocity (`root_lin_vel_b`)
- Computed via `quat_apply_inverse(root_quat_w, root_lin_vel_w)` — always relative to CURRENT body orientation.
- With identity quaternion: `vx_b = vx_w`, `vy_b = vy_w` (at spawn; diverges during yaw rotation).
- Commands (`cmd_vx`, `cmd_vy`) are body-frame — reward is rotation-invariant.
- **Hardware deployment**: IMU +X must point toward the robot's visual forward for alignment with sim.

## 2026-03-15: Repo Secret Hygiene
- A tracked `*.nmconnection` file is still a secret leak even if `.gitignore` already lists `*.nmconnection`; ignore rules do not retroactively untrack files.
- Public-secret cleanup for git requires both steps: remove the file from `HEAD` and rewrite history. Deleting only the current file is insufficient once the repo has been pushed.
- After a public-history rewrite, cached clones/forks may still hold the credential. Treat the credential as compromised and rotate it.

## 2026-03-16: Export Metadata + Status Diagnostics
- Old direct `skrl` runs can be manifestless but still preserve the training config under `params/env.yaml`; export fallback logic should prefer that snapshot over repo defaults.
- Isaac Lab `params/env.yaml` snapshots use `!!python/tuple` tags. `yaml.safe_load` fails on these files; `yaml.full_load` is required if we want to read the saved config directly.
- Category-level export metadata should fail closed if a saved env snapshot ever contains per-joint ranges or limits that differ within a category. Silently collapsing asymmetric per-joint limits into one category bound would corrupt deployment/offline replay.
- `harold status` auxiliary termination counters only resolve when the TensorBoard tags include the `Info /` prefix used by skrl (`Info / Episode_Termination/*` and `Info / Episode_Metric/termination_*`).

## 2026-03-15: Desktop Environment And Runtime Context
- Desktop Isaac Lab work should use `/home/matteo/Desktop/env_isaaclab`; Raspberry Pi runtime work should use system `python3`.
- `isaaclab` import success in the desktop venv does not imply Isaac Sim runtime modules are available.
- `omni.*` import failures from a plain shell are usually runtime-context failures, not missing-package failures.
- For simulator-backed checks, agents should prefer `python scripts/harold.py ...` or Isaac Lab launcher entrypoints over ad hoc import probes.

## 2026-03-19: Session 49 (16384 envs) — Damage Control Findings

### All 10 experiments produced standing/falling — not walking
- EXP-488 through EXP-498 all showed the robot standing still or falling forward in video, despite positive vx metrics.
- **vx metric is unreliable at 16384 envs**: falling forward generates positive vx that looks like locomotion in metrics. Video verdict overrides metrics — always.
- Metrics were repeatedly trusted over video agent verdicts during the session, leading to a falsely-KEPTed config (EXP-493) and 4 follow-up experiments against a wrong baseline. All corrections applied post-session.

### Two-phase training infrastructure works
- `harold record` post-hoc video recording works correctly at 1 env from best checkpoint.
- `use_fabric=True` is required in record.py — `False` causes frozen T-pose rendering.
- The 16384-env throughput (2.88M samples/s) is real, but walking has not been achieved at this env count yet.

### PPO hyperparameters may need tuning for 16384 envs
- Walking was previously confirmed at 1024 and 4096 envs. Jumping to 16384 without adjusting PPO rollouts/learning-rate/mini-batches may explain the standing policies.
- The batch size at 16384 envs is 4x larger than at 4096 — PPO effective learning rate and advantage estimation change significantly.

## 2026-03-16: Autoresearch Session 47 - Key Findings
- **1024 envs is transformative**: Doubled environments from 512 to 1024 increased ep_len from ~123 to ~170-191 without any reward changes. More diverse training data produces much more stable policies.
- **Walking basin is extremely fragile at 512 envs**: ANY reward function change at 512 envs (even 5x increase in near-zero lin_vel_z penalty) pushes policy to standing/crouching. The walking equilibrium is narrow.
- **upright_weight=2.0 + stance_height=3.0 + 1024 envs is the new best config**: Produces ep_len=174, vx=0.057, upright=0.906 with confirmed forward displacement in video. Front legs now motor-active.
- **Phased reward curriculum produces STANDING, not walking**: All variants (floor 0, 0.3, 0.5; phase 30, 100 steps) converge to standing policies confirmed by video. Curriculum prevents the walking equilibrium from forming.
- **Video review is essential**: Metrics like upright=0.9+ and ep_len=189 can mask DEGENERATE behavior (crouching, static standing). Video reveals the true policy behavior that metrics miss.
- **Front-leg passivity is the current bottleneck**: Video shows rear legs stepping with clearance, but front legs act as passive props. Need to activate front legs for coordinated gait.
- **feet_air_time_threshold reduction (0.3→0.2) produces in-place stepping**: More foot lifting but zero forward progress. Air time reward doesn't care about step direction.

## 2026-03-15: Autoresearch Session 46 - Key Findings
- **World-frame vs body-frame velocities**: The code upgrade changed track_lin_vel_xy and forward_motion rewards from world-frame to body-frame velocities. This caused the robot to appear to go backward (negative vx_w_mean) because it could earn reward by walking in any direction in its own frame. Reverting to world-frame velocities for rewards fixed this.
- **upright_weight=3.0 + orientation_threshold=-0.6 is the best config**: This combination produced vx=0.109 (best ever), exceeding EXP-246. The relaxed orientation threshold allows more dynamic motion while the high upright weight prevents exploit behavior.
- **30-min training is optimal**: vx peaks at ~25-30 min then oscillates/regresses. 15 min too short, 60 min causes regression.
- **Gait quality is the next bottleneck**: Video shows chaotic shuffle with near-zero foot clearance, not actual stepping. Increasing feet_air_time or rewards_shaper_scale trades vx for posture - doesn't break shuffle equilibrium.
- **Must use `/home/matteo/Desktop/env_isaaclab/bin/python` for all harold.py commands** (sys.executable in harold.py picks up system python which lacks isaaclab).

## 2026-03-15: Simulation Audit Findings
- Flat-task reward/command telemetry currently mixes world-frame velocity with body-frame observations and commands; this should be treated as a correctness bug, not tuning noise.
- Flat-task forward reward still leaks positive reward into low-upright states, so the existing anti-fall guard is weaker than intended.
- Rough-task domain randomization currently overstates robustness: several reset-time randomization paths sample values but do not apply them to simulator physics.
- Rough-task terrain sampling is limited to the easiest levels under current config/use of `max_init_terrain_level`.
- Export/deployment tooling still contains 50D policy artifacts and assumptions even though the active sim/controller stack is now 48D.
- Optional sim policy logging breaks for multi-env runs because `_time` is serialized as if it were scalar.
- EMA action-filter state carries across episode resets unless explicitly cleared.

## 2026-03-15: Audit Remediation Outcomes
- Flat-task reward alignment is now body-frame end-to-end: observations, velocity-tracking reward, and command-error telemetry all compare body-frame commands to `root_lin_vel_b`.
- Forward reward leakage is closed: clearly fallen/contact-heavy states now get zero or negative forward reward instead of farming positive velocity reward.
- Reset hygiene matters in PyTorch: advanced-index calls like `tensor[env_ids].zero_()` do not write back. Reset helpers must assign (`tensor[env_ids] = 0`) or use an in-place indexed op.
- The `harold.py` launcher cannot rely on `bash ... & echo $!` for long Isaac runs. Detached `subprocess.Popen(..., start_new_session=True)` fixes PID tracking, keeps the trainer alive after the parent exits, and lets the watchdog attach reliably.
- Detached launch still must pin the trainer to `~/Desktop/env_isaaclab/bin/python`; using the caller's `sys.executable` breaks `python3 scripts/harold.py train ...` outside the Isaac Lab venv.
- The custom multi-camera recorder had two blockers: the USD camera transform constructor was invalid for `Gf.Matrix4d`, and buffering whole videos in Python caused large step-0 memory spikes. Using the 16-arg matrix constructor plus streaming frames directly to `ffmpeg` fixes both issues.
- Rough/task video must tolerate envs without `capture_multi_cameras()`. A `render()` fallback is enough to preserve the mandatory-video rule on rough runs.
- Rough-task randomization is now materially applied: friction, joint stiffness, joint damping, and mass/inertia scaling update simulator properties at reset. Effective values show up in TensorBoard under `Episode_Metric/randomized_*`.
- Rough terrain sampling now spans the generated terrain range instead of staying pinned to the easiest levels. TensorBoard now shows `terrain_level_min/mean/max` so coverage is visible.
- Termination counters only show up in TensorBoard when logged as tensor scalars, not plain Python numbers. Mirroring them into `Episode_Metric/termination_*` makes the reset reasons observable in the current skrl logging pipeline.
- Export/deployment is now checkpoint-derived 48D: exporter metadata, quick validation, ONNX-vs-sim validation, offline conversion, and controller metadata loading all agree on 48D and the canonical stance/config.
- Export metadata stores `checkpoint_path` repo-relatively, so local validators must resolve that path against the repo root instead of the current working directory.
- Harold manifests use `started_at` as the authoritative start timestamp; autoresearch backfills should only fall back to `created` for older manifests.
- The desktop Isaac Lab venv needed `deployment/requirements.txt` installed (`onnxruntime`, `pyserial`, `smbus2`) before local export/controller validation could run end-to-end.

## 2026-03-15: Deployment Sign Convention Alignment
- The real robot path treats shoulders as non-mirrored semantic joints; only thighs/calves are sign-inverted. Right-side servo mirroring already lives in `deployment/config/hardware.yaml` `direction` and the ESP32 `DIR_TABLE`, not in high-level shoulder signs.
- `deployment/config/cpg.yaml` had drifted from the hardware path by mirroring the FR/BR shoulders. That mismatch was already present on `main`; export metadata with shoulder signs `[1, 1, 1, 1]` matched the real hardware code better than the stale CPG config did.
- Deployment should fail closed if export metadata and `hardware.yaml` disagree on `joint_sign`. Silently choosing one convention risks commanding the opposite physical shoulder motion from what the controller observes.
- `harold train` should reject active or orphan training processes instead of auto-stopping them on re-invocation. Explicit `harold stop` is safer than destructive launch-time cleanup.

## 2026-01-04: Hardware CPG Baseline (Current)
- Duty-cycle stance/swing gait reduced foot drag; shorter stride reduced impact.
- Lowering calf lift softened touchdown without reintroducing severe drag.
- Rear legs improved from dragging to skimming but still no clear air time.
- ESP32 handshake failed after calibration firmware; reflash StreamingControl restored comms.
- `harold` systemd service auto-restarts gait; keep it inactive during manual observation.
- Test 3 (0.4 Hz, duty 0.5) on a lower-friction surface looked best; use as baseline for sim-to-real alignment.
- Hardware logs for Test 1-3 are copied to `logs/hardware_sessions/` for sim comparison.

## Sim-to-Real Alignment Notes (Active)
- Backlash dead zone is ~10 degrees (2026-01-03); treat older 30 degree references as historical only.
- Hardware telemetry logs now include `cmd_pos_*` columns for commanded vs measured comparison.
- Scripted/CPG sim still shows very low vx even with higher stiffness or amplitude scaling; see archives for 2026-01-02 analysis.
- Sim CPG leg trajectory now matches hardware generator math via the shared kernel in `common/cpg_math.py`.
- Best actuator tracking match (effort=2.8) is stiffness=1200, damping=75; sim calf tracking now ~0.078 rad (matches hardware).
- CPG mode is now open-loop (policy ignored); observation size remains 48D.
- Sim validation logs are gitignored under `deployment/validation/sim_logs/`.

## Evergreen Guardrails
- Video is the primary signal for success/failure. Metrics can lie (vx from falling, ep_len from standing). Run video review after every experiment.
- Use the 5-metric protocol (episode_length, upright_mean, height_reward, body_contact_penalty, vx_w_mean) as the quantitative complement to video, not a replacement.
- "On elbows" exploit remains a risk; always check height_reward and body_contact_penalty.

## Hardware Session Logs
- RPi logs: `/home/pi/harold/deployment/sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- Desktop copies: `logs/hardware_sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- Logged at 5 Hz during 20 Hz control; includes cmd_pos, measured positions, currents, temps, and system stats.

## 2026-03-19: Scoring Consolidation & Session 48 Findings

### walk_score was blind — 200+ experiments optimized without signal
- `compute_walk_score()` had a hard gate: `if ep_len < 300: return 0.0`. No experiment ever reached ep_len=300 (best was ~195), so walk_score was 0 for every experiment ever run.
- `compute_progress_score()` (soft gate via `tanh(ep_len/150)`) was computed and logged but never used for decisions. The system was optimizing blind.
- Consolidated to single `compute_score()` using the soft-gate formula. Baseline score: ~37.4 (ep_len=174, vx=0.057, upright=0.906).
- Results.tsv migrated from `walk_score`/`progress_score` columns to single `score` column.

### Session 48 key findings
- **alive_bonus=2.0 is promising**: EXP-474 showed improved ep_len but front legs still passive. Worth combining with other changes.
- **forward_motion_weight increase causes tipping**: EXP-473 (forward_motion_weight:5.0->8.0) caused robot to tip forward aggressively.
- **undesired_contacts_threshold=0.5 + alive_bonus=2.0**: EXP-476 showed front legs shifting but not stepping. Contact penalty tightening alone doesn't break the passive-front-legs equilibrium.
- **elbow_pose_termination + alive_bonus=2.0**: EXP-475 showed standing with minimal movement. Joint-angle termination too aggressive for current policy.
- **episode_length_s=15**: EXP-477/478 tested shorter episodes. Results pending.

### Parameter drift found
5 parameter values in program.md and PARAMETER_REGISTRY.md had drifted from actual config:
- `ang_vel_xy_weight`: documented -0.0001, actual -0.01
- `upright_weight`: documented 2.0, actual 3.0
- `forward_motion_weight`: documented 3.0, actual 5.0
- `orientation_threshold`: documented -0.5, actual -0.6
- `action_filter_beta`: documented 0.40, actual 0.2

All corrected to match `harold_isaac_lab_env_cfg.py` (source of truth).

## Archives
Historical observations are moved to `docs/memory/archives/index.md` to keep this file short.
