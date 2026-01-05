# Harold Observations & Insights

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
- Use the 5-metric protocol (episode_length, upright_mean, height_reward, body_contact_penalty, vx_w_mean).
- Do not rely on video for success/failure; base conclusions on metrics only.
- "On elbows" exploit remains a risk; always check height_reward and body_contact_penalty.

## Hardware Session Logs
- RPi logs: `/home/pi/harold/deployment/sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- Desktop copies: `logs/hardware_sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- Logged at 5 Hz during 20 Hz control; includes cmd_pos, measured positions, currents, temps, and system stats.

## Archives
Historical observations are moved to `docs/memory/archives/index.md` to keep this file short.
