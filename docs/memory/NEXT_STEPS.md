# Harold Next Steps

1. Execute Phase 1 of `PLAN.md`: fix flat-task frame alignment, forward-fall reward leakage, EMA reset behavior, policy-log scalar handling, and termination telemetry before trusting new training results.
2. Execute Phase 2 of `PLAN.md`: make rough-task domain randomization real or disable misleading defaults, and restore intended terrain-level coverage/curriculum behavior.
3. Execute Phase 3 of `PLAN.md`: migrate exporter, validation scripts, and checked-in policy metadata fully to 48D and regenerate deployment artifacts from a current checkpoint.
4. Run the verification matrix in `PLAN.md` after each phase, including `py_compile`, targeted pytest, and short `python scripts/harold.py` smoke runs with video enabled.
5. Resume the hardware-alignment workflow only after the audit fixes above are complete and verified.
6. Run a real world hardware test with the robot off of the test stand under its own weight to verify all the changes landed correctly and did not result in regressions to the walking pattern. (Ensure the IMU is recording data)
7. Take the logs from the real world hardware test and copy them from the robot's raspberry pi to the Desktop computer
8. Run the simulated walking and compare the commands from the real robot to the simulated robot. If they do not match, update the simulated robot settings (stiffness and dampening) until the sim matches the real hardware exactly.
9. Run the simulated walking and compare the actual joint positions from the real robot to the simulated robot. If they do not match, update the simulated robot settings (stiffness and dampening) until the sim matches the real hardware within a realistic margin of error.
10. Run the simulated walking and compare the actual IMU data from the real robot's IMU data. If they do not match, attempt to add noise to simulated IMU data until it resonably matches what we see in the real world. If the match is impossible, consider ways we can clean or get better data from the real robot. Work with me to plan out a strategy.
11. If the desktop environment path or Isaac Sim launch workflow changes, update the environment/runtime guidance in `AGENTS.md`, `docs/index.md`, `docs/overview.md`, and `docs/sim/isaac_lab_extension.md` immediately so future agents do not misdiagnose `omni` import failures as missing packages.
