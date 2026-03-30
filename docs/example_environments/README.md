# Isaac Lab Example Environments

Technical breakdowns of every quadruped locomotion environment in Isaac Lab. Each document covers the robot specs, observation/action spaces, full reward structure with weights, command ranges, domain randomization, training config, and relevance to Harold.

These are reference implementations that Harold's training is based on. Use them to compare reward terms, thresholds, and scaling when debugging or tuning Harold's config.

## Manager-Based Environments

| Robot | Terrain | File | Gym ID |
|-------|---------|------|--------|
| Boston Dynamics Spot | Flat | [spot_flat.md](spot_flat.md) | `Isaac-Velocity-Flat-Spot-v0` |
| Unitree Go2 | Flat | [go2_flat.md](go2_flat.md) | `Isaac-Velocity-Flat-Unitree-Go2-v0` |
| Unitree Go2 | Rough | [go2_rough.md](go2_rough.md) | `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| Unitree Go1 | Flat | [go1_flat.md](go1_flat.md) | `Isaac-Velocity-Flat-Unitree-Go1-v0` |
| Unitree Go1 | Rough | [go1_rough.md](go1_rough.md) | `Isaac-Velocity-Rough-Unitree-Go1-v0` |
| Unitree A1 | Flat | [a1_flat.md](a1_flat.md) | `Isaac-Velocity-Flat-Unitree-A1-v0` |
| ANYbotics ANYmal D | Flat | [anymal_d_flat.md](anymal_d_flat.md) | `Isaac-Velocity-Flat-Anymal-D-v0` |
| ANYbotics ANYmal D | Rough | [anymal_d_rough.md](anymal_d_rough.md) | `Isaac-Velocity-Rough-Anymal-D-v0` |
| ANYbotics ANYmal B | Flat | [anymal_b_flat.md](anymal_b_flat.md) | `Isaac-Velocity-Flat-Anymal-B-v0` |
| ANYbotics ANYmal B | Rough | [anymal_b_rough.md](anymal_b_rough.md) | `Isaac-Velocity-Rough-Anymal-B-v0` |

## Direct Environments

| Robot | Terrain | File | Gym ID |
|-------|---------|------|--------|
| ANYbotics ANYmal C | Flat/Rough | [anymal_c_direct.md](anymal_c_direct.md) | `Isaac-Velocity-Flat-Anymal-C-Direct-v0` |

**ANYmal C Direct is the most relevant reference for Harold** — it uses the same direct environment architecture (not manager-based) and reveals important porting considerations like manual `step_dt` multiplication on rewards.

## Robot Size Comparison

| Robot | Mass | Leg Length | DOF | Notes |
|-------|------|-----------|-----|-------|
| **Harold** | **2 kg** | **~0.15 m** | **12** | **Our robot** |
| Unitree A1 | ~12 kg | ~0.25 m | 12 | Closest in size |
| Unitree Go1 | ~12 kg | ~0.25 m | 12 | Similar to A1 |
| Unitree Go2 | ~15 kg | ~0.3 m | 12 | Mid-size |
| BD Spot | ~32 kg | ~0.5 m | 12 | Our reward source |
| ANYmal B | ~30 kg | ~0.4 m | 12 | Large |
| ANYmal C | ~50 kg | ~0.55 m | 12 | Large, direct env |
| ANYmal D | ~50 kg | ~0.55 m | 12 | Large |

Harold is significantly smaller than all reference robots. Key scaling considerations:
- **Velocity commands**: Scale down proportionally to leg length (Froude scaling)
- **Velocity thresholds**: Must be lowered for Harold's achievable speeds
- **Air time / gait cadence**: Smaller animals step faster
- **Foot clearance**: Scale to ~20% of leg length
- **Domain randomization**: Mass perturbations, push forces, etc. must be proportional to body mass
- **Reward weights**: Torque and force penalties may need rebalancing for smaller actuators
