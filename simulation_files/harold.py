# Isaac Lab Imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.sensors import ContactSensorCfg


# ROS COMMANDS
"""
Train model:
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Harold-Direct-v3 --num_envs 1024

Play back trained model:
(BUILT IN ISAAC LAB VERSION)
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Harold-Direct-v3 --num_envs 1 --use_last_checkpoint
(MY CUSTOM VERSION)
./isaaclab.sh -p source/standalone/environments/run_harold_v3.py --task Isaac-Harold-Direct-v3 --num_envs 1 --use_last_checkpoint
(MY CUSTOM VERSION WITH MODEL EXPORT Note: You may need to run this with sudo for it to get write priveleges)
./isaaclab.sh -p source/standalone/environments/run_harold_v3.py --task Isaac-Harold-Direct-v3 --num_envs 1 --use_last_checkpoint --export_policy


Play back trained model with ROS2 Bridge:
 
How to start ROS2 Bridge:
conda deactivate
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

How to listen to ROS2 Bridge joint states:
ros2 topic echo /joint_states

How to start Tensorboard:
python3 -m tensorboard.main --logdir /home/matteo/IsaacLab/logs/rl_games/harold_direct


Edit USD file:
usdedit Harold_V4_STABLE_V2.usd

Resource Monitors:
nvtop
system monitor


"""


# robot
HAROLD_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/matteo/Desktop/Harold_V5/harold_v5_11.usd",
        activate_contact_sensors=True,
        scale=(1, 1, 1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=2,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.20),
        rot=(1.0, 0.0, 0.0, 0.0), 
        joint_pos={
            'fl_shoulder_joint': 0.0,
            'fr_shoulder_joint': 0.0,
            'bl_shoulder_joint': 0.0,
            'br_shoulder_joint': 0.0,
            
            'fl_thigh_joint': 0.3,
            'fr_thigh_joint': 0.3,
            'bl_thigh_joint': 0.3,
            'br_thigh_joint': 0.3,

            'fl_knee_joint': -0.75,
            'fr_knee_joint': -0.75,
            'bl_knee_joint': -0.75,
            'br_knee_joint': -0.75,
        }
    ),

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.8, #0.92 is the quoted real world value
            stiffness=300.0, #200.0,
            damping=100.0,
        ),
    },
)

