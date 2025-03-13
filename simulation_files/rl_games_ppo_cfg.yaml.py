params:
  seed: 42

  # environment wrapper clipping
  env:
    clip_observations: 5.0
    clip_actions: 1.0

  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: False
    space:
      continuous:
        mu_activation: None
        sigma_activation: None

        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: False  #True
    mlp:
      units: [512, 256, 256]
      activation: elu
      d2rl: False

      initializer:
        name: default
      regularizer:
        name: l2
        weight: 1e-5


  load_checkpoint: False # flag which sets whether to load the checkpoint
  load_path: '' # path to the checkpoint to load

  config:
    name: harold_direct
    env_name: rlgpu
    device: 'cuda:0'
    device_name: 'cuda:0'
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: True
    num_actors: -1  # configured from the script (based on num_envs)
    reward_shaper:
      scale_value: 0.1
    normalize_advantage: True
    gamma: 0.99
    tau : 0.95
    learning_rate: 3e-4
    lr_schedule: adaptive
    kl_threshold: 0.008
    score_to_win: 20000
    max_epochs: 2000
    save_best_after: 50
    save_frequency: 25
    grad_norm: 1.0
    entropy_coef: 0.01 #0.015 (2)
    truncate_grads: True
    e_clip: 0.2
    horizon_length: 128
    minibatch_size: 8192
    mini_epochs: 8 #15 (1)
    critic_coef: 1
    clip_value: True
    seq_length: 4
    bounds_loss_coef: 0.0001

