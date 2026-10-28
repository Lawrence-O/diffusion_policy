# Task043 non-broad 600k RGB student

`agent.yaml` is the original saved configuration for
`task043_three_camera_relu_weighted_fixed50_32env_600k_seed42_continuation_v4`.
Training checkpoint SHA256:
`dce4c6007ec39f500de15fb0e6bd7533c472a1b6828ae8fc7d24cb22d5d7e75f`.
Historical teacher/pretraining paths in this file are provenance only; they
are not accessed by the student-only deployment loader.

With a trusted original training checkpoint, use the existing robot evaluator:

```bash
python eval_real_robot.py --input /path/to/model_update_600000.pt \
  --agent_config deployment/task043_nonbroad_600k/agent.yaml \
  --policy_format uwlab_checkpoint --output ./recordings/student600k \
  --robot_ip YOUR_ROBOT_IP --controller_profile stage1_soft \
  --no-automatic_gripper_release --frequency 10 --steps_per_inference 1 \
  --action_noise 0 --save_video
```

This command connects hardware. Validate camera mapping, calibration,
observation conventions and safety before enabling policy control.
No teacher JIT, encoder-pretraining file or optimizer restoration is required.
The loader uses the matching RSL-RL ResNet18Encoder, MLP and normalization
utilities: install our compatible RSL-RL code and its torchvision dependencies.
Only the trained student inference modules are built; noise/auxiliary/teacher
checkpoint fields are not used for deterministic mean inference. Required
student weights are loaded strictly. Unsupported policy/camera interfaces fail.

These settings are for this checkpoint, not universal defaults for every run.
Use each other run's own saved agent configuration. This does not load recurrent
state PPO policies or diffusion-model checkpoints.
