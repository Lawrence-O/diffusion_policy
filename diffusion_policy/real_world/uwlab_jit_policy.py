"""Stateless adapter for the checked Task043 single-frame, three-camera student."""
import torch
from pathlib import Path


VISION_GROUPS = ('side_rgb', 'wrist_rgb', 'front_rgb')
PROPRIO_FIELDS = (('last_gripper_action', 1), ('last_arm_action', 6),
                 ('arm_joint_pos', 6), ('end_effector_pose', 6))
SHAPE_META = {'obs': {
    **{key: {'shape': [3, 224, 224], 'type': 'rgb'} for key in VISION_GROUPS},
    **{key: {'shape': [width], 'type': 'low_dim'} for key, width in PROPRIO_FIELDS},
}, 'action': {'shape': [7]}}


class UWLabJITPolicy(torch.nn.Module):
    """Keep the existing predict_action API; normalization stays inside the JIT."""

    def __init__(self, checkpoint, device='cpu'):
        super().__init__()
        self.model = torch.jit.load(str(checkpoint), map_location=device).eval()
        self.eval()

    def reset(self):
        pass

    @torch.inference_mode()
    def predict_action(self, obs_dict):
        batch = None
        proprio = []
        images = []
        for key, width in PROPRIO_FIELDS:
            value = obs_dict[key]
            if value.ndim != 3 or value.shape[1:] != (1, width):
                raise ValueError(f'{key} must be [B,1,{width}], got {tuple(value.shape)}')
            batch = value.shape[0] if batch is None else batch
            if value.shape[0] != batch or not torch.isfinite(value).all():
                raise ValueError(f'Invalid batch/nonfinite proprio: {key}')
            proprio.append(value[:, 0].to(dtype=torch.float32))
        for key in VISION_GROUPS:
            value = obs_dict[key]
            if value.shape != (batch, 1, 3, 224, 224):
                raise ValueError(f'{key} must be [B,1,3,224,224], got {tuple(value.shape)}')
            if (not value.is_floating_point() or not torch.isfinite(value).all()
                    or (value < 0).any() or (value > 1).any()):
                raise ValueError(f'{key} must contain finite RGB floats in[0,1]')
            images.append(value[:, 0].to(dtype=torch.float32))
        action = self.model(torch.cat(proprio, dim=-1), images)
        if action.shape != (batch, 7) or not torch.isfinite(action).all():
            raise ValueError('JIT must return finite[B,7] raw mean actions')
        return {'action': action}


class _NativeVisionMean(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, proprio, images):
        from tensordict import TensorDict
        obs = TensorDict({'policy': proprio,
                          **dict(zip(VISION_GROUPS, images))}, batch_size=[len(proprio)])
        return self.policy.act_inference(obs)


class UWLabCheckpointPolicy(UWLabJITPolicy):
    """Load trusted Task043 training checkpoints through their native model class.

    The teacher artifact is required by that class's constructor, but is never
    used to produce deployment actions. Optimizer state is not restored.
    """
    def __init__(self, checkpoint, device='cpu', agent_config=None, teacher_jit=None):
        torch.nn.Module.__init__(self)
        import yaml
        from tensordict import TensorDict
        from rsl_rl.modules.student_teacher_vision import StudentTeacherVision
        checkpoint = Path(checkpoint)
        config_path = Path(agent_config) if agent_config else checkpoint.parent / 'params/agent.yaml'
        config = yaml.safe_load(config_path.read_text())
        options = dict(config['policy'])
        if options.pop('class_name') != 'StudentTeacherVision':
            raise ValueError('Only Task043 StudentTeacherVision RGB checkpoints are supported')
        if options['vision_groups'] != list(VISION_GROUPS):
            raise ValueError('Checkpoint camera order does not match the deployment interface')
        teacher = Path(teacher_jit or options['teacher_jit_path'])
        if not teacher.is_file():
            raise ValueError('Supply the matching accessible --teacher_jit artifact for model construction')
        options.update(teacher_jit_path=str(teacher), encoder_pretrained_path='')
        obs = TensorDict({'policy': torch.zeros(1, 19), 'teacher': torch.zeros(1, 215),
                          **{k: torch.zeros(1, 3, 224, 224) for k in VISION_GROUPS}}, batch_size=[1])
        policy = StudentTeacherVision(obs, config['obs_groups'], 7, **options)
        # torch.load may execute pickle code: only use trusted training files.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        policy.load_state_dict(saved['model_state_dict'], strict=True)
        self.model = _NativeVisionMean(policy).to(device).eval()
        self.eval()
