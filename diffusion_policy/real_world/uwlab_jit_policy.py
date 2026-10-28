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


class UWLabCheckpointPolicy(UWLabJITPolicy):
    """Load only the trained RGB student's inference modules; never load a teacher."""
    def __init__(self, checkpoint, device='cpu', agent_config=None, teacher_jit=None):
        torch.nn.Module.__init__(self)
        import yaml
        checkpoint = Path(checkpoint)
        config_path = Path(agent_config) if agent_config else checkpoint.parent / 'params/agent.yaml'
        config = yaml.safe_load(config_path.read_text())
        options = dict(config['policy'])
        if options.pop('class_name') != 'StudentTeacherVision':
            raise ValueError('Only Task043 StudentTeacherVision RGB checkpoints are supported')
        if options['vision_groups'] != list(VISION_GROUPS):
            raise ValueError('Checkpoint camera order does not match the deployment interface')
        if options.get('encoder_type') != 'resnet18' or config['obs_groups']['policy'] != ['policy']:
            raise ValueError('Student-only loader supports the single-frame Task043 ResNet18 interface')
        # torch.load may execute pickle code: only use trusted training files.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = _StudentOnlyMean(options)
        state = saved['model_state_dict']
        expected = self.model.state_dict()
        allowed_unused = ('teacher.', 'teacher_obs_normalizer.', 'std_head.', 'aux_heads.')
        unknown = [k for k in state if k not in expected and k not in ('std', 'log_std')
                   and not k.startswith(allowed_unused)]
        if unknown:
            raise ValueError(f'Unsupported checkpoint fields: {unknown}')
        self.model.load_state_dict({k: v for k, v in state.items() if k in expected}, strict=True)
        self.model.to(device).eval()
        self.eval()


class _StudentOnlyMean(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        from rsl_rl.modules.student_teacher_vision import ResNet18Encoder
        from rsl_rl.networks import MLP, EmpiricalNormalization
        embed = options.get('embed_dim', 128)
        self.encoders = torch.nn.ModuleDict({key: ResNet18Encoder(
            3, embed_dim=embed, pretrained_path='',
            projection_activation=options.get('projection_activation', 'relu')) for key in VISION_GROUPS})
        # Preserve the training state's alias keys so strict loading checks them too.
        self.depth_encoder = self.encoders[VISION_GROUPS[0]]
        self.student_obs_normalizer = (EmpiricalNormalization(19)
            if options.get('student_obs_normalization', True) else torch.nn.Identity())
        self.student = MLP(19 + 3 * embed, 7, options['student_hidden_dims'], options['activation'])
        self.squash_student = options.get('squash_student', False)

    def forward(self, proprio, images):
        features = [self.student_obs_normalizer(proprio)]
        features.extend(self.encoders[key](image) for key, image in zip(VISION_GROUPS, images))
        action = self.student(torch.cat(features, dim=-1))
        return torch.tanh(action) if self.squash_student else action
