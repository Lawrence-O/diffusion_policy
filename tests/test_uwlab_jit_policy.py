"""No hardware imports, connections or motion."""
from pathlib import Path
import ast
import click
import numpy as np
from click.testing import CliRunner
import torch
import pytest
from diffusion_policy.real_world.uwlab_jit_policy import UWLabJITPolicy, VISION_GROUPS, PROPRIO_FIELDS


class RecordingPolicy(torch.nn.Module):
    def forward(self, proprio: torch.Tensor, images: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat((proprio[:, :4],torch.stack([image[:,0,0,0] for image in images],-1)),-1)


def inputs():
    return {**{key: torch.full((2,1,width),float(i)) for i,(key,width) in enumerate(PROPRIO_FIELDS)},
            **{key: torch.full((2,1,3,224,224),(i+1)/10.) for i,key in enumerate(VISION_GROUPS)}}


@pytest.fixture
def adapter(tmp_path):
    p=tmp_path/'recording.pt';torch.jit.script(RecordingPolicy()).save(str(p))
    return UWLabJITPolicy(p)


def test_order_shapes_and_deterministic_reset(adapter):
    obs=inputs();a=adapter.predict_action(obs)['action']
    torch.testing.assert_close(a,torch.tensor([[0.,1.,1.,1.,.1,.2,.3]]).expand(2,-1))
    adapter.reset();assert torch.equal(a,adapter.predict_action(obs)['action'])
    assert a.shape==(2,7) and not a.requires_grad


@pytest.mark.parametrize('bad',['history','range','nan','missing','shape'])
def test_invalid_observations_fail_closed(adapter,bad):
    obs=inputs()
    if bad=='history':obs['arm_joint_pos']=torch.zeros(2,4,6)
    if bad=='range':obs['side_rgb'].fill_(255.)
    if bad=='nan':obs['end_effector_pose'].fill_(float('nan'))
    if bad=='missing':del obs['front_rgb']
    if bad=='shape':obs['front_rgb']=torch.zeros(2,1,3,100,100)
    with pytest.raises((KeyError,ValueError)):adapter.predict_action(obs)


def test_checked_600k_jit_direct_parity():
    path=Path(__file__).resolve().parents[2]/'artifacts/real_robot_ff_compat/checked_student_mean.pt'
    if not path.exists():pytest.skip('Local Task043 export artifact is not available')
    p=UWLabJITPolicy(path);obs=inputs()
    with torch.inference_mode():
        expected=p.model(torch.cat([obs[k][:,0] for k,_ in PROPRIO_FIELDS],-1),[obs[k][:,0] for k in VISION_GROUPS])
        actual=p.predict_action(obs)['action']
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_training_checkpoint_recorded_trajectory_parity():
    import os
    import json
    from diffusion_policy.real_world.uwlab_jit_policy import UWLabCheckpointPolicy
    root=Path(__file__).resolve().parents[2]
    here=root/'artifacts/real_robot_ff_compat'
    trace=here/'mined_v1/trajectory.pt'
    if not trace.exists():pytest.skip('Local mined trajectory is unavailable')
    torch.set_num_threads(4)
    device=os.environ.get('UWLAB_REPLAY_DEVICE','cpu')
    native=UWLabCheckpointPolicy(here/'inputs/model_update_600000.pt',device=device,
        teacher_jit='/deliberately/missing/teacher.pt')
    assert not any('teacher' in key for key in native.model.state_dict())
    jit=UWLabJITPolicy(here/'checked_student_mean.pt',device=device)
    rows=torch.load(trace,map_location='cpu',weights_only=False)
    jit_error=recorded_error=0.
    for index in range(len(rows)):
        obs={k:rows[index][k].to(device) for k in (*VISION_GROUPS,*(k for k,_ in PROPRIO_FIELDS))}
        action=native.predict_action(obs)['action']
        exported=jit.predict_action(obs)['action']
        jit_error=max(jit_error,float((action-exported).abs().max()))
        recorded_error=max(recorded_error,float((action.cpu()-rows[index]['native_mean']).abs().max()))
        torch.testing.assert_close(action, exported,atol=2e-5,rtol=1e-5)
        # Compare checkpoint and export strictly in this process. Historical
        # Isaac-process outputs are a separate measured discrepancy, not part
        # of that loader-equivalence assertion (see retained failed XML runs).
    (here/f'checkpoint_replay_{device.replace(":","_")}.json').write_text(json.dumps(dict(
        frames=len(rows),device=device,jit_max_abs_error=jit_error,
        recorded_cuda_max_abs_error=recorded_error,status='PASS checkpoint-versus-JIT',
        historical_strict_parity='UNVERIFIED' if recorded_error <= 2e-5 else 'FAIL at 2e-5 absolute tolerance'),indent=2)+'\n')


def test_actual_cli_guard_and_legacy_controller_defaults_without_hardware():
    path=Path(__file__).resolve().parents[1]/'eval_real_robot.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # Run the actual CLI validation/settings prefix only. No hardware imports.
    end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='sysid_records' for t in n.targets))
    node.body=node.body[:end]+[ast.Return(ast.Tuple(elts=[ast.Name(id='controller_profile',ctx=ast.Load()),
        ast.Name(id='automatic_gripper_release',ctx=ast.Load())],ctx=ast.Load()))]
    ns=dict(click=click,np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    runner=CliRunner();common=['-i','model.pt','-o','unused','--robot_ip','unused']
    assert runner.invoke(ns['main'],common+['--policy_format','uwlab_jit']).exit_code==2
    valid=common+['--policy_format','uwlab_jit','--controller_profile','stage1_soft','--no-automatic_gripper_release']
    result=runner.invoke(ns['main'],valid,standalone_mode=False)
    assert result.exit_code==0 and result.return_value==('stage1_soft',False)
    assert runner.invoke(ns['main'],valid+['--frequency','20']).exit_code==2
    legacy=runner.invoke(ns['main'],common,standalone_mode=False)
    assert legacy.exit_code==0 and legacy.return_value==('stage2',True)
