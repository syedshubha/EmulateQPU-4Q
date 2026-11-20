"""Supported commands; planning and inspection perform no simulation."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from . import __version__
from .execution import DEFAULT_MAX_RETRIES
from .io import read_json,atomic_json,inspect_result


def emit(value):
    print(json.dumps(value,indent=2,allow_nan=False))


def doctor():
    return {'application':'EmulateQPU','version':__version__,'python':sys.version.split()[0],
            'optional':{x:importlib.util.find_spec(x) is not None for x in ('shapely','gmsh')},
            'executables':{x:shutil.which(x) for x in ('sbatch','squeue','sacct','palace','mpiexec')},
            'scope':'Finite circuit simulations; no numerical work performed by doctor',
            'capabilities':{'chain':['full-carrier coherent states','ideal projective measurement','effective QND classified/continuous records',
                                   'conditional physical pulses','local/full replacement reset','operation-boundary checkpoints'],
                            'cavity':['corrected pair/cavity joint density','grouped physical-flux bath','photon RWA','matched idles and ringdown'],
                            'execution':['serial independent tasks','bounded local processes','Slurm arrays (site profile required)']}}


def quickstart(output):
    from .contracts import example_request
    from .execution import run_study
    quiet=example_request();quiet['schedule']['drives'][0]['amplitude_GHz']=0.
    driven=example_request()
    study={'schema':'emulateqpu.study.v1','tasks':[{'id':'quiet','request':quiet},{'id':'driven','request':driven}]}
    collection=run_study(study,output,backend='serial')
    comparison={}
    for name in ('quiet','driven'):
        p=Path(output)/'tasks'/name/'attempt_0001'/'output'
        result=inspect_result(p)
        comparison[name]=result['summary']['bare_level_populations']['q0'][1]
    delta=comparison['driven']-comparison['quiet']
    record={'status':'PASS' if collection['complete'] and abs(delta)>1e-6 else 'FAIL',
            'q0_level_1_probability':comparison,'difference':delta,
            'scope':'Uncalibrated assumed small circuit; fresh multilevel propagation and ideal measurement'}
    atomic_json(Path(output)/'comparison.json',record);return record


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(prog='emulateqpu',description='Finite multilevel circuit experiments with explicit controls, budgets and provenance')
    parser.add_argument('--version',action='version',version=__version__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Inspect capabilities and optional dependencies without computation')
    ex=sub.add_parser('examples',help='List compact bundled editable examples');ex.add_argument('action',choices=['list'])
    init=sub.add_parser('init',help='Write an explicit fresh experiment request');init.add_argument('--example',default='small-pulse');init.add_argument('--output',type=Path,required=True)
    for name in ('validate','plan'):
        p=sub.add_parser(name,help='Validate without constructing a model' if name=='validate' else 'Estimate resources before any large allocation')
        p.add_argument('request',type=Path)
        if name=='plan':p.add_argument('--profile',type=Path);p.add_argument('--output',type=Path)
    run=sub.add_parser('run',help='Execute a request or independent study with a bounded worker')
    run.add_argument('request',type=Path)
    run.add_argument('--backend',choices=['serial','local-process'],help='Defaults to local-process with multiple workers')
    run.add_argument('--workers',type=int,default=1)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--max-retries',type=int,default=DEFAULT_MAX_RETRIES,help='Extra attempts allowed on resume (default: %(default)s)')
    q=sub.add_parser('quickstart',help='Run quiet and changed-pulse examples and compare measurements');q.add_argument('--output',type=Path,required=True)
    for name in ('inspect','resume','collect'):
        p=sub.add_parser(name,help={'inspect':'Verify and inspect saved results without propagation','resume':'Resume an incomplete run, preserving previous attempts','collect':'Verify every expected result and detect incomplete studies'}[name]);p.add_argument('path',type=Path)
    hpc=sub.add_parser('hpc',help='Render, explicitly submit, inspect and collect Slurm arrays');hs=hpc.add_subparsers(dest='hpc_command',required=True)
    hp=hs.add_parser('render');hp.add_argument('plan',type=Path);hp.add_argument('--profile',type=Path,required=True);hp.add_argument('--output',type=Path,required=True)
    for name in ('submit','status','collect'):
        p=hs.add_parser(name);p.add_argument('path',type=Path)
        if name=='submit':p.add_argument('--retry',action='store_true')
    field=sub.add_parser('field',help='Plan and run CPU Palace field jobs on Slurm');fs=field.add_subparsers(dest='field_command',required=True)
    fp=fs.add_parser('plan');fp.add_argument('config',type=Path);fp.add_argument('--profile',type=Path,required=True);fp.add_argument('--output',type=Path,required=True)
    fp.add_argument('--input',type=Path,action='append',default=[],help='Additional field input to fingerprint')
    fp.add_argument('--require-table',action='append',default=[],metavar='NAME.csv=MIN_ROWS',help='Expected output table and minimum rows')
    fr=fs.add_parser('render');fr.add_argument('plan',type=Path);fr.add_argument('--output',type=Path,required=True)
    for name in ('submit','status','collect'):
        fs.add_parser(name).add_argument('path',type=Path)
    args=parser.parse_args(argv)
    try:
        from .contracts import example_request,validate_request
        from .planner import plan_request
        from .execution import run_study,plan_study,collect_run,resume_run
        if args.command=='doctor':value=doctor()
        elif args.command=='examples':value={'examples':['small-pulse','feedback','four-qubit','six-data','cavity'],'calibration':'Editable assumed fixtures; uncalibrated'}
        elif args.command=='init':
            value=example_request(args.example)
            args.output.mkdir(parents=True,exist_ok=False)
            atomic_json(args.output/'request.json',value)
            value={'request':str(args.output/'request.json'),'status':'PASS'}
        elif args.command=='validate':
            r=read_json(args.request)
            if r.get('schema')=='emulateqpu.study.v1':plan_study(r)
            else:validate_request(r)
            value={'status':'PASS','numerical_work':False}
        elif args.command=='plan':
            r=read_json(args.request);profile=read_json(args.profile) if args.profile else None
            value=plan_study(r,profile=profile) if r.get('schema')=='emulateqpu.study.v1' else {**plan_request(r,profile),'request':r}
            if args.output:
                if args.output.exists():raise ValueError('Plan output already exists')
                atomic_json(args.output,value)
        elif args.command=='run':value=run_study(read_json(args.request),args.output,backend=args.backend,workers=args.workers,max_retries=args.max_retries)
        elif args.command=='quickstart':value=quickstart(args.output)
        elif args.command=='inspect':value=collect_run(args.path,write=False) if (args.path/'run.json').exists() else inspect_result(args.path)
        elif args.command=='collect':value=collect_run(args.path)
        elif args.command=='resume':
            if (args.path/'run.json').exists():value=resume_run(args.path)
            else:
                from .api import resume_request
                value=resume_request(args.path)
        elif args.command=='hpc':
            if sys.platform=='win32' and args.hpc_command!='collect':
                raise ValueError('Slurm commands require a POSIX host with the site scheduler tools')
            from . import hpc
            value=hpc.render(args.plan,args.profile,args.output) if args.hpc_command=='render' else (hpc.submit(args.path,retry=args.retry) if args.hpc_command=='submit' else getattr(hpc,args.hpc_command)(args.path))
        elif args.command=='field':
            if sys.platform=='win32' and args.field_command!='collect':
                raise ValueError('Palace field Slurm commands require POSIX and the site scheduler tools')
            from . import field_slurm
            if args.field_command=='plan':
                tables={}
                for item in args.require_table:
                    name,count=item.split('=',1)
                    if name in tables:
                        raise ValueError('Duplicate required field table: '+name)
                    tables[name]=int(count)
                if args.output.exists() or args.output.is_symlink():
                    raise ValueError('Field plan output already exists')
                value=field_slurm.plan(args.config,args.profile,inputs=args.input,required_outputs=tables)
                atomic_json(args.output,value)
            elif args.field_command=='render':
                value=field_slurm.render(args.plan,args.output)
            else:
                value=getattr(field_slurm,args.field_command)(args.path)
        else:raise ValueError('Unsupported command')
        emit(value)
        if value.get('status') in ('FAIL','INCOMPLETE') or value.get('complete') is False:return 2
        return 0
    except (ValueError,RuntimeError,KeyError,OSError,MemoryError) as e:
        print(json.dumps({'status':'FAIL','error':str(e)}),file=sys.stderr);return 2
