"""Sequential resumable execution; stops immediately on any failed stage."""
import datetime
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
logdir=HERE/'run_v2'/'logs';logdir.mkdir(parents=True,exist_ok=True)
commands=[['cache','--split','train'],['cache','--split','valid'],['train'],['cache','--split','test'],['evaluate'],['xai'],['demo']]
for command in commands:
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    log=logdir/(stamp+'_'+'_'.join(command)+'.log')
    print('START',command,'LOG',log,flush=True)
    with log.open('x',encoding='utf-8') as out:
        proc=subprocess.run([sys.executable,'-B',str(HERE/'pipeline.py'),*command],cwd=HERE.parent,stdout=out,stderr=subprocess.STDOUT)
    print('END',command,'EXIT',proc.returncode,flush=True)
    if proc.returncode:sys.exit(proc.returncode)
