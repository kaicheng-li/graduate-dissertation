"""Formal continuous LoCoMo evaluation using the same BF16/vLLM bank rollout."""
import argparse
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timeline',required=True)
    p.add_argument('--manager-model',required=True)
    p.add_argument('--manager-adapter',required=True)
    p.add_argument('--answer-model',required=True)
    p.add_argument('--answer-adapter',required=True)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--output',required=True)
    args,extra=p.parse_known_args()
    devices=[v.strip() for v in args.gpus.split(',')]
    if len(devices)!=4 or len(set(devices))!=4:
        print('TEST_WARNING this configuration requests four distinct GPUs; using supplied devices', flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(devices)
    from verl_memory.train import main as backend
    backend(['--task','test','--timeline',args.timeline,'--manager-model',args.manager_model,
             '--manager-adapter',args.manager_adapter,'--answer-model',args.answer_model,
             '--answer-adapter',args.answer_adapter,'--output-dir',str(Path(args.output).with_suffix('.trace')),
             '--results-file',args.output,*extra])

if __name__=='__main__': main()
