"""Plot current verl training/evaluation metrics without loading full bank evidence."""
import argparse
import csv
import html
import json
import math
from pathlib import Path


def read_lines(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def analyze(root):
    root=Path(root)
    groups=read_lines(root/'training_metrics.jsonl')
    epochs=[]
    for path in sorted((root/'evaluation').glob('epoch_*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        for mode in ('roll_forward','answer'):
            if mode in data:
                epochs.append(dict(epoch=data['epoch'],mode=mode,**data[mode]['summary']))
    epochs.sort(key=lambda r:r['epoch'])
    return dict(groups=groups,epochs=epochs)


def flatten(row,prefix=''):
    result={}
    for key,value in row.items():
        name=prefix+key
        if isinstance(value,dict): result.update(flatten(value,name+'.'))
        elif isinstance(value,(str,int,float,bool)) or value is None: result[name]=value
    return result


def export_csv(path,rows):
    rows=[flatten(r) for r in rows]
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render(root,out=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    target=Path(out) if out else root/'plots'
    target.mkdir(parents=True,exist_ok=True)
    data=analyze(root); groups,epochs=data['groups'],data['epochs']
    export_csv(target/'training_metrics.csv',groups)
    export_csv(target/'validation_metrics.csv',epochs)
    fig,axes=plt.subplots(4,3,figsize=(17,17),constrained_layout=True)
    x=list(range(1,len(groups)+1))
    epoch_x={r['epoch']:i for i,r in enumerate(groups,1)}
    def line(ax,label,values,positions=None):
        positions=x if positions is None else positions
        selected=[(a,b) for a,b in zip(positions,values) if isinstance(b,(int,float)) and math.isfinite(b)]
        if selected: ax.plot([a for a,b in selected],[b for a,b in selected],marker='.',label=label)
    a=axes[0,0]; a.set_title('Training reward curves (token F1)')
    line(a,'Train mean',[r.get('reward') for r in groups])
    for index,r in enumerate(groups,1):
        a.scatter([index]*len(r.get('trajectory_rewards',[])),r.get('trajectory_rewards',[]),s=10,alpha=.3,color='steelblue')
    line(a,'Validation F1',[r.get('f1') for r in epochs],[epoch_x.get(r['epoch'],0) for r in epochs])
    a=axes[0,1]; a.set_title('Validation EM / F1 / BLEU-1')
    for key in ('em','f1','bleu1','llm_judge'):
        line(a,key,[r.get(key) for r in epochs],[epoch_x.get(r['epoch'],0) for r in epochs])
    a=axes[0,2]; a.set_title('Training QA metrics')
    for key in ('em','f1','bleu1'): line(a,key,[r.get(key) for r in groups])
    a=axes[1,0]; a.set_title('Final advantage and informative QA fraction')
    for key in ('advantage_abs_mean','signal_ratio'): line(a,key,[r.get(key) for r in groups])
    a=axes[1,1]; a.set_title('KL / policy loss / clipping')
    for key in ('actor/kl_loss','actor/pg_loss','actor/pg_clipfrac','actor/ppo_kl'):
        line(a,key,[r.get('metrics',{}).get(key) for r in groups])
    a=axes[1,2]; a.set_title('Policy entropy before update (nats)')
    line(a,'Entropy',[r.get('entropy') for r in groups])
    a=axes[2,0]; a.set_title('Gradient norm before clipping')
    line(a,'Grad norm',[r.get('metrics',{}).get('actor/grad_norm') for r in groups])
    a.axhline(1.,color='gray',ls='--',label='Configured clip norm')
    a=axes[2,1]; a.set_title('Learning rate used by optimizer')
    line(a,'LR',[r.get('metrics',{}).get('actor/lr') for r in groups])
    a.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    a=axes[2,2]; a.set_title('Updates and skips')
    line(a,'Optimizer steps',[r.get('global_step') for r in groups])
    line(a,'Skipped group',[int(r.get('skipped_update',False)) for r in groups])
    a=axes[3,0]; a.set_title('Generation and bank errors')
    line(a,'Invalid turn fraction',[r.get('invalid_turns',0)/r['turns'] if r.get('turns') else None for r in groups])
    line(a,'Syntax repair fraction',[r.get('repaired_turns',0)/r['turns'] if r.get('turns') else None for r in groups])
    for key,label in [('action_repaired_turns','Action repair fraction'),
                      ('partial_applied_turns','Partial execution fraction'),
                      ('parse_error_turns','Parse error fraction')]:
        line(a,label,[r[key]/r['turns'] if key in r and r.get('turns') else None for r in groups])
    line(a,'Token limit fraction',[r.get('token_limit_count',0)/r['sampled_sequences'] if r.get('sampled_sequences') else None for r in groups])
    a=axes[3,1]; a.set_title('Output length')
    for key in ('output_tokens_mean','answer_output_tokens','answer_words'): line(a,key,[r.get(key) for r in groups])
    a=axes[3,2]; a.set_title('Group wall time (seconds)')
    for key in ('sampling_scoring_seconds','log_probability_seconds','update_seconds','total_seconds'):
        line(a,key,[r.get(key) for r in groups])
    for a in axes.flat:
        a.set_xlabel('Sampled group (includes skipped groups)')
        a.grid(alpha=.2)
        if a.get_legend_handles_labels()[0]: a.legend(fontsize=8)
    fig.savefig(target/'training_curves.png',dpi=160)
    plt.close(fig)
    # Block credit is plotted separately, retaining sign and actual update group.
    credit=[dict(epoch=r['epoch'],group=r['group'],**b) for r in groups for b in r.get('credit_blocks',[])]
    export_csv(target/'block_credit.csv',credit)
    if credit:
        fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
        for r in groups:
            for tid in sorted({b['trajectory_id'] for b in r.get('credit_blocks',[])}):
                selected=[b for b in r['credit_blocks'] if b['trajectory_id']==tid]
                ax.plot([b['end_turn'] for b in selected],[b['auxiliary_advantage'] for b in selected],
                        alpha=.7,label=f'E{r["epoch"]}/G{r["group"]}/T{tid}')
        ax.axhline(0,color='gray',lw=1); ax.set_xlabel('Processed dialogue turns')
        ax.set_ylabel('Auxiliary advantage'); ax.grid(alpha=.2)
        if len(groups)<5: ax.legend(fontsize=7)
        fig.savefig(target/'block_credit.png',dpi=160); plt.close(fig)
    body=f'<h1>Training reward and diagnostics</h1><p>{html.escape(str(root.resolve()))}</p>'
    body+='<p>Train uses sampled trajectories; validation uses greedy generation on held-out dialogues. Missing measurements stay blank. Skipped groups do not advance the optimizer.</p>'
    body+='<p><a href="training_metrics.csv">Training CSV</a> | <a href="validation_metrics.csv">Validation CSV</a> | <a href="block_credit.csv">Block credit CSV</a></p>'
    body+='<img style="max-width:100%" src="training_curves.png">'
    if credit: body+='<img style="max-width:100%" src="block_credit.png">'
    (target/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Training diagnostics</title>'+body,encoding='utf-8')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--out')
    args=p.parse_args()
    render(args.run,args.out)
    print(Path(args.out or Path(args.run)/'plots')/'report.html')

if __name__=='__main__': main()
