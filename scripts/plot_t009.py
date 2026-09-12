"""Fixed first-ID diagnostic plots, generated offline from persisted numbers."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
config=json.loads((args.audit/'config.json').read_text())
first=config['representative_image_id'];out=args.audit/'figures';out.mkdir(exist_ok=True)
surfaces=json.loads((args.audit/'surfaces.json').read_text())
records=json.loads((args.audit/'trajectories.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for condition in ('homogeneous_dark','homogeneous_bright'):
    item=next(r for r in surfaces if r['image_id']==first and r['condition']==condition)
    ev=config['surface_ev'];gamma=config['surface_gamma']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,label in zip(axes,('semantic_loss','mse'),('Frozen semantic loss','Offline reference MSE')):
        z=np.array([r[key] for r in item['rows']]).reshape(len(ev),len(gamma)).T
        mesh=ax.pcolormesh(ev,gamma,z,shading='nearest',cmap='viridis');fig.colorbar(mesh,ax=ax)
        for name,marker,color in (('semantic_argmin','x','red'),('mse_argmin','o','white')):
            best=item['summary'][name];ax.scatter(best['ev'],best['gamma'],marker=marker,color=color,s=65,label=name)
        ax.set(xlabel='Exposure EV',ylabel='Gamma',title=label);ax.legend(fontsize=8)
    fig.suptitle(f'T009 development ID {first}: {condition} (fixed grid)')
    fig.savefig(out/(condition+'_surface.png'),dpi=180);fig.savefig(out/(condition+'_surface.pdf'));plt.close(fig)
for condition in ('homogeneous_dark','homogeneous_bright','left_right','quadrants'):
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,size in enumerate((1,2)):
        for coordinates,color in (('ev_only','#1976d2'),('gamma_only','#e67e22'),('ev_gamma','#338855')):
            r=next(r for r in records if r['image_id']==first and r['condition']==condition and r['size']==size and r['coordinates']==coordinates)
            steps=[t['step'] for t in r['trace']]
            for ax,key in zip(axes[row],('semantic_loss','mse')):
                ax.plot(steps,[t[key] for t in r['trace']],label=coordinates,color=color,marker='.' if len(steps)==1 else None)
        for col,ax in enumerate(axes[row]):
            ax.set(xlabel='Completed Adam updates',ylabel='Semantic loss' if col==0 else 'Offline reference MSE',
                   title=('Global' if size==1 else 'Spatial 2x2')+(' / semantic' if col==0 else ' / reference'))
            ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'T009 development ID {first}: {condition}; oracle MSE never stops TTT')
    fig.savefig(out/(condition+'_trajectory.png'),dpi=180);fig.savefig(out/(condition+'_trajectory.pdf'));plt.close(fig)
print('Saved six fixed first-ID PNG/PDF plots for',first)
