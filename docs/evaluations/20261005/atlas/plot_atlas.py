"""Reproduce the public figures from aggregate atlas.json; no network or model needed."""
from pathlib import Path
import json, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'atlas.json').read_text(encoding='utf-8'))
MODELS={r['id']:r for r in D['models']}
TEAL='#007f86';BLUE='#395a9e';ORANGE='#c56a23';PURPLE='#8667a7';GRAY='#657580'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#283a47','text.color':'#283a47','axes.titleweight':'bold','svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig,name):
 for ext in ['png','svg']:
  p=ROOT/(name+'.'+ext)
  fig.savefig(p,dpi=170,bbox_inches='tight',**({'metadata':{'Date':None}} if ext=='svg' else {}))
  if ext=='svg':p.write_bytes(('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
 plt.close(fig)

def matched():
 fig,axes=plt.subplots(2,2,figsize=(12,8));fig.suptitle('MiDM: nominal base parameters vs matched decision accuracy',fontsize=17,y=.98)
 for ax,(key,title) in zip(axes.flat,[('td_test','Typed Decisions test · N=2000'),('kevT_test','Kev transfer test · N=764'),('jb_public','JevBench public · N=231'),('jb_hard','JevBench hard · N=111')]):
  y=[100*MODELS[m]['metrics'][key]['accuracy'] for m in ['midm_4b','midm_9b']]
  ax.plot([4,9],y,color=TEAL,linewidth=1.5,linestyle='--',zorder=2);ax.scatter([4,9],y,c=[BLUE,TEAL],s=95,zorder=3)
  for x,v in zip([4,9],y):ax.annotate(f'{v:.2f}%',(x,v),xytext=(0,11),textcoords='offset points',ha='center',weight='bold')
  ax.set(title=title,xlim=(2.7,10.3),ylim=(0,103),xticks=[4,9],xlabel='Nominal base parameters (billions)',ylabel='Hard-label accuracy (%)');ax.grid(axis='y',alpha=.16)
  ax.text(.5,.1,f'9B minus 4B: +{y[1]-y[0]:.2f} percentage points',ha='center',transform=ax.transAxes,fontsize=10)
 fig.subplots_adjust(left=.085,right=.97,bottom=.13,top=.89,wspace=.24,hspace=.45)
 fig.text(.08,.035,'Same fixed items, NF4/BF16, RTX 3090, context 4096, TTA=1. Retrospective point estimates.\nJevBench public accuracy is not the official composite. Connecting lines are not fitted scaling laws.',fontsize=10,color=GRAY)
 save(fig,'matched_size_performance')

def context():
 ids=['midm_q35_2','clm_style4','pointer_frozen4','midm_q3_4','midm_q3_8','midm_4b','midm_9b','kev-4b','kev-8b','qwen3_4_reason']
 label={'midm_q35_2':'MiDM Q3.5 2B','clm_style4':'CLM-style 4B','pointer_frozen4':'Frozen pointer 4B','midm_q3_4':'MiDM Q3 4B','midm_q3_8':'MiDM Q3 8B','midm_4b':'MiDM Q3.5 4B','midm_9b':'MiDM Q3.5 9B','kev-4b':'Kev 4B [published]','kev-8b':'Kev 8B [published]','qwen3_4_reason':'Qwen3 4B reasoning'}
 offsets=[{'midm_q35_2':(.8,58),'clm_style4':(4.3,36),'pointer_frozen4':(4.3,48),'midm_q3_4':(2.2,77),'midm_q3_8':(6.2,66),'midm_4b':(4.3,73),'midm_9b':(7.1,85),'kev-4b':(1,67),'kev-8b':(6,79),'qwen3_4_reason':(4.3,88)},
 {'midm_q35_2':(.8,49),'clm_style4':(2.4,23),'pointer_frozen4':(4.3,30),'midm_q3_4':(4.3,42),'midm_q3_8':(6.4,36),'midm_4b':(4.3,52),'midm_9b':(7,62),'kev-4b':(1,32),'kev-8b':(6.2,49),'qwen3_4_reason':(4.3,71)}]
 fig,axes=plt.subplots(1,2,figsize=(16,7));fig.suptitle('Parameter context: local measurements and published comparison rows',fontsize=17,y=.97)
 for j,(ax,key,title) in enumerate(zip(axes,['jb_public','jb_hard'],['Same public 231 items','Public hard tier · 111 items'])):
  for id in ids:
   r=MODELS[id];x=r['parameters_b_nominal'];y=r['metrics'][key]['accuracy']*100
   published=id.startswith('kev-');primary=id in ['midm_4b','midm_9b'];reason=id=='qwen3_4_reason'
   color=PURPLE if published else ORANGE if reason else TEAL if primary else GRAY
   marker='s' if published else 'D' if reason else 'o' if primary else '^'
   ax.scatter(x,y,s=75,marker=marker,c=color,zorder=3)
   ax.annotate(label[id],(x,y),xytext=offsets[j][id],textcoords='data',ha='left',fontsize=8.5,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.5})
  ax.set(title=title,xlabel='Nominal base parameters (billions)',ylabel='Hard-label accuracy (%)',xlim=(.5,10.7),ylim=(0,103),xticks=[2,4,8,9]);ax.grid(alpha=.12)
 fig.subplots_adjust(left=.06,right=.97,bottom=.2,top=.86,wspace=.20)
 fig.text(.06,.04,'Teal circles: matched MiDM 4B/9B. Gray triangles: historical local settings. Purple squares: benchmark-published Kev.\nOrange diamonds: local generative reasoning. Bases, precision, data, epochs and compute differ across these groups.\nUnknown-size Jev and Solar are excluded here, not assigned an estimated parameter count. No architecture-only or scaling-law claim.',fontsize=10,color=GRAY)
 save(fig,'context_size_performance')

def comparisons():
 rows=sorted(D['models'],key=lambda r:r['metrics']['jb_public']['accuracy']);labels=[r['name'] for r in rows]
 fig,axes=plt.subplots(1,2,figsize=(15,8.5),sharey=True);fig.suptitle('Public JevBench cohort: capability comparison, not equal compute',fontsize=17,y=.97)
 for ax,key,title in zip(axes,['jb_public','jb_hard'],['Public accuracy · 231 items','Hard accuracy · 111 items']):
  vals=[r['metrics'][key]['accuracy']*100 for r in rows]
  colors=[TEAL if r['id'] in ['midm_4b','midm_9b'] else PURPLE if 'published' in r['name'] else ORANGE if r['id'] in ['solar','jev_free','qwen3_4_reason'] else GRAY for r in rows]
  ax.barh(range(len(rows)),vals,color=colors,height=.66)
  for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.2f}%',va='center',fontsize=9)
  ax.set(title=title,xlim=(0,112),xlabel='Hard-label accuracy (%)',yticks=range(len(rows)),yticklabels=labels);ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
 axes[0].tick_params(axis='y',labelsize=9)
 fig.subplots_adjust(left=.25,right=.96,bottom=.15,top=.88,wspace=.15)
 fig.text(.05,.045,'Teal: matched MiDM. Gray: historical local. Purple: benchmark-published outcomes. Orange: local API/reasoning runs.\nAll rows cover the same 231 public items, but service versions, prompts, training exposure and computation differ.\nJev published and Jev Free measured are separate observations. Incomplete cohorts are excluded. Not the official composite.',fontsize=10,color=GRAY)
 save(fig,'public_comparison')

def architecture():
 fig,ax=plt.subplots(figsize=(16,7));ax.set(xlim=(0,16),ylim=(0,7));ax.axis('off');fig.suptitle('Architecture comparison: how a bounded decision is produced',fontsize=18,y=.96)
 lanes=[(5.3,'CLM reference 8B',PURPLE,['State and candidates\nencoded separately','Frozen Qwen3-8B\nlast-token embeddings','Two projection MLPs\nstate / action','Scaled cosine score\nrank / softmax']),
 (3.35,'MiDM 4B / 9B',TEAL,['State + question +\nordered option lines','Causal base in NF4\n+ rank-16 LoRA','Hidden state at\neach option end','Shared linear pointer\nsoftmax over options']),
 (1.4,'Generative reasoning',ORANGE,['State + question +\ncandidate prompt','Autoregressive decoder\nreasoning / answer tokens','Parse generated\nanswer or tool output','Typed choice\nwith parser checks'])]
 for y,title,color,boxes in lanes:
  ax.text(.15,y+.62,title,fontsize=12,weight='bold',color=color)
  for j,text in enumerate(boxes):
   x=.15+j*4.02;box=FancyBboxPatch((x,y-.45),3.55,.82,boxstyle='round,pad=0.08',edgecolor=color,facecolor=color+'12',linewidth=1.4);ax.add_patch(box);ax.text(x+1.775,y-.035,text,ha='center',va='center',fontsize=10.5)
   if j<3:ax.add_patch(FancyArrowPatch((x+3.65,y-.035),(x+3.9,y-.035),arrowstyle='-|>',mutation_scale=14,color=color))
  note={'CLM reference 8B':'Independent embedding reuse is possible; released task-specific heads are separate artifacts.','MiDM 4B / 9B':'One causal pass: later options see earlier options, not vice versa. This is not bidirectional attention.','Generative reasoning':'More decoding work; a higher accuracy does not imply equivalent latency, cost or parameter disclosure.'}[title]
  ax.text(.15,y-.8,note,color=GRAY,fontsize=9.5)
 fig.subplots_adjust(left=.025,right=.985,bottom=.03,top=.91);save(fig,'architecture')

def control():
 ids=['clm_style4','pointer_frozen4','midm_q3_4'];names=['Independent frozen\nprojection heads','Frozen causal\npointer head','Causal pointer\n+ QLoRA'];x=np.arange(3);fig,ax=plt.subplots(figsize=(11,6.5))
 for delta,key,name,color in [(-.18,'td_test','Typed Decisions test (2000)',BLUE),(.18,'kevT_test','Kev transfer test (764)',TEAL)]:
  vals=[100*MODELS[i]['metrics'][key]['accuracy'] for i in ids];bars=ax.bar(x+delta,vals,.34,label=name,color=color)
  ax.bar_label(bars,labels=[f'{v:.2f}%' for v in vals],padding=4,fontsize=11)
 ax.set(title='Historical Qwen3-4B controls: input interaction and adaptation',ylabel='Hard-label accuracy (%)',ylim=(0,104),xticks=x,xticklabels=names);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper left',frameon=False)
 fig.subplots_adjust(left=.08,right=.97,bottom=.24,top=.88)
 fig.text(.08,.045,'Same nominal base family/size and original task mix; training recipes and recorded precision are not fully matched.\nThis is a historical design control, not an architecture-only causal estimate and not the released CLM 8B.\nPrimary Qwen3.5 4B/9B comparisons are shown separately.',fontsize=10,color=GRAY)
 save(fig,'architecture_control')

if __name__=='__main__':
 for f in [matched,context,comparisons,architecture,control]:f()
 print('Wrote five figures as PNG and SVG from atlas.json.')
