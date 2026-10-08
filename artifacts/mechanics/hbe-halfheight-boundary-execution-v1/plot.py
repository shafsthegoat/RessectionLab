"""Plot saved numerical diagnostics; no solver or measured response access."""
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path.cwd());p.add_argument("--output",type=Path,required=True);args=p.parse_args()
base=args.root/"outputs/mechanics/hbe-01-03-halfheight-boundary-v1/experiment"
paths=[base/"comparison.json",base/"runs/compression-N24-P2-S60-reference/readout.json"]
raw=[x.read_bytes() for x in paths]
assert hashlib.sha256(raw[0]).hexdigest()=="01e0fdcba6a21ba6c28e5845fd32c2c61feef7e8ae223368eee7ed632c5d2302"
assert hashlib.sha256(raw[1]).hexdigest()=="89ed53c3012f63ebe2b441efad2107910dfffc9f06fc6770ab002917d1ffee96"
comp,readout=map(json.loads,raw);x=-np.array(readout["load_coordinate_m"])*1000
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"font.family":"DejaVu Sans"})
fig,axs=plt.subplots(2,1,figsize=(9,7.5),sharex=True)
fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.85,hspace=.2)
curves=[("I2:P2","P2 − I2 (equal cell count)","#00798c"),("P1:P2","P2 − P1 (plate refinement)","#df782a"),("P1:I2","I2 − P1 (interior refinement)","#6b7b3c"),("P2:P4","P4 − P2 (finer plate layer)","#765ba7")]
for key,label,color in curves:
 row=comp["comparisons"][key]
 axs[0].plot(x,np.array(row["signed_force_difference_N"])*1e6,label=label,color=color,lw=2.1)
 axs[1].plot(x,np.array(row["maximum_probe_difference_m_by_state"])*1e6,color=color,lw=2.1)
for ax in axs:ax.grid(axis="y",alpha=.2);ax.axhline(0,color="#8c939c",lw=.7);ax.set_xlim(0,x[-1])
axs[0].set_ylabel("Signed force difference (µN)");axs[0].legend(frameon=False,fontsize=9,loc="upper left",ncol=2)
axs[1].set_ylabel("Largest probe difference (µm)");axs[1].set_xlabel("Prescribed specimen shortening (mm)")
fig.text(.12,.95,"Plate versus interior refinement",fontsize=19,weight="bold")
fig.text(.12,.91,"Numerical predictions on HBE_01_03 specimen dimensions",fontsize=11)
fig.text(.12,.88,"Fixed radial grid and 60 load steps; μ = 1,000 Pa is a numerical scale.",fontsize=10,color="#525b66")
fig.text(.12,.09,"Primary P2 − I2: maximum |force difference| 55.31 µN; probe difference 0.657 µm.",fontsize=10)
fig.text(.12,.035,"These are differences between predictions, not errors against measurements.\nTotal spatial/step convergence and physical calibration remain unproved.",fontsize=10,color="#525b66")
args.output.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output/"comparison.png",dpi=180);fig.savefig(args.output/"comparison.pdf");plt.close(fig)
record={"meaning":"plot of saved numerical predictions only; no measured-response access","inputs":[{"path":str(q.relative_to(args.root)),"sha256":hashlib.sha256(b).hexdigest()} for q,b in zip(paths,raw)],"outputs":{n:hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in ["comparison.png","comparison.pdf"]}}
(args.output/"figure-provenance.json").write_text(json.dumps(record,indent=2)+"\n")
