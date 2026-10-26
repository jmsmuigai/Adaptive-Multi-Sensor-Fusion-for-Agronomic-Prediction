"""jobrun.py -- resumable runner: python3 jobrun.py <module> [workers] [model_filter]
Each job's result is written to out/exp/<module>/<key>.json; finished jobs are skipped."""
import importlib, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
mod = importlib.import_module(sys.argv[1])
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
filt = sys.argv[3].split(',') if len(sys.argv) > 3 else None
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'exp', sys.argv[1])
os.makedirs(D, exist_ok=True)
key = lambda j: '__'.join(str(x) for x in j).replace(' ', '_').replace('/', '-').replace(',', '').replace('(', '').replace(')', '')
def run(j):
    p = os.path.join(D, key(j) + '.json')
    if os.path.exists(p):
        return 0
    r = mod.job(j)
    json.dump(r, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
    return 1
todo = [j for j in mod.jobs() if not os.path.exists(os.path.join(D, key(j) + '.json'))]
if filt:
    todo = [j for j in todo if any(str(x) in filt for x in j)]
print(len(todo), 'jobs', flush=True)
t0 = time.time()
with Pool(W) as p:
    n = sum(p.imap_unordered(run, todo, chunksize=1))
print('done', n, round(time.time() - t0), flush=True)
