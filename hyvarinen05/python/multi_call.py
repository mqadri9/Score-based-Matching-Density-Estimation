import subprocess
import sys

n_iter = 8000

for i in range(n_iter):
    subprocess.call(['python', "convergence_vs_frechetDist.py"], stdout=sys.stdout, stderr=subprocess.STDOUT)
    print("n_iter = {}".format(i))