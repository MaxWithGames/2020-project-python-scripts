import math 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import scipy.spatial.distance
from sklearn.metrics import pairwise_distances
from numba import njit, jit, cuda

SIZE = 8.5 / 2

POS = 0
VELOCITY = 1
ACC = 2

X = 0
Y = 1
Z = 2

def get_column(a, b):
    return 3 * a + b

@cuda.jit
def get_pairwise_matrix(p, count, res):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx + ty * bw

    for i in range(0, count):
        d1 = (p[idx][0] - p[i][0]) / SIZE
        d2 = (p[idx][1] - p[i][1]) / SIZE 
        d3 = (p[idx][2] - p[i][2]) / SIZE

        d1 = d1 - int(d1)
        d2 = d2 - int(d2)
        d3 = d3 - int(d3)
        res[idx][i] = ((d1*d1 + d2*d2 + d3*d3) ** (0.5)) * SIZE

files_path = 'data/'
files_format = '.xyz'

files = glob.glob(files_path + "*" + files_format)
files = [(f.split(files_path)[1]).split(files_format)[0] for f in files]

figure_full_acc = make_subplots(rows=len(files), cols=1)
figure_axis_acc = make_subplots(rows=len(files), cols=1)

R_MIN = 0.0001
R_MAX = (SIZE / 2)
R_STEPS = 100
g = np.empty([R_STEPS], dtype=float)

for index, file_name in enumerate(files):
    xyz = open(files_path + file_name + files_format, 'r').readlines()
    P_COUNT = int(xyz[0])
    FRAMES_COUNT = int(len(xyz) / (P_COUNT + 2))
    TIME_STEP = float(xyz[1])

    print(file_name, ' : ', P_COUNT, FRAMES_COUNT, TIME_STEP)
    
    data = np.empty([FRAMES_COUNT, P_COUNT, 9], dtype=float)

    for frame in range(0, FRAMES_COUNT):
        for particle in range(0, P_COUNT):
            line = frame * (P_COUNT + 2) + (particle + 2)
            data[frame][particle] = (xyz[line]).split()
    
    pos = np.empty([P_COUNT, 3], dtype=float)
    dists = np.empty([P_COUNT, P_COUNT], dtype=float)
    threadsperblock = 32
    blockspergrid = (P_COUNT + (threadsperblock - 1)) // threadsperblock
    for frame in range(0, FRAMES_COUNT):
        print(frame)
        pos = np.vstack((
            data[frame, 0:P_COUNT, get_column(POS, X)], 
            data[frame, 0:P_COUNT, get_column(POS, Y)], 
            data[frame, 0:P_COUNT, get_column(POS, Z)]
        )).T

        get_pairwise_matrix[blockspergrid, threadsperblock](pos, P_COUNT, dists)

        for d in dists:
            h, b = np.histogram(d, bins=R_STEPS)
            g = np.add(g, h / (FRAMES_COUNT * P_COUNT))
            
    a = np.sqrt(
        np.add(
            np.square(data[0:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, X)]), 
            np.add(
                np.square(data[0:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Y)]), 
                np.square(data[0:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Z)])
            )
        )
    ).flatten()

    a_axis = np.concatenate((
        data[0:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, X)].flatten(),
        data[0:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Y)].flatten(),
        data[0:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Z)].flatten()
    ), axis=None)

    figure_axis_acc.add_trace(go.Histogram(x=a_axis, name=file_name), row = index + 1, col = 1)
    figure_full_acc.add_trace(go.Histogram(x=a, name=file_name), row = index + 1, col = 1)
figure_full_acc.update_layout(height=700 * len(files), title = "Acc module distribution")
figure_full_acc.show()

figure_axis_acc.update_layout(height=700 * len(files), title = "Acc value distribution")
figure_axis_acc.show()
