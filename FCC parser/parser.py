import math 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from numba import njit, jit, cuda
from lmfit import Model, Parameter, report_fit

SIZE = 11.3137

POS = 0
VELOCITY = 1
ACC = 2

X = 0
Y = 1
Z = 2

R_MAX = SIZE / 2
R_STEPS = 1000

def get_column(a, b):
    return 3 * a + b

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

def gaussian(x, A, s):
    return A * np.exp(-x*x/s/s/2)

@cuda.jit
def get_g_r(p, count, res):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx + ty * bw
    
    for i in range(0, count):
        if (i != idx):
            d1 = (p[idx][0] - p[i][0]) / SIZE
            d2 = (p[idx][1] - p[i][1]) / SIZE 
            d3 = (p[idx][2] - p[i][2]) / SIZE

            d1 = d1 - round(d1)
            d2 = d2 - round(d2)
            d3 = d3 - round(d3)

            d = ((d1*d1 + d2*d2 + d3*d3) ** (0.5)) * SIZE
            j = int(d / R_MAX * R_STEPS)
            if j < R_STEPS:
                res[idx][j] = res[idx][j] + 1

files_path = 'data/'
files_format = '.xyz'

files = glob.glob(files_path + "*" + files_format)
files = [(f.split(files_path)[1]).split(files_format)[0] for f in files]

figure_full_acc = make_subplots(rows=len(files), cols=1)
figure_axis_acc = make_subplots(rows=len(files), cols=1)
figure_g_r = make_subplots(rows=len(files), cols=1)

g = np.zeros([R_STEPS], dtype=float)
v = np.zeros([R_STEPS], dtype=float)
b = np.zeros([R_STEPS + 1], dtype=float)

for i in range(0, R_STEPS + 1):
    b[i] = R_MAX / R_STEPS * i

for i in range(0, R_STEPS):
    v[i] = 1 / (4/3 * 3.14 * (b[i + 1] ** 3 - b[i] ** 3)) 

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
        if (frame < FRAMES_COUNT - 1):
            printProgressBar(frame + 1, FRAMES_COUNT, prefix = 'g(r)')
        else:
            printProgressBar(frame + 1, FRAMES_COUNT, prefix = 'g(r)', printEnd="\n")

        pos = np.vstack((
            data[frame, 0:P_COUNT, get_column(POS, X)], 
            data[frame, 0:P_COUNT, get_column(POS, Y)], 
            data[frame, 0:P_COUNT, get_column(POS, Z)] 
        )).T

        h_m = np.zeros([P_COUNT, R_STEPS], dtype=float)     
        get_g_r[blockspergrid, threadsperblock](pos, P_COUNT, h_m)
       
        for h in h_m:
            g = np.add(g, h * v)
   
    a = np.sqrt(
        np.add(
            np.square(data[1:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, X)]), 
            np.add(
                np.square(data[1:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Y)]), 
                np.square(data[1:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Z)])
            )
        )
    ).flatten()

    a_axis = np.concatenate((
        data[1:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, X)].flatten(),
        data[1:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Y)].flatten(),
        data[1:FRAMES_COUNT, 0:P_COUNT, get_column(ACC, Z)].flatten()
    ), axis=None)
    
    h, b = np.histogram(a_axis, bins=1000)
    gmodel = Model(gaussian, independent_vars=['x'])    
    result = gmodel.fit(h, x=b[0:len(b) - 1], A = 10000, s = 3)
    A = float(result.params["A"])
    s = float(result.params["s"])
    print(result.fit_report())    

    figure_axis_acc.add_trace(go.Bar(x=b[0:len(b) - 1], y = h, name=file_name), row = index + 1, col = 1)
    figure_axis_acc.add_trace(go.Scatter(x=b, y=[gaussian(b, A, s) for b in b], name=file_name), row = index + 1, col = 1)
    figure_full_acc.add_trace(go.Histogram(x=a, name=file_name), row = index + 1, col = 1)
    k = (P_COUNT ** 2) * (FRAMES_COUNT) /  (SIZE ** 3)
    figure_g_r.add_trace(go.Scatter(x=[i / R_STEPS * R_MAX + 0.5 * R_MAX / R_STEPS  for i in range(0, R_STEPS)], y=g / k, mode="markers", name=file_name), row = index + 1, col = 1)

figure_full_acc.update_layout(height=700 * len(files), title = "Acc module distribution")
figure_full_acc.show()

figure_axis_acc.update_layout(height=700 * len(files), title = "Acc value distribution")
figure_axis_acc.show()

figure_g_r.update_layout(height=700 * len(files), title = "g(r)")
figure_g_r.show()
