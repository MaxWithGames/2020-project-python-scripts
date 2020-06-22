import math 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from numba import njit, jit, cuda
from lmfit import Model, Parameter, report_fit
import sys
import pandas as pd

SIZE = 11.3137

POS = 0
VELOCITY = 1
ACC = 2

X = 0
Y = 1
Z = 2

def get_column(a, b):
    return 3 * a + b

files_path = 'data/'
files_format = '.xyz'

files = glob.glob(files_path + "*" + files_format)
files = [(f.split(files_path)[1]).split(files_format)[0] for f in files]

fig = make_subplots(rows=len(files), cols=1)

for index, file_name in enumerate(files):
    xyz = open(files_path + file_name + files_format, 'r').readlines()
    P_COUNT = int(xyz[0])
    FRAMES_COUNT = int(len(xyz) / (P_COUNT + 2))
    TIME_STEP = float(xyz[1])

    print(file_name, ' : ', P_COUNT, FRAMES_COUNT, TIME_STEP)
    
    data = np.empty([FRAMES_COUNT, P_COUNT, 9], dtype=float)

    for frame in range(1, FRAMES_COUNT):
        for particle in range(0, P_COUNT):
            line = frame * (P_COUNT + 2) + (particle + 2)
            data[frame][particle] = (xyz[line]).split()

    du_vec = np.empty([P_COUNT, 3], dtype=float)
    u2 = np.empty([FRAMES_COUNT - 1], dtype=float)

    for frame in range(1, FRAMES_COUNT):
        du_cm = np.sum(
            data[frame, 0:P_COUNT, get_column(POS, X) : get_column(POS, X) + 3] - 
            data[0,          0:P_COUNT, get_column(POS, X) : get_column(POS, X) + 3], axis = 0
        ) / P_COUNT

        du_vec =  (
            data[frame, 0:P_COUNT, get_column(POS, X) : get_column(POS, X) + 3] - 
            data[0,          0:P_COUNT, get_column(POS, X) : get_column(POS, X) + 3]
        )
    
        du_vec[0:P_COUNT] -= du_cm

        du_vec = np.square(du_vec)
        u2[frame - 1] = (np.sum(du_vec) / P_COUNT)**0.5

    temp = pd.read_csv(files_path + file_name +'.csv', names=["Time", "T"])
    fig.add_trace(go.Scatter(x=(temp["T"].to_numpy())[1:], y=u2, line=dict(width=4), name=file_name), row = index + 1, col = 1)
    
fig.update_layout(height=700 * len(files), title = "u^2")
fig.show()

fig.update_xaxes(showline=True, linewidth=6, linecolor='black', mirror=True, showgrid=False)
fig.update_yaxes(showline=True, linewidth=6, linecolor='black', mirror=True, rangemode="nonnegative", showgrid=False)
fig.update_layout(font=dict(size=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.write_image("img/lindemann.png", width=1080 * 3, height=700*len(files)*3)

