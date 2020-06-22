import math 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COORDS = 0
VELOCITY = 1
ACC = 2

X = 0
Y = 1
Z = 2

def get_column(a, b):
    return 3 * a + b

files_path = 'data/'
files_format = '.xyz'
files = [
    'FCC_B_1_0_A_1_5',
    'FCC_B_1_5_A_1_5',
    'FCC_B_2_A_0_5',
    'FCC_B_2_A_1_0',
    'FCC_B_2_A_1_5',
    'FCC_LJ'
]

figure_full_acc = make_subplots(rows=len(files), cols=1)
figure_axis_acc = make_subplots(rows=len(files), cols=1)

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
