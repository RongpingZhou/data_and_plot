# Running with docker container with pytorch 2.5.1 and necessary cuda and cudnn

# docker run --gpus all -u $(id -u):$(id -g) -ti --rm -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw  -v $(realpath ~/mygit/rl/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_pytorch2-5-1:2.5.1

# docker run -u $(id -u):$(id -g) -ti --rm -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v $(realpath ~/mygit/rl/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_pytorch2-5-1:2.5.1

# xhost +local:docker

# docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

import time
import numpy as np

from evaluation.atari_data import get_human_normalized_score, get_env_id
from evaluation import library as rly
from evaluation import metrics
from evaluation import plot_utils


import tkinter as tk

import os
import copy

import gymnasium as gym

import argparse
from distutils.util import strtobool

import ale_py
gym.register_envs(ale_py)
gym.pprint_registry()

import matplotlib
import matplotlib.pyplot as plt

# Use 'TkAgg', 'Qt5Agg', 'Qt4Agg', etc.
# matplotlib.use('TkAgg')

def setup_matplotlib_backend():
    """Configure matplotlib backend based on environment"""

    # Check if running in headless environment
    if os.environ.get('DISPLAY') is None or os.environ.get('SSH_CONNECTION'):
        print("Headless environment detected, using Agg backend")
        matplotlib.use('Agg')
        return 'headless'

    # Try interactive backends in order of preference
    interactive_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg']

    for backend in interactive_backends:
        try:
            matplotlib.use(backend)
            # Test if backend works
            fig = plt.figure()
            plt.close(fig)
            print(f"Using interactive backend: {backend}")
            return 'interactive'
        except ImportError:
            continue

    # Fallback to Agg if no interactive backend works
    print("No interactive backend available, falling back to Agg")
    matplotlib.use('Agg')
    return 'headless'

matplotlib_backend = setup_matplotlib_backend()

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gym-id', type=str, default="FlappyBird",
    # parser.add_argument('--gym-id', type=str, default="ALE/Breakout-v5",
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
    parser.add_argument("--plot", type=int, default=0,
        help="show plot during training, 0 is not showing, 1 is showing")
    args = parser.parse_args()
    
    return args

args = parse_args()
print("args: ", args)
print(vars(args))

def main():

    env_id = get_env_id(args.gym_id)

    plt.ion()  # Turn on interactive mode
    fig1, ax1 = plt.subplots()
    line1, = ax1.plot([], [], 'o-', label='median') 
    line2, = ax1.plot([], [], 'x-', label='mean')

    # fig3, ax3 = plt.subplots()
    # line31, = ax3.plot([], [], 'o-', label='median') 
    # line32, = ax3.plot([], [], 'x-', label='mean')

    # ax.set_xlim(0, 10)
    # ax.set_ylim(0, 100)
    ax1.legend() 
    ax1.grid(True)

    # ax3.legend()
    # ax3.grid(True)
    
    plt.figure(fig1.number)
    # plt.figure(fig3.number)
    if args.plot == 1:
        plt.show(block=False)
    x_data = []
    y1_data = []
    y2_data = []


    files = ['./datalogs/3systems/breakout-data-dqn-model-r10_000_000.npz',
             './datalogs/3systems/breakout-data-dqn-model-c10_000_000.npz',
             './datalogs/3systems/breakout-data-dqn-model-s10_000_000.npz']
    
    # files = ['./datalogs/3systems/breakout-data-dqn-model-c10_000_000.npz',
    #          './datalogs/system1s0a/test_with_0s0a/breakout-data-dqn-model-9_000_000.npz']
                
    # files = ['./datalogs/realworld/breakout-data-dqn-model-1s1a-10_000_000.npz',
    #          './datalogs/realworld/breakout-data-dqn-model-1s0a-1_000_000.npz',
    #          './datalogs/realworld/breakout-data-dqn-model-0s0a-10_000_000.npz']

    # labels = ['1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000', 's1_000_000', 's2_000_000', 's3_000_000', 's4_000_000', 's5_000_000', 's6_000_000']

    # labels = ['real world', 'simulation with real world input', 'simulation']
    labels = ['real world system', 'real world input', 'simulation']
    # labels = ['real world input', 'simulation']
    # labels = ['real world system', 'real world input']

    # labels = ['real world', 'simulation with real world input']
    
    # labels = ['simulation with real world input', 'real world']
    # labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']
    # labels = ['1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']

    # labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000', 's1_000_000', 's2_000_000', 's3_000_000', 's4_000_000', 's5_000_000', 's6_000_000', 's7_000_000', 's8_000_000', 's9_000_000', 's10_000_000']

    # test_steps = list(range(TEST_STEP_SIZE, MAX_TEST_STEPS + TEST_STEP_SIZE, TEST_STEP_SIZE))
    # print(f"test_steps: {test_steps}")

    f = 0
    
    for file_path in files:
        
        f += 1
        
        # Load the existing data from the .npz file
        loaded_data = np.load(file_path)
        
        # Retrieve the existing array and datetime
        raw_scores = loaded_data['array']
        existing_array = np.array([get_human_normalized_score(env_id, score) for score in raw_scores])
        if len(existing_array) == 101:
            existing_array = existing_array[:-1]
            # existing_array = existing_array[1:]
        print(f"{file_path} Existing array: {existing_array}")
        csvfile = file_path.split('/')[-1].replace('.npz', '.csv')
        np.savetxt(csvfile, existing_array.reshape(-1, 1), delimiter=",")

        min_score = existing_array.min()
        max_score = existing_array.max()
        median = np.median(existing_array)
        mean = np.mean(existing_array)
        
        print("min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " mean: " + str(mean))
        x_data.append(f)
        y1_data.append(median)
        y2_data.append(mean)
        plt.figure(fig1.number)

        line1.set_data(x_data, y1_data)
        line2.set_data(x_data, y2_data)
        ax1.set_xlim(0, max(x_data) + 1)
        ax1.set_ylim(0, max(max(y1_data), max(y2_data)) + 10)
        plt.draw()
        if args.plot == 1:
            plt.pause(0.1)
            plt.show()
        
        if f == 1:
            array_for_dict = existing_array
        else:
            array_for_dict = np.vstack((array_for_dict, existing_array))

    # Save the training progress plot after all test steps are completed
    # fig1.savefig("training_progress_final.png", dpi=300, bbox_inches='tight')
    # fig3.savefig("training_progress_final3.png", dpi=300, bbox_inches='tight')
    # print("*"*5 + " Training progress plot saved as 'training_progress_final.png'")

    # fig, ax1 = plt.subplots()
    
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(list(array_for_dict), labels=labels)
    plt.ylabel('Values')
    plt.title('Box and Whisker Plot')
    plt.grid(True, alpha=0.3)
    plt.draw()
    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
    
    hns_dict = {}

    for i in range(min(len(labels), array_for_dict.shape[0])):
        # The evaluation library expects 2D arrays: [num_runs, num_tasks]
        # array_for_dict[i] is 1D (101 scores from 101 test episodes)
        # We need to reshape it to 2D: [101 runs, 1 task] for single-task evaluation
        # print(f"Original shape of array_for_dict[{i}]: {array_for_dict[i].shape}")
        scores_2d = np.array(array_for_dict[i]).reshape(-1, 1)  # Convert 1D to 2D
        # print(f"Reshaped scores_2d for {labels[i]}: {scores_2d.shape}")
        # scores_2d = np.array(array_for_dict[i]).reshape(1, -1)  # Convert 1D to 2D
        hns_dict[labels[i]] = scores_2d
        # hns_dict[labels[i]] = array_for_dict[i]
    
    # for key, value in hns_dict.items():
    #     print(f"{key}: ")
    #     print(f"{value}")    

    # aggregate_func = lambda x: np.array([
    #     metrics.aggregate_median(x),
    #     metrics.aggregate_iqm(x),
    #     metrics.aggregate_mean(x),
    #     metrics.aggregate_optimality_gap(x)])
        
    # aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(hns_dict, aggregate_func, reps=50000)

    aggregate_scores = {
        "real world system": np.array([0.40972222, 0.41041667, 0.42638889, 0.57361111]), 
        "real world input": np.array([0.70486111, 0.72291667, 0.75      , 0.27517361]), "simulation": np.array([9.07291667, 9.07291667, 8.38055556, 0.        ])}
    aggregate_score_cis = {
        "real world system": np.array([[0.35763889, 0.38055556, 0.39722222, 0.54375   ],
                                       [0.44444444, 0.44513889, 0.45625   , 0.60277778]]), 
        "real world input": np.array([[0.65277778, 0.67430556, 0.70451389, 0.23888889],
                                      [0.79166667, 0.77152778, 0.79791667, 0.31163194]]), 
        "simulation": np.array([[8.93402778, 8.73680556, 7.90347222, 0.        ],
                                [9.21180556, 9.29791667, 8.83333333, 0.        ]])}


    aggregate_scores_3sys = copy.deepcopy(aggregate_scores)
    aggregate_score_cis_3sys = copy.deepcopy(aggregate_score_cis)
    
    print(f"aggregate_scores: type: {type(aggregate_scores)}, data: {aggregate_scores}")
    print(f"aggregate_scores: type: {type(aggregate_score_cis)}, data: {aggregate_score_cis}")

    for key, value in aggregate_scores.items():
        print(f"aggregate_scores {key}: ")
        print(f"{value}")
    
    for key, value in aggregate_score_cis.items():
        print(f"aggregate_score_cis {key}: ")
        print(f"{value}")        

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig2, ax2 = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
        algorithms=available_algorithms,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig2.set_size_inches(12, 4)
    
    plt.figure(fig2.number)
    
    fig2.savefig("hns_all_3sys.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
    # plt.draw()

    aggregate_scores_median = {key: value[0:1] for key, value in aggregate_scores.items()}
    aggregate_score_cis_median = {key: value[:, 0:1]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_median: type: {type(aggregate_scores_median)}, data: {aggregate_scores_median}")
    print(f"aggregate_score_cis_median: type: {type(aggregate_score_cis_median)}, data: {aggregate_score_cis_median}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig3, ax3 = plot_utils.plot_interval_estimates(
        aggregate_scores_median,
        aggregate_score_cis_median,
        metric_names=['Median'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig3.set_size_inches(12, 4)
    
    plt.figure(fig3.number)
    
    fig3.savefig("hns_median_3sys.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_iqm = {key: value[1:2] for key, value in aggregate_scores.items()}
    aggregate_score_cis_iqm = {key: value[:, 1:2]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_iqm: type: {type(aggregate_scores_iqm)}, data: {aggregate_scores_iqm}")
    print(f"aggregate_score_cis_iqm: type: {type(aggregate_score_cis_iqm)}, data: {aggregate_score_cis_iqm}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig4, ax4 = plot_utils.plot_interval_estimates(
        aggregate_scores_iqm,
        aggregate_score_cis_iqm,
        metric_names=['IQM'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig4.set_size_inches(12, 4)
    
    plt.figure(fig4.number)
    
    fig4.savefig("hns_iqm_3sys.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_mean = {key: value[2:3] for key, value in aggregate_scores.items()}
    aggregate_score_cis_mean = {key: value[:, 2:3]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_mean: type: {type(aggregate_scores_mean)}, data: {aggregate_scores_mean}")
    print(f"aggregate_score_cis_mean: type: {type(aggregate_score_cis_mean)}, data: {aggregate_score_cis_mean}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig5, ax5 = plot_utils.plot_interval_estimates(
        aggregate_scores_mean,
        aggregate_score_cis_mean,
        metric_names=['Mean'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig5.set_size_inches(12, 4)
    
    plt.figure(fig5.number)
    
    fig5.savefig("hns_mean_3sys.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_optimality_gap = {key: value[3:4] for key, value in aggregate_scores.items()}
    aggregate_score_cis_optimality_gap = {key: value[:, 3:4]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_optimality_gap: type: {type(aggregate_scores_optimality_gap)}, data: {aggregate_scores_optimality_gap}")
    print(f"aggregate_score_cis_optimality_gap: type: {type(aggregate_score_cis_optimality_gap)}, data: {aggregate_score_cis_optimality_gap}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig6, ax6 = plot_utils.plot_interval_estimates(
        aggregate_scores_optimality_gap,
        aggregate_score_cis_optimality_gap,
        metric_names=['Optimality Gap'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig6.set_size_inches(12, 4)
    
    plt.figure(fig6.number)
    
    fig6.savefig("hns_optimality_gap_3sys.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
        
    # if args.plot == 1:
    #     plt.show(block=False)


    # Real World Input System
    files = ['./datalogs/3systems/breakout-data-dqn-model-c10_000_000.npz',
             './datalogs/system1s0a/test_with_0s0a/breakout-data-dqn-model-9_000_000.npz']
                
    labels = ['real world input', 'simulation']

    f = 0
    
    for file_path in files:
        
        f += 1
        
        # Load the existing data from the .npz file
        loaded_data = np.load(file_path)
        
        # Retrieve the existing array and datetime
        raw_scores = loaded_data['array']
        existing_array = np.array([get_human_normalized_score(env_id, score) for score in raw_scores])
        if len(existing_array) == 101:
            existing_array = existing_array[:-1]
            # existing_array = existing_array[1:]
        print(f"{file_path} Existing array: {existing_array}")
        csvfile = file_path.split('/')[-1].replace('.npz', '.csv')
        np.savetxt(csvfile, existing_array.reshape(-1, 1), delimiter=",")

        min_score = existing_array.min()
        max_score = existing_array.max()
        median = np.median(existing_array)
        mean = np.mean(existing_array)
        
        print("min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " mean: " + str(mean))
        x_data.append(f)
        y1_data.append(median)
        y2_data.append(mean)
        plt.figure(fig1.number)

        line1.set_data(x_data, y1_data)
        line2.set_data(x_data, y2_data)
        ax1.set_xlim(0, max(x_data) + 1)
        ax1.set_ylim(0, max(max(y1_data), max(y2_data)) + 10)
        plt.draw()
        if args.plot == 1:
            plt.pause(0.1)
            plt.show()
        
        if f == 1:
            array_for_dict = existing_array
        else:
            array_for_dict = np.vstack((array_for_dict, existing_array))

    # Save the training progress plot after all test steps are completed
    # fig1.savefig("training_progress_final.png", dpi=300, bbox_inches='tight')
    # fig3.savefig("training_progress_final3.png", dpi=300, bbox_inches='tight')
    # print("*"*5 + " Training progress plot saved as 'training_progress_final.png'")

    # fig, ax1 = plt.subplots()
    
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(list(array_for_dict), labels=labels)
    plt.ylabel('Values')
    plt.title('Box and Whisker Plot')
    plt.grid(True, alpha=0.3)
    plt.draw()
    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
    
    hns_dict = {}

    for i in range(min(len(labels), array_for_dict.shape[0])):
        # The evaluation library expects 2D arrays: [num_runs, num_tasks]
        # array_for_dict[i] is 1D (101 scores from 101 test episodes)
        # We need to reshape it to 2D: [101 runs, 1 task] for single-task evaluation
        # print(f"Original shape of array_for_dict[{i}]: {array_for_dict[i].shape}")
        scores_2d = np.array(array_for_dict[i]).reshape(-1, 1)  # Convert 1D to 2D
        # print(f"Reshaped scores_2d for {labels[i]}: {scores_2d.shape}")
        # scores_2d = np.array(array_for_dict[i]).reshape(1, -1)  # Convert 1D to 2D
        hns_dict[labels[i]] = scores_2d
        # hns_dict[labels[i]] = array_for_dict[i]
    
    # for key, value in hns_dict.items():
    #     print(f"{key}: ")
    #     print(f"{value}")    

    # aggregate_func = lambda x: np.array([
    #     metrics.aggregate_median(x),
    #     metrics.aggregate_iqm(x),
    #     metrics.aggregate_mean(x),
    #     metrics.aggregate_optimality_gap(x)])
        
    # aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(hns_dict, aggregate_func, reps=50000)

    aggregate_scores = {
        "real world input": np.array([0.70486111, 0.72291667, 0.75      , 0.27517361]), 
        "simulation": np.array([ 0.01041667, -0.00763889, -0.01111111,  1.01111111])}
    aggregate_score_cis = {
        "real world input": np.array([[0.65277778, 0.67430556, 0.70451389, 0.23888889],
       [0.79166667, 0.77152778, 0.79791667, 0.31163194]]), 
        "simulation": np.array([[-0.02430556, -0.01944444, -0.01840278,  1.00381944],
       [ 0.01041667,  0.00138889, -0.00381944,  1.01840278]])}

    
    aggregate_scores['real world input'] = copy.deepcopy(aggregate_scores_3sys['real world input'])
    aggregate_score_cis['real world input'] = copy.deepcopy(aggregate_score_cis_3sys['real world input'])
    
    print(f"aggregate_scores: type: {type(aggregate_scores)}, data: {aggregate_scores}")
    print(f"aggregate_scores: type: {type(aggregate_score_cis)}, data: {aggregate_score_cis}")

    for key, value in aggregate_scores.items():
        print(f"aggregate_scores {key}: ")
        print(f"{value}")
    
    for key, value in aggregate_score_cis.items():
        print(f"aggregate_score_cis {key}: ")
        print(f"{value}")        

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig7, ax7 = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
        algorithms=available_algorithms,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig7.set_size_inches(12, 4)
    
    plt.figure(fig7.number)
    
    fig7.savefig("hns_all_real_input.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
    # plt.draw()

    aggregate_scores_median = {key: value[0:1] for key, value in aggregate_scores.items()}
    aggregate_score_cis_median = {key: value[:, 0:1]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_median: type: {type(aggregate_scores_median)}, data: {aggregate_scores_median}")
    print(f"aggregate_score_cis_median: type: {type(aggregate_score_cis_median)}, data: {aggregate_score_cis_median}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig8, ax8 = plot_utils.plot_interval_estimates(
        aggregate_scores_median,
        aggregate_score_cis_median,
        metric_names=['Median'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig8.set_size_inches(12, 4)
    
    plt.figure(fig8.number)
    
    fig8.savefig("hns_median_real_input.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_iqm = {key: value[1:2] for key, value in aggregate_scores.items()}
    aggregate_score_cis_iqm = {key: value[:, 1:2]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_iqm: type: {type(aggregate_scores_iqm)}, data: {aggregate_scores_iqm}")
    print(f"aggregate_score_cis_iqm: type: {type(aggregate_score_cis_iqm)}, data: {aggregate_score_cis_iqm}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig9, ax9 = plot_utils.plot_interval_estimates(
        aggregate_scores_iqm,
        aggregate_score_cis_iqm,
        metric_names=['IQM'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig9.set_size_inches(12, 4)
    
    plt.figure(fig9.number)
    
    fig9.savefig("hns_iqm_real_input.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_mean = {key: value[2:3] for key, value in aggregate_scores.items()}
    aggregate_score_cis_mean = {key: value[:, 2:3]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_mean: type: {type(aggregate_scores_mean)}, data: {aggregate_scores_mean}")
    print(f"aggregate_score_cis_mean: type: {type(aggregate_score_cis_mean)}, data: {aggregate_score_cis_mean}")


    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig10, ax10 = plot_utils.plot_interval_estimates(
        aggregate_scores_mean,
        aggregate_score_cis_mean,
        metric_names=['Mean'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig10.set_size_inches(12, 4)
    
    plt.figure(fig10.number)
    
    fig10.savefig("hns_mean_real_input.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_optimality_gap = {key: value[3:4] for key, value in aggregate_scores.items()}
    aggregate_score_cis_optimality_gap = {key: value[:, 3:4]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_optimality_gap: type: {type(aggregate_scores_optimality_gap)}, data: {aggregate_scores_optimality_gap}")
    print(f"aggregate_score_cis_optimality_gap: type: {type(aggregate_score_cis_optimality_gap)}, data: {aggregate_score_cis_optimality_gap}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig11, ax11 = plot_utils.plot_interval_estimates(
        aggregate_scores_optimality_gap,
        aggregate_score_cis_optimality_gap,
        metric_names=['Optimality Gap'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig11.set_size_inches(12, 4)
    
    plt.figure(fig11.number)
    
    fig11.savefig("hns_optimality_gap_real_input.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
        
    # Real World System
    # files = ['./datalogs/realworld/breakout-data-dqn-model-1s1a-10_000_000.npz',
    #          './datalogs/realworld/breakout-data-dqn-model-1s0a-1_000_000.npz',
    #          './datalogs/realworld/breakout-data-dqn-model-0s0a-10_000_000.npz']

    files = ['./datalogs/3systems/breakout-data-dqn-model-r10_000_000.npz',
             './datalogs/realworld/breakout-data-dqn-model-1s0a-1_000_000.npz',
             './datalogs/realworld/breakout-data-dqn-model-0s0a-10_000_000.npz']
             
    labels = ['real world system', 'real world input', 'simulation']

    f = 0
    
    for file_path in files:
        
        f += 1
        
        # Load the existing data from the .npz file
        loaded_data = np.load(file_path)
        
        # Retrieve the existing array and datetime
        raw_scores = loaded_data['array']
        existing_array = np.array([get_human_normalized_score(env_id, score) for score in raw_scores])
        if len(existing_array) == 101:
            existing_array = existing_array[:-1]
            # existing_array = existing_array[1:]
        print(f"{file_path} Existing array: {existing_array}")
        csvfile = file_path.split('/')[-1].replace('.npz', '.csv')
        np.savetxt(csvfile, existing_array.reshape(-1, 1), delimiter=",")

        min_score = existing_array.min()
        max_score = existing_array.max()
        median = np.median(existing_array)
        mean = np.mean(existing_array)
        
        print("min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " mean: " + str(mean))
        x_data.append(f)
        y1_data.append(median)
        y2_data.append(mean)
        plt.figure(fig1.number)

        line1.set_data(x_data, y1_data)
        line2.set_data(x_data, y2_data)
        ax1.set_xlim(0, max(x_data) + 1)
        ax1.set_ylim(0, max(max(y1_data), max(y2_data)) + 10)
        plt.draw()
        if args.plot == 1:
            plt.pause(0.1)
            plt.show()
        
        if f == 1:
            array_for_dict = existing_array
        else:
            array_for_dict = np.vstack((array_for_dict, existing_array))

    # Save the training progress plot after all test steps are completed
    # fig1.savefig("training_progress_final.png", dpi=300, bbox_inches='tight')
    # fig3.savefig("training_progress_final3.png", dpi=300, bbox_inches='tight')
    # print("*"*5 + " Training progress plot saved as 'training_progress_final.png'")

    # fig, ax1 = plt.subplots()
    
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(list(array_for_dict), labels=labels)
    plt.ylabel('Values')
    plt.title('Box and Whisker Plot')
    plt.grid(True, alpha=0.3)
    plt.draw()
    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
    
    hns_dict = {}

    for i in range(min(len(labels), array_for_dict.shape[0])):
        # The evaluation library expects 2D arrays: [num_runs, num_tasks]
        # array_for_dict[i] is 1D (101 scores from 101 test episodes)
        # We need to reshape it to 2D: [101 runs, 1 task] for single-task evaluation
        # print(f"Original shape of array_for_dict[{i}]: {array_for_dict[i].shape}")
        scores_2d = np.array(array_for_dict[i]).reshape(-1, 1)  # Convert 1D to 2D
        # print(f"Reshaped scores_2d for {labels[i]}: {scores_2d.shape}")
        # scores_2d = np.array(array_for_dict[i]).reshape(1, -1)  # Convert 1D to 2D
        hns_dict[labels[i]] = scores_2d
        # hns_dict[labels[i]] = array_for_dict[i]
    
    # for key, value in hns_dict.items():
    #     print(f"{key}: ")
    #     print(f"{value}")    

    # aggregate_func = lambda x: np.array([
    #     metrics.aggregate_median(x),
    #     metrics.aggregate_iqm(x),
    #     metrics.aggregate_mean(x),
    #     metrics.aggregate_optimality_gap(x)])
        
    # aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(hns_dict, aggregate_func, reps=50000)

    aggregate_scores = {
        "real world system": np.array([0.40972222, 0.41041667, 0.42638889, 0.57361111]), 
        "real world input": np.array([0.07986111, 0.09722222, 0.10208333, 0.89791667]), 
        "simulation": np.array([0.01041667, 0.02152778, 0.02847222, 0.97152778])}
    aggregate_score_cis = {
        "real world system": np.array([[0.35763889, 0.38055556, 0.39722222, 0.54375   ],
       [0.44444444, 0.44513889, 0.45625   , 0.60277778]]), 
        "real world input": np.array([[0.07986111, 0.08263889, 0.08819444, 0.88368056],
       [0.11458333, 0.11111111, 0.11631944, 0.91180556]]), 
        "simulation": np.array([[0.01041667, 0.01041667, 0.01736111, 0.96006944],
       [0.04513889, 0.03333333, 0.03993056, 0.98263889]])}
        
    aggregate_scores['real world system'] = copy.deepcopy(aggregate_scores_3sys['real world system'])
    aggregate_score_cis['real world system'] = copy.deepcopy(aggregate_score_cis_3sys['real world system'])
        
    print(f"aggregate_scores: type: {type(aggregate_scores)}, data: {aggregate_scores}")
    print(f"aggregate_scores: type: {type(aggregate_score_cis)}, data: {aggregate_score_cis}")

    for key, value in aggregate_scores.items():
        print(f"aggregate_scores {key}: ")
        print(f"{value}")
    
    for key, value in aggregate_score_cis.items():
        print(f"aggregate_score_cis {key}: ")
        print(f"{value}")        

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig12, ax12 = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
        algorithms=available_algorithms,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig12.set_size_inches(12, 4)
    
    plt.figure(fig2.number)
    
    fig12.savefig("hns_all_real.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
    # plt.draw()

    aggregate_scores_median = {key: value[0:1] for key, value in aggregate_scores.items()}
    aggregate_score_cis_median = {key: value[:, 0:1]  for key, value in aggregate_score_cis.items()}


    print(f"aggregate_scores_median: type: {type(aggregate_scores_median)}, data: {aggregate_scores_median}")
    print(f"aggregate_score_cis_median: type: {type(aggregate_score_cis_median)}, data: {aggregate_score_cis_median}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig13, ax13 = plot_utils.plot_interval_estimates(
        aggregate_scores_median,
        aggregate_score_cis_median,
        metric_names=['Median'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig13.set_size_inches(12, 4)
    
    plt.figure(fig13.number)
    
    fig13.savefig("hns_median_real.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_iqm = {key: value[1:2] for key, value in aggregate_scores.items()}
    aggregate_score_cis_iqm = {key: value[:, 1:2]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_iqm: type: {type(aggregate_scores_iqm)}, data: {aggregate_scores_iqm}")
    print(f"aggregate_score_cis_iqm: type: {type(aggregate_score_cis_iqm)}, data: {aggregate_score_cis_iqm}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig14, ax14 = plot_utils.plot_interval_estimates(
        aggregate_scores_iqm,
        aggregate_score_cis_iqm,
        metric_names=['IQM'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig14.set_size_inches(12, 4)
    
    plt.figure(fig14.number)
    

    fig14.savefig("hns_iqm_real.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_mean = {key: value[2:3] for key, value in aggregate_scores.items()}
    aggregate_score_cis_mean = {key: value[:, 2:3]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_mean: type: {type(aggregate_scores_mean)}, data: {aggregate_scores_mean}")
    print(f"aggregate_score_cis_mean: type: {type(aggregate_score_cis_mean)}, data: {aggregate_score_cis_mean}")


    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig15, ax15 = plot_utils.plot_interval_estimates(
        aggregate_scores_mean,
        aggregate_score_cis_mean,
        metric_names=['Mean'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig15.set_size_inches(12, 4)
    
    plt.figure(fig15.number)
    
    fig15.savefig("hns_mean_real.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()

    aggregate_scores_optimality_gap = {key: value[3:4] for key, value in aggregate_scores.items()}
    aggregate_score_cis_optimality_gap = {key: value[:, 3:4]  for key, value in aggregate_score_cis.items()}

    print(f"aggregate_scores_optimality_gap: type: {type(aggregate_scores_optimality_gap)}, data: {aggregate_scores_optimality_gap}")
    print(f"aggregate_score_cis_optimality_gap: type: {type(aggregate_score_cis_optimality_gap)}, data: {aggregate_score_cis_optimality_gap}")

    # Use only the algorithms that actually have data
    available_algorithms = list(hns_dict.keys())
    print("*"*5 + " Available algorithms for plotting: ", available_algorithms)
    
    fig16, ax16 = plot_utils.plot_interval_estimates(
        aggregate_scores_optimality_gap,
        aggregate_score_cis_optimality_gap,
        metric_names=['Optimality Gap'],
        algorithms=available_algorithms,
        subfigure_width=3.4,
        row_height=1.0,
        # xlabel_y_coordinate=-0.05,
        xlabel_y_coordinate=-0.09,
        xlabel='Human Normalized Score')

    fig16.set_size_inches(12, 4)
    
    plt.figure(fig16.number)
    
    fig16.savefig("hns_optimality_gap_real.png", dpi=300, bbox_inches='tight')

    if args.plot == 1:    
        plt.pause(0.1)
        plt.show()
        

    if args.plot == 1:
        
        # Show the plot based on backend
        current_backend = matplotlib.get_backend()
        print("*"*5 + " Using matplotlib backend: ", current_backend)
    
        plt.show(block=False)
        
        # Keep the plot alive - multiple approaches to ensure persistence
        try:
            print("*"*5 + " Plot window should be visible with backend: ", current_backend)
            print("*"*5 + " Waiting for plot interaction... (Close the plot window to exit)")
            
            # Robust approach to keep the plot window open
            was_interactive = plt.isinteractive()
            
            # First: ensure the window is shown and focused
            plt.pause(0.5)  # Give time for window to appear
            
            # Check if figure is still open
            if plt.get_fignums():
                
                print("*"*5 + " Checking plot window status...")
                
                # Try to bring window to absolute front
                try:
                    if hasattr(fig2.canvas.manager, 'window'):
                        root = fig2.canvas.manager.window
                        root.attributes('-topmost', True)  # Bring to very front
                        root.after(100, lambda: root.attributes('-topmost', False))  # Then allow normal behavior
                        print("*"*5 + " Window brought to front")
                except Exception as e:
                    print("*"*5 + " Could not force window to front:", e)
                
                # Now use blocking show to keep window open
                plt.ioff()  # Turn off interactive mode for blocking
                print("*"*5 + " Plot window should now be persistent...")
                plt.show(block=True)  # This should block until window is closed
            
            # Fallback: If the window closed too quickly, offer alternative
            if plt.get_fignums():  # If figures are still somehow open
                print("\n" + "*"*5 + " Plot window is still open. Keep this terminal open to maintain the plot.")
                print("*"*5 + " You can close the plot window directly, or press Enter here to exit.")
                try:
                    input("Press Enter to exit...")
                except (EOFError, KeyboardInterrupt):
                    pass
            
            # Restore interactive mode if it was on
            if was_interactive:
                plt.ion()
                
        except KeyboardInterrupt:
            print("*"*5 + " Plot display interrupted by user (Ctrl+C)")
            plt.close('all')
                
    plt.ioff()
    plt.close()
    
    return

if __name__ == "__main__":
    main()
