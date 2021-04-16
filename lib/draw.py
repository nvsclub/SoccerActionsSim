import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Wedge, Arc

from random import random
import numpy as np

dev_neutralizer = 3
horizontal_scalling = 10.5/6.8

#bg_color = '#091442', line_color = '#3562A6'
def pitch(bg_color = '#FFFFFF', line_color = '#000000', dpi = 144):
    # Background cleanup
    plt.rcParams['figure.figsize'] = (10.5,6.8)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.facecolor'] = bg_color
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.scatter(50, 50, s=1000000, marker='s', color=bg_color)

    # Set plotting limit
    plt.xlim([-5, 105])
    plt.ylim([-5, 105])

    # Outside lines
    plt.axvline(0, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    plt.axvline(100, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    plt.axhline(0, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)
    plt.axhline(100, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)

    # Midfield line
    plt.axvline(50, ymin=0.0455, ymax=0.9545, linewidth=1, color=line_color)

    # Goals
    plt.axvline(0, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(100, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(-1, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(101, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)

    # Small Box
    ## (Width-SmallboxWidth)/2/ScaleTo100, (Margin+(Width-SmallboxWidth)/2/ScaleTo100)/(100+Margins)
    ## (68-7.32-11)/2/0.68, (5+((68-7.32-11)/2/.68))/110
    ## (5+5.5/1.05)/110, 5.25/1.05
    plt.axvline(5.24, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)
    plt.axvline(94.76, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)

    plt.axhline(36.53, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)
    plt.axhline(63.47, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)

    plt.axhline(36.53, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)
    plt.axhline(63.47, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)

    # Big Box
    plt.axvline(15.72, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    plt.axhline(20.37, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)
    plt.axhline(79.63, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)

    plt.axvline(84.28, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    plt.axhline(20.37, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color)
    plt.axhline(79.63, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color);

    # Penalty and starting spots and arcs
    plt.scatter([10.4762, 89.5238, 50], [50,50,50], s=1, color=line_color)
    e1 = Arc((10.4762,50), 17.5, 27, theta1=-64, theta2=64, fill=False, color=line_color)
    e2 = Arc((89.5238,50), 17.5, 27, theta1=116, theta2=244, fill=False, color=line_color)
    e3 = Arc((50,50), 17.5, 27, fill=False, color=line_color)
    plt.gcf().gca().add_artist(e1)
    plt.gcf().gca().add_artist(e2)
    plt.gcf().gca().add_artist(e3)



# Plot the latest sequence of actions made by the agent
def plot_action_sequence(action_storage):
    pitch()
    for action_no, action_sample in enumerate(action_storage):
        x, y = action_sample['observation'][0], action_sample['observation'][1]

        if action_sample['action'][0] < 0.5:
            plt.scatter(x*100, y*100, color = 'C0', zorder = 8 + 4 * action_no)
            xt = 1
            yt = 0.5
        else:
            plt.scatter(x*100, y*100, color = 'C1', zorder = 8 + 4 * action_no)
            r, a = action_sample['action'][1], action_sample['action'][2]
            xt = x + r * np.cos((a - 0.5) * 2 * np.pi)
            yt = y + r * np.sin((a - 0.5) * 2 * np.pi)

        x, y, xt, yt = x*100, y*100, xt*100, yt*100

        if xt - x == 0:
            if yt - y > 0:
                angle = 90
            else:
                angle = 270
        else:
            if xt - x > 0:
                angle = np.arctan((yt - y) / (xt - x)) * 360 / np.pi / 2
            else:
                angle = 180 + np.arctan((yt - y) / (xt - x)) * 360 / np.pi / 2
        distance = ((yt - y) ** 2 + ((xt - x)) ** 2) ** (1/2)

        all_patches = []
        all_patches.append(Wedge((x,y), distance, angle-0.5, angle+0.5, fc='#091442', zorder = 11 + 4 * action_no))
        all_patches.append(Arc((x,y), 3, 3*horizontal_scalling, 0, angle-45, angle+45, lw = 7 * action_sample['success'], ec='#091442', zorder = 10 + 4 * action_no))
        all_patches.append(Arc((x,y), 3, 3*horizontal_scalling, 0, angle-45, angle+45, lw = 7, ec='#a2b3ff', zorder = 9 + 4 * action_no))

        for patch in all_patches:
            plt.gcf().gca().add_artist(patch)

def plot_multiple_action_sequences(action_storages, dpi=144):
    pitch(dpi=dpi)
    actions_group = []
    for action_storage in action_storages:
        for action_no, action_sample in enumerate(action_storage):
            if action_no >= len(actions_group):
                actions_group.append({'action': action_sample['action'], 'observation': action_sample['observation'], 'success': int(action_sample['success']), 'count': 1})
            else:
                actions_group[action_no]['count'] += 1
                if action_sample['success']:
                    actions_group[action_no]['success'] += 1
    
    for action_no, _ in enumerate(actions_group):
        actions_group[action_no]['success'] /= actions_group[action_no]['count']

    plot_action_sequence(actions_group)