# Imports
import tkinter
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
import lib.draw as draw

from env.SoccerActionsEnv import SoccerActionsEnv

import numpy as np
from random import random
import time

dev_neutralizer = 3
horizontal_scalling = 10.5/6.8

# Initializing Tkinter
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# Defines
img_dir = 'tmp/screen.png'
rules_text = 'Rules:\nYou are given a random starting position.'
rules_text += '\nLeft click with the mouse on the map to pass the ball'
rules_text += '\nPress space to shoot.'
rules_text += '\nPress f to change action (Pass/Blue, Dribble/Orange).'

# Tracking variables
global rewards_cumulative
rewards_cumulative = 0
global games_played
games_played = 1
global done
done = False
global currently_passing
currently_passing = True

# Summon initial agent
env = SoccerActionsEnv(50, 50, randomized_start=True, end_on_xg=True)
env.reset()
# Draw initial image
draw.pitch()
plt.scatter(env.x*100, env.y*100, color = 'C0')
plt.text(0, 105, rules_text, color = 'black')
reward_text =  'Games Finished: ' + str(games_played)
reward_text += '\nTotal Rewards: ' + str(round(rewards_cumulative, 3))
reward_text += '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

# Update screen flow
def update_screen():
    # Update frame
    img = ImageTk.PhotoImage(Image.open(img_dir))
    panel.configure(image=img)
    panel.image = img

# Retrieve mouse information and display it on screen
def get_clicks(event):
    global rewards_cumulative
    global games_played
    global done
    global currently_passing

    if done:
        restart(None)
        return

    # Standardize coordinates
    xt = (event.x - 240) / 1072
    yt = (event.y - 90) / 690 * -1 + 1 # Inverting coordinates in the end

    # Check if out of bounds action
    if xt > 1 or xt < 0 or yt > 1 or yt < 0:
        plt.text(50, -0.3, 'Out of Bounds!', color = 'C1', horizontalalignment='center', verticalalignment='center')

        plt.savefig(img_dir)
        img = ImageTk.PhotoImage(Image.open(img_dir))
        panel.configure(image=img)
        panel.image = img

        return

    # Setup action
    action_type = 1
    ## Convert xt, yt to r, a
    r = np.sqrt((xt - env.x) ** 2 + (yt - env.y) ** 2)
    a = np.arctan2(yt - env.y, xt - env.x) / (2 * np.pi) + 0.5

    # Do action
    obs, rewards, done, info = env.step([action_type, r, a])


    # Reset to plot next picture
    plt.clf()
    # Plot action sequence
    draw.plot_action_sequence(env.action_storage)

    # End play clause
    if done:
        plt.text(50, 101, 'Play over. Press R to Restart. Press S to save.', color = 'C1', horizontalalignment='center', verticalalignment='center')
    
    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# End play by shooting at goal
def shot(event):
    global rewards_cumulative
    global games_played
    global done

    if done:
        restart(None)
        return

    # Do action
    action_type = 0
    r = np.sqrt((1 - env.x) ** 2 + (0.5 - env.y) ** 2)
    a = np.arctan2(0.5 - env.y, 1 - env.x) / (2 * np.pi) + 0.5
    obs, rewards, done, info = env.step([action_type, r, a])
    rewards_cumulative += info['expectedGoals']

    # Reset to plot next picture
    plt.clf()
    # Plot action sequence
    draw.plot_action_sequence(env.action_storage)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')
    plt.text(50, 101, 'Play over. Press R to Restart. Press S to save.', color = 'C1', horizontalalignment='center', verticalalignment='center')
    plt.text(103, 50, 'XG: ' + str(round(rewards, 3)), color = 'black')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to reset current play
def restart(event):
    global games_played
    global done
    global currently_passing

    # Increment game counter
    games_played += 1
    done = False

    # Generate new initial coordinates and reset agent
    env.reset()

    # Clear figure
    plt.clf()

    # Plot initial point
    draw.pitch()
    plt.scatter(env.x * 100, env.y * 100, color = 'C1')

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to save current frame displayed
def save(event):
    plt.savefig('img/saves/' + time.asctime().replace(':','') + '.png')

    # Indicate to the player that the play has been saved
    plt.text(50, -10, 'Saved', color='black', horizontalalignment='center', verticalalignment='center')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to correcly quit interface
def _quit():
    root.quit()
    root.destroy()


# Bind keys to functions
root.bind('<space>', shot)
root.bind('r', restart)
root.bind('s', save)

# Get full screen
root.attributes("-fullscreen", True)

# Prints first image
plt.savefig(img_dir)
img = ImageTk.PhotoImage(Image.open(img_dir))
panel = tkinter.Label(root, image=img)
panel.bind('<Button-1>', get_clicks)
panel.pack()

# Add quit button
button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

# Enter mainloop
tkinter.mainloop()