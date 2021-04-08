import numpy as np
import pandas as pd
import scipy
import json
import glob

from tqdm import tqdm

# Loading all files
all_files = glob.glob('data/json/*.json')
all_games = []
for f in tqdm(all_files):
    all_games.append(json.load(open(f)))

# Data processing pipeline
## Iterate all games and return table with all events from every game
data_li = []
for game in tqdm(all_games):
    # Save flags about previous action
    prev_forward_action = False
    prev_was_cross = False
    prev_was_pass = False
    prev_was_dribble = False

    # Get all game events
    events = game['events']
    # Process all events
    for i in range(len(game['events'])):
        # Filter only relevant events for the simulator
        if events[i]['type']['displayName'] in ['CrossNotClaimed','Goal','MissedShots','OffsidePass','Pass','SavedShot','ShotOnPost','TakeOn']:
            # Retrieve base information
            attr_type = events[i]['type']['displayName']
            attr_player = events[i]['playerId']
            attr_team = events[i]['teamId']
            attr_success = events[i]['outcomeType']['value']
            attr_x = events[i]['x'] / 100 # Normalize between 0-1
            attr_y = events[i]['y'] / 100

            # Retrieve information that is only available in certain events
            if 'endX' in events[i]:
                attr_endX = events[i]['endX'] / 100
                attr_endY = events[i]['endY'] / 100
            else:
                attr_endX = events[i]['x'] / 100
                attr_endY = events[i]['y'] / 100
            if 'isShot' in events[i].keys():
                attr_isShot = True
            else:
                attr_isShot = False
            if 'isGoal' in events[i].keys():
                attr_isGoal = True
            else:
                attr_isGoal = False

            # Retrieve qualifier information about the event
            qualifiers = []
            for qualifier in events[i]['qualifiers']:
                qualifiers.append(qualifier['type']['displayName'])
            attr_rightFoot = 'RightFoot' in qualifiers
            attr_leftFoot = 'LeftFoot' in qualifiers
            attr_header = 'Head' in qualifiers
            attr_headPass = 'HeadPass' in qualifiers
            attr_blocked = 'Blocked' in qualifiers
            attr_blockedCross = 'BlockedCross' in qualifiers
            attr_chipped = 'Chipped' in qualifiers
            attr_cross = 'Cross' in qualifiers
            attr_layOff = 'LayOff' in qualifiers
            attr_regularPlay = 'RegularPlay' in qualifiers
    
            attr_cornerTaken = 'CornerTaken' in qualifiers
            attr_directFK = 'DirectFreekick' in qualifiers
            attr_FK = 'FreekickTaken' in qualifiers
            attr_corner = 'FromCorner' in qualifiers
            attr_goalKick = 'GoalKick' in qualifiers
            attr_indirectFK = 'IndirectFreekickTaken' in qualifiers
            attr_ownGoal = 'OwnGoal' in qualifiers
            attr_setPiece = 'SetPiece' in qualifiers
            attr_throwIn = 'ThrowIn' in qualifiers
            
            # If not a set piece
            if not attr_cornerTaken and not attr_directFK and not attr_FK and not attr_corner and not attr_goalKick and not attr_indirectFK and not attr_ownGoal and not attr_setPiece and not attr_throwIn:
                # Retrieve information for rebound
                attr_xrebound, attr_yrebound = attr_x, attr_y
                attr_rebound = False
                if not attr_success:
                    if not (type(attr_endX) is str or type(attr_endY) is str):
                        if (attr_endX > 0) and (attr_endY > 0) and (attr_endX < 100) and (attr_endY < 100):
                            for j in range(1,5):
                                if events[i+j]['teamId'] == attr_team:
                                    attr_rebound += 1
                                    if (attr_xrebound == attr_x) and (attr_yrebound == attr_y):
                                        attr_xrebound = events[i+j]['x'] / 100
                                        attr_yrebound = events[i+j]['y'] / 100
                            attr_rebound = attr_rebound >= 1
                
                # Retrieve information about the dribble
                attr_xdribble, attr_ydribble = attr_x, attr_y
                attr_toFoul = False
                if (attr_type == 'TakeOn') and attr_success:
                    for j in range(1,5):
                        if events[i+j]['teamId'] == attr_team:
                            attr_xdribble = events[i+j]['x'] / 100
                            attr_ydribble = events[i+j]['y'] / 100
                            break
                    for j in range(1,3):
                        if events[i+j]['type']['displayName'] == 'Foul':
                            attr_toFoul = True
                elif (attr_type == 'TakeOn'):
                    if j == 4 and attr_xdribble == attr_x:
                        attr_xdribble = events[i+2]['x'] / 100
                        attr_ydribble = events[i+2]['y'] / 100
                
                # Retrieve information on if the shot ended up in a corner
                attr_toCorner = False
                if (attr_type in ['SavedShot', 'MissedShots', 'ShotOnPost']):
                    for j in range(1,2):
                        if events[i]['type']['displayName'] == 'CornerAwarded':
                            attr_toCorner = True

                # Append to list in order to concat
                data_li.append([attr_type, attr_player, attr_team, attr_success, attr_x, attr_y, attr_endX, attr_endY, attr_isShot, attr_isGoal, attr_rightFoot, attr_leftFoot, attr_header, attr_headPass, attr_blocked, attr_blockedCross, attr_chipped, attr_cross, attr_layOff, attr_regularPlay, attr_rebound, attr_xrebound, attr_yrebound, attr_xdribble, attr_ydribble, attr_toFoul, attr_toCorner, prev_forward_action, prev_was_cross, prev_was_pass, prev_was_dribble])

                prev_forward_action = attr_x < attr_endX
                prev_was_cross = attr_cross
                prev_was_pass = events[i]['type']['displayName'] == 'Pass'
                prev_was_dribble = events[i]['type']['displayName'] == 'Dribble'

# Converting to data frame
df = pd.DataFrame(data_li, columns=['type', 'player', 'team', 'success', 'x', 'y', 'endX', 'endY', 'isShot', 'isGoal', 'rightFoot', 'leftFoot', 'header', 'headPass', 'blocked', 'blockedCross', 'chipped', 'cross', 'layOff', 'regularPlay', 'rebound', 'xrebound', 'yrebound', 'xdribble', 'ydribble', 'toFoul', 'toCorner', 'prevForwardAct', 'prevCross', 'prevPass', 'prevDribble'])

# Adding polar coordinate system to the variables
df['r'] = np.sqrt((df.endX - df.x) ** 2 + (df.endY - df.y) ** 2)
df['a'] = np.arctan2(df.endY - df.y, df.endX - df.x) / (2 * np.pi) + 0.5
df['rebound_r'] = np.sqrt((df.xrebound - df.x) ** 2 + (df.yrebound - df.y) ** 2)
df['rebound_a'] = np.arctan2(df.yrebound - df.y, df.xrebound - df.x) / (2 * np.pi) + 0.5
df['dribble_r'] = np.sqrt((df.xdribble - df.x) ** 2 + (df.ydribble - df.y) ** 2)
df['dribble_a'] = np.arctan2(df.ydribble - df.y, df.xdribble - df.x) / (2 * np.pi) + 0.5

# Save the DataFrame
df.to_csv('data/formated_data.csv')