import numpy as np
import gym
import gym_simpletetris

"""
    References: https://github.com/uiucanh/tetris
"""

def get_feature_len():
    env = gym.make('SimpleTetris-v0')
    obs = env.reset()
    return len(get_features(obs, 0, [], None, env))

def get_features(obs, lines_cleared, rows_cleared, piece_coords, env):
    obs = np.transpose(obs)
    obs = np.fliplr(obs)

    # column heights 
    peaks = get_peaks(obs) # [list]
    highest_col = np.max(peaks)


    # Total number of holes
    holes = get_holes(obs, peaks)
    
    # Row transitions
    row_transitions = get_row_transition(obs, highest_col)


    # Columns transitions
    col_transitions = get_col_transition(obs, peaks)


    # Abs height differences between consecutive cols [list]
    bumpiness = get_bumpiness(peaks)

    wells_sum = get_wells(peaks)

    landing_height = get_landing_height(obs, env)

    eroded_piece_cells = get_eroded_piece_cells(lines_cleared, rows_cleared, piece_coords)

    feature_vector = np.array([
        landing_height, eroded_piece_cells, row_transitions, col_transitions, holes, 
        wells_sum, *peaks, *bumpiness, highest_col
    ])

    # # Dellacherie features
    # feature_vector = np.array([
    #     landing_height, eroded_piece_cells, row_transitions, col_transitions, holes, 
    #     wells_sum
    # ])
    
    # # Bertsekas features
    # feature_vector = np.array([
    #     holes, *peaks, *bumpiness, highest_col
    # ])

    # minimised features
    # feature_vector = np.array([
    #     holes, sum(peaks), sum(bumpiness), lines_cleared
    # ])
    return feature_vector


def get_peaks(obs):
    peaks = np.array([])
    for col in range(obs.shape[1]):
        if 1 in obs[:, col]:
            p = obs.shape[0] - np.argmax(obs[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)
    return peaks


def get_holes(obs, peaks):
    # Count from peaks to bottom
    holes = []

    for col in range(obs.shape[1]):
        start = -peaks[col]
        # If there's no holes i.e. no blocks on that column
        if start == 0:
            holes.append(0)
        else:
            holes.append(np.count_nonzero(obs[int(start):, col] == 0))
    return sum(holes)


# def get_bumpiness(peaks):
#     sum = 0
#     for i in range(len(peaks) - 1):
#         sum += np.abs(peaks[i] - peaks[i + 1])
#     return sum


def get_bumpiness(peaks):
    differences = []
    for i in range(len(peaks) - 1):
        differences.append(np.abs(peaks[i] - peaks[i + 1]))
    return np.array(differences)


def get_row_transition(obs, highest_col):
    sum = 0
    # From highest peak to bottom
    for row in range(int(obs.shape[0] - highest_col), obs.shape[0]):
        for col in range(1, obs.shape[1]):
            if obs[row, col] != obs[row, col - 1]:
                sum += 1
    return sum


def get_col_transition(obs, peaks):
    sum = 0
    for col in range(obs.shape[1]):
        if peaks[col] <= 1:
            continue
        for row in range(int(obs.shape[0] - peaks[col]), obs.shape[0] - 1):
            if obs[row, col] != obs[row + 1, col]:
                sum += 1
    return sum


def get_landing_height(future_board, env):
    highest = 0

    original_board = np.transpose(env.engine.board)
    original_board = np.fliplr(original_board)

    new_board = future_board - original_board
    for col_index in range(new_board.shape[1]):
        counter = 0
        column = new_board[:, col_index]
        
        for number in column:
            if number == 1:
                counter = len(column) - counter
                if counter >= highest:
                    highest = counter
                break
            counter += 1
    return highest

# def get_landing_height(landing_piece_coords, env):
#     if landing_piece_coords is None:
#         return 0
    
#     height = np.Inf
#     for tuple in landing_piece_coords:
#         height = min(height, tuple[1])

#     return env.engine.board.shape[1] - height

def get_wells(peaks):
    wells = []
    sum = 0
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0]
            w = w if w > 0 else 0
            wells.append(w)
        elif i == len(peaks) - 1:
            w = peaks[-2] - peaks[-1]
            w = w if w > 0 else 0
            wells.append(w)
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            w1 = w1 if w1 > 0 else 0
            w2 = w2 if w2 > 0 else 0
            w = w1 if w1 >= w2 else w2
            wells.append(w)
    
    wells = [int(number) for number in wells]

    for number in wells:
        count = 0
        for i in range(number + 1):
            count += i
            sum += count

    return sum


def get_eroded_piece_cells(lines_cleared, rows_cleared, piece_coords):
    if piece_coords is None:
        return 0
    
    counter = 0
    for number in rows_cleared:
        for cell in piece_coords:
            if number == cell[1]:
                counter += 1
                
    return lines_cleared * counter