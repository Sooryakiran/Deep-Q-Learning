import my_little_dqn as dqn

GAME = "LunarLander-v2"

# Hyperparameters
BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
DQN_TARGET_SYNC = 5000
BATCH_SIZE = 4096
GAMMA = 0.99
INITIAL_EPSILON = 0.7
FINAL_EPSILON = 0.02
EPSILON_DECAY_FRAMES = 10e3
SOLVED_POINTS = 200
OBSERVATION_SHAPE = 8
ACTION_SPACE = 3

a = dqn.DQNagent(game = GAME, \
                 buffer_size = BUFFER_SIZE, \
                 learning_rate = LEARNING_RATE, \
                 dqn_target_sync = DQN_TARGET_SYNC, \
                 batch_size = BATCH_SIZE, \
                 gamma = GAMMA, \
                 initial_epsilon= INITIAL_EPSILON, \
                 final_epsilon = FINAL_EPSILON, \
                 epsilon_decay_frames = EPSILON_DECAY_FRAMES, \
                 solved_points = SOLVED_POINTS, \
                 observation_shape = 8, \
                 action_space = 3, \
                 log_dir="logdir", \
                 video_dir= "videos")

a.initialise()
a.learn()