import tensorflow as tf
import numpy as np
import gym
from gym.wrappers.monitoring import video_recorder as vd
import random
import matplotlib.pyplot as plt 
import sys
import collections

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
GAME = "LunarLander-v2"

# Hyperparameters
BUFFER_SIZE = 100000
LEARNING_RATE = 1e-4
DQN_TARGET_SYNC = 5000
BATCH_SIZE = 4096
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.02
EPSILON_DECAY_FRAMES = 10e4
SOLVED_POINTS = 200
OBSERVATION_SHAPE = 8
ACTION_SPACE = 3

# environment = gym.make(GAME)

# Experience = collections.namedtuple('Experience', field_names = ["state", "action", "reward", "done", "next_state"])

class ReplayMemory:
  def __init__(self,buffer_size = BUFFER_SIZE):
    self.buffer = collections.deque(maxlen = buffer_size)
  def __len__(self):
    return len(self.buffer)
  def append(self, new_experience):
    self.buffer.append(new_experience)
  def sample(self, batch_size = BATCH_SIZE):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states)


class DQNagent:
    def __init__(self, game, buffer_size = 1e4 , learning_rate = 1e-3, dqn_target_sync = 10e3, batch_size = 4096 , gamma = 0.99, initial_epsilon = 1.0, \
                 final_epsilon = 0.02, epsilon_decay_frames = 1e4, solved_points = 200, observation_shape = 8, action_space = 3, log_dir = "log_dir", video_dir = "video"):
        
        self.game = game
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.dqn_target_sync = dqn_target_sync
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_frames = epsilon_decay_frames
        self.solved_points = solved_points
        self.observation_shape = observation_shape
        self.action_space = action_space
        
        self.environment = gym.make(self.game)
        self.Experience = collections.namedtuple('Experience', field_names = ["state", "action", "reward", "done", "next_state"])

        self.epsilon = self.initial_epsilon
        self.memory = ReplayMemory()
        self.average_reward = - 1e10
        print("Building graphs...")

        self.graph = tf.Graph()
        self.log_dir = log_dir
        self.video_dir = video_dir + "/"
        with self.graph.as_default():

            # placeholders
            self.tf_observation_train = tf.placeholder(dtype=tf.float32, shape = [self.batch_size,self.observation_shape], \
                                                       name = "train_placeholder_observation")
            self.tf_next_state = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, self.observation_shape], \
                                                name = "train_placeholder_next_state")
            self.tf_rewards = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, 1], \
                                            name = "train_placeholder_rewards")
            self.current_action = tf.placeholder(dtype = tf.int32)
        
            action_one_hot = tf.one_hot(self.current_action, depth = self.action_space)

            # train ops
            self.predicted_q_values = self.network(self.tf_observation_train, "dqn")
            self.next_state_q_values = tf.stop_gradient(self.network(self.tf_next_state,"target"))
            
            self.real_q_values = tf.reduce_max((self.next_state_q_values)*action_one_hot) * self.gamma + self.tf_rewards
            self.real_q_values = tf.reshape(self.real_q_values,shape= (-1,))
            self.predicted_required = tf.reduce_max(self.predicted_q_values, reduction_indices=[1])
            self.loss = tf.reduce_mean(tf.losses.huber_loss(labels = self.real_q_values, predictions = self.predicted_required) )
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, momentum = 0.3)
            self.train_operation = self.optimizer.minimize(self.loss)
            
            # prediction ops
            self.tf_observation_once = tf.placeholder(dtype=tf.float32, shape = [1,self.observation_shape], \
                                                      name = "predict_placeholder_observation")
            self.q_for_action = self.network(self.tf_observation_once,"dqn")

            # target sync ops
            self.dqn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "dqn")
            self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "target")
            self.update_target = [target_var.assign(dqn_var) for dqn_var, target_var in zip(self.dqn_vars, self.target_vars)]


            # summary
            self.summary_loss_placeholder = tf.placeholder(dtype = tf.float32)
            self.summary_reward_placeholder = tf.placeholder(dtype = tf.float32)
            tf.summary.scalar('Loss' , self.summary_loss_placeholder)
            tf.summary.scalar('Rewards', self.summary_reward_placeholder)
            self.merged_summary = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_dir)
            
            self.saver = tf.train.Saver()

    def network(self, inputs, name):
        with tf.variable_scope(name):
            x = tf.layers.dense(inputs,128,activation = tf.nn.relu, name= "fc1", reuse=tf.AUTO_REUSE)
            x = tf.layers.dense(x,64,activation = tf.nn.relu, name= "fc2", reuse=tf.AUTO_REUSE)
            x = tf.layers.dense(x,32,activation = tf.nn.relu, name= "fc3", reuse=tf.AUTO_REUSE)
            x = tf.layers.dense(x,ACTION_SPACE,activation = None, name= "fc4", reuse=tf.AUTO_REUSE)
        return x

    def initialise(self):
        current_state = self.environment.reset()
        frame_id = 0
        print('Populating memory...')

        total_rewards = 0 
        total_episodes = 0
        
        while(self.memory.__len__() < self.buffer_size):
            print("\rCompleted %f" %(self.memory.__len__()*100/self.buffer_size))
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            frame_id += 1
            action = self.environment.action_space.sample()
            next_state, reward, done, _ = self.environment.step(action)
            total_rewards += reward
            exp = self.Experience(current_state, action, reward, done, next_state)
            self.memory.append(exp)
            if done :
                total_episodes += 1
                current_state = self.environment.reset()
        self.average_reward = total_rewards/total_episodes
        print("\nAverage_reward: %f" % self.average_reward)
    
    def learn(self):
        step = 0
        self.episode = 0 
        # rewards = []
        # losses = []
        current_state = self.environment.reset()
        print('Learning process will begin soon...')
        with tf.Session(graph= self.graph) as sess:
            tf.global_variables_initializer().run()
            try:
                self.saver.restore(sess, "model/model.ckpt")
            except:
                print("No model restored")

            moving_average_reward = []
            num_steps = 0

            while(self.average_reward < self.solved_points):
                done = False
                self.episode += 1
                total_reward = 0
                total_loss = 0
                num_steps = 0 

                if self.episode % 100 == 1: 
                    video_recorder = None
                    video_path = self.video_dir + "video_" + "episode_" + str(self.episode) + ".mp4"
                    video_recorder = vd.VideoRecorder(
                        self.environment, video_path, enabled=video_path is not None)

                while not done:
                    step += 1
                    num_steps +=1
                    epsilon = 1
                    if step < self.epsilon_decay_frames :
                        epsilon = self.initial_epsilon*(1.0-step/self.epsilon_decay_frames)
                    else:
                        epsilon = 0
                    if epsilon < self.final_epsilon:
                        epsilon = self.final_epsilon
                    
                    random_number = random.uniform(0,1)
                    if random_number < epsilon :
                        # choose random action
                        action = self.environment.action_space.sample()
                    
                    else:
                        # use network
                        input_observation_tensor = np.reshape(current_state,(1, self.observation_shape))
                        feed_dict = {self.tf_observation_once : input_observation_tensor}
                        q_values = sess.run([self.q_for_action], feed_dict = feed_dict)
                        action = np.argmax(q_values)

                    next_state, reward, done, info = self.environment.step(action)

                    if self.episode % 100 == 1:
                        self.environment.unwrapped.render()
                        video_recorder.capture_frame()
                        if done :
                            video_recorder.close()
                    
                    exp = self.Experience(current_state, action, reward, done, next_state)
                    current_state = next_state
                    self.memory.append(exp)
                    total_reward += reward 

                    observation_batch = self.memory.sample()
                    batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = observation_batch
                    batch_rewards = np.reshape(batch_rewards, (-1,1))
      
                    feed_dict = {self.tf_observation_train: batch_states, \
                                 self.tf_next_state: batch_next_states, \
                                 self.tf_rewards: batch_rewards, \
                                 self.current_action: batch_actions}

                    for i in range(1):
                        loss_, train_operation_ = sess.run([self.loss,self.train_operation],feed_dict = feed_dict)
                    total_loss += loss_

                    if step%self.dqn_target_sync == 1:
                        sess.run(self.update_target)
                        save_path = self.saver.save(sess, "model/model.ckpt")
                        print("Target Updated")
                current_state = self.environment.reset()
                moving_average_reward.append(total_reward)
                if len(moving_average_reward) >200:
                    moving_average_reward.pop(0)

                smoothened = (sum(moving_average_reward)/len(moving_average_reward))
                smoothened_loss = (total_loss/num_steps)
                if self.episode% 5 == 0:
                    feed_dict = { self.summary_loss_placeholder : smoothened_loss, \
                                  self.summary_reward_placeholder : smoothened }
                    
                    summary = sess.run(self.merged_summary, feed_dict= feed_dict)
                    self.writer.add_summary(summary, self.episode)
                    self.writer.flush()
                print("[%d] Episode %d \t\t Reward %f\tEpsilon %0.4f\tAverage:%f\t Average loss: %f" %(step,self.episode,total_reward,epsilon,smoothened, smoothened_loss))

