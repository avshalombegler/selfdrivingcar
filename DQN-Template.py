import numpy as np
import tensorflow as tf
from collections import deque
from PIL import ImageGrab, Image
import win32gui
from DirectInput import press_key, release_key, W, A, S, D
from GetKeys import key_check
import time
import random
import cv2
from sklearn.cluster import KMeans

# Detect the position of The window 'racer'
windows_list = []
toplist = []


def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))


win32gui.EnumWindows(enum_win, toplist)
game_hwnd = 0
for (hwnd, win_text) in windows_list:
    if "racer" in win_text:
        game_hwnd = hwnd


# ----------------------------------------------------------------------------------------------- #
def dense(x, weights, bias, activation=tf.identity):
    # Dense layer
    z = tf.matmul(x, weights) + bias
    return activation(z)


# ----------------------------------------------------------------------------------------------- #
def init_weights(shape, initializer):
    # Initialize weights for tensorflow layer
    weights = tf.Variable(
        initializer(shape),
        trainable=True,
        dtype=tf.float32
    )

    return weights


# ----------------------------------------------------------------------------------------------- #
class Network(object):
    # Q-function approximator

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=[50, 50],
                 weights_initializer=tf.initializers.glorot_uniform(),
                 bias_initializer=tf.initializers.zeros(),
                 optimizer=tf.optimizers.Adam,
                 **optimizer_kwargs):
        # Initialize weights and hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        np.random.seed(41)

        self.initialize_weights(weights_initializer, bias_initializer)
        self.optimizer = optimizer(**optimizer_kwargs)

    # ------------------------------------------------------------------------------------------- #
    def initialize_weights(self, weights_initializer, bias_initializer):
        # Initialize and store weights
        wshapes = [
            [self.input_size, self.hidden_size[0]],
            [self.hidden_size[0], self.hidden_size[1]],
            [self.hidden_size[1], self.output_size]
        ]

        bshapes = [
            [1, self.hidden_size[0]],
            [1, self.hidden_size[1]],
            [1, self.output_size]
        ]

        self.weights = [init_weights(s, weights_initializer) for s in wshapes]
        self.biases = [init_weights(s, bias_initializer) for s in bshapes]

        self.trainable_variables = self.weights + self.biases

    # ------------------------------------------------------------------------------------------- #
    def model(self, inputs):
        # Given a state vector, return the Q values of actions
        h1 = dense(inputs, self.weights[0], self.biases[0], tf.nn.relu)
        h2 = dense(h1, self.weights[1], self.biases[1], tf.nn.relu)

        out = dense(h2, self.weights[2], self.biases[2])

        return out

    # ------------------------------------------------------------------------------------------- #
    def train_step(self, inputs, targets, actions_one_hot):
        # Update weights
        with tf.GradientTape() as tape:
            qvalues = tf.squeeze(self.model(inputs))
            preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
            loss = tf.losses.mean_squared_error(targets, preds)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


# ----------------------------------------------------------------------------------------------- #
class Memory(object):
    # Memory buffer for Experience Replay

    # ------------------------------------------------------------------------------------------- #
    def __init__(self, max_size):
        # Initialize a buffer containing max_size experiences
        self.buffer = deque(maxlen=max_size)

    # ------------------------------------------------------------------------------------------- #
    def add(self, experience):
        # Add an experience to the buffer
        self.buffer.append(experience)

    # ------------------------------------------------------------------------------------------- #
    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        return [self.buffer[i] for i in index]

    # ------------------------------------------------------------------------------------------- #
    def __len__(self):
        # Interface to access buffer length
        return len(self.buffer)


# ----------------------------------------------------------------------------------------------- #
class Agent(object):
    # Deep Q-learning agent

    # ------------------------------------------------------------------------------------------- #
    def __init__(self,
                 state_space_size,
                 action_space_size,
                 target_update_freq=1000,
                 discount=0.99,
                 batch_size=32,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1 / 100000),
                 replay_memory_size=100000,
                 replay_start_size=10000):
        # Set parameters, initialize network
        self.action_space_size = action_space_size

        self.online_network = Network(state_space_size, action_space_size)
        self.target_network = Network(state_space_size, action_space_size)

        self.update_target_network()

        # training parameters
        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.memory = Memory(replay_memory_size)
        self.replay_start_size = replay_start_size
        self.experience_replay = Memory(replay_memory_size)

    # ------------------------------------------------------------------------------------------- #
    def handle_episode_start(self):
        self.last_state, self.last_action = None, None

    # ------------------------------------------------------------------------------------------- #
    def step(self, observation, training=True):
        """
        Observe state and rewards, select action.
        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """
        last_state, last_action = self.last_state, self.last_action
        last_reward = observation.reward
        state = observation.state

        action = self.policy(state, training)

        if training:
            self.steps += 1

            if last_state is not None:
                experience = {
                    "state": last_state,
                    "action": last_action,
                    "reward": last_reward,
                    "next_state": state
                }

                self.memory.add(experience)

            if self.steps > self.replay_start_size:
                self.train_network()

                if self.steps % self.target_update_freq == 0:
                    self.update_target_network()

        self.last_state = state
        self.last_action = action

        return action

    # ------------------------------------------------------------------------------------------- #
    def policy(self, state, training):
        # Epsilon-greedy policy for training, greedy policy otherwise
        explore_prob = self.max_explore - (self.steps * self.anneal_rate)
        explore = max(explore_prob, self.min_explore) > np.random.rand()
        # print(f"explore_prob = {explore_prob}")

        if training and explore:
            action = np.random.randint(self.action_space_size)
        else:
            inputs = np.expand_dims(state, 0)
            qvalues = self.online_network.model(inputs)
            action = np.squeeze(np.argmax(qvalues, axis=-1))

        return action

    # ------------------------------------------------------------------------------------------- #
    def update_target_network(self):
        # Update target network weights with current online network values
        variables = self.online_network.trainable_variables
        variables_copy = [tf.Variable(v) for v in variables]
        self.target_network.trainable_variables = variables_copy

    # ------------------------------------------------------------------------------------------- #
    def train_network(self):
        # Update online network weights
        batch = self.memory.sample(self.batch_size)
        inputs = np.array([b["state"] for b in batch])
        actions = np.array([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch])
        next_inputs = np.array([b["next_state"] for b in batch])

        actions_one_hot = np.eye(self.action_space_size)[actions]

        next_qvalues = np.squeeze(self.target_network.model(next_inputs))
        targets = rewards + self.discount * np.amax(next_qvalues, axis=-1)

        self.online_network.train_step(inputs, targets, actions_one_hot)

    # ------------------------------------------------------------------------------------------- #
    def take_action(self, action):
        if action == 0:
            self.left()
        elif action == 1:
            self.straight()
        elif action == 2:
            self.right()
        # else:
        #     self.reverse()

    def straight(self):
        press_key(W)
        release_key(A)
        release_key(D)
        release_key(S)

    def left(self):
        if random.randrange(0, 3) == 1:
            press_key(W)
        else:
            release_key(W)
        press_key(A)
        release_key(S)
        release_key(D)

    def right(self):
        if random.randrange(0, 3) == 1:
            press_key(W)
        else:
            release_key(W)
        press_key(D)
        release_key(A)
        release_key(S)

    def reverse(self):
        press_key(S)
        release_key(A)
        release_key(W)
        release_key(D)


# ----------------------------------------------------------------------------------------------- #
class Observe(object):

    # ------------------------------------------------------------------------------------------- #
    def __init__(self):
        self.current_screen = None
        self.done = False
        self.state = None
        self.reward = 0.

    # ------------------------------------------------------------------------------------------- #
    def reset_check(self):
        bbox = win32gui.GetWindowRect(game_hwnd)
        image = np.array(ImageGrab.grab(bbox))

        # convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        im = Image.fromarray(rgb_image)

        # crop
        mirror_l = im.crop((400, 435, 430, 455))  # (left, top, right, bottom)
        mirror_r = im.crop((600, 435, 630, 455))  # (left, top, right, bottom)

        # resize(x10)
        new_size = (mirror_l.size[0] * 10, mirror_l.size[1] * 10)
        resized_mirror_l = mirror_l.resize(new_size)
        resized_mirror_r = mirror_r.resize(new_size)
        resized_mirror_l = np.array(resized_mirror_l)
        resized_mirror_r = np.array(resized_mirror_r)

        # convert back to np array
        mirror_l = np.array(mirror_l)
        mirror_r = np.array(mirror_r)

        # find dominant color
        dominant_color_l = self.find_dominant_color(mirror_l)
        dominant_color_r = self.find_dominant_color(mirror_r)
        # print(dominant_color_l)
        # print(dominant_color_r)

        # l_b = np.array([50, 80, 80])
        # u_b = np.array([80, 110, 110])
        #
        # if l_b < dominant_color_l < u_b and \
        #         l_b < dominant_color_r < u_b:
        #     print("FUCK YES!")

        if 50 < dominant_color_l[0] < 80 and \
                80 < dominant_color_l[1] < 110 and \
                80 < dominant_color_l[2] < 110 and \
                50 < dominant_color_r[0] < 80 and \
                80 < dominant_color_r[1] < 110 and \
                80 < dominant_color_r[2] < 110:
            self.reset()

        cv2.imshow('mirror_l', resized_mirror_l)
        cv2.imshow('mirror_r', resized_mirror_r)

    # ------------------------------------------------------------------------------------------- #
    def find_dominant_color(self, image):
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=1)
        clt.fit(image)
        dominant_color = clt.cluster_centers_[0]

        return dominant_color

    # ------------------------------------------------------------------------------------------- #
    def reset(self):
        shift = 0x2A
        f = 0x21
        # r = 0x13

        release_key(W)
        release_key(A)
        release_key(D)
        press_key(shift)
        press_key(f)
        release_key(shift)
        release_key(f)

        self.reward -= 1.
        self.current_screen = None
        self.done = True

        for i in list(range(2))[::-1]:
            print(i + 1)
            time.sleep(1)

    # ------------------------------------------------------------------------------------------- #
    def just_starting(self):
        return self.current_screen is None

    # ------------------------------------------------------------------------------------------- #
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = tf.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2

            return s2 - s1

    # ------------------------------------------------------------------------------------------- #
    def get_processed_screen(self):
        bbox = win32gui.GetWindowRect(game_hwnd)
        screen = np.array(ImageGrab.grab(bbox))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        # screen = np.array(ImageGrab.grab(bbox=(0, 35, 1024, 800)))
        # screen = screen.transpose((1, 0, 2))  # default is (h,w,c), transpose is (w,h,c) for pytorch
        screen = self.crop_screen(screen)
        # cv2.imshow('screen', screen)

        return self.transform_screen_data(screen)

    # ------------------------------------------------------------------------------------------- #
    def crop_screen(self, screen):
        screen_height = screen.shape[0]

        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[top:bottom, :, :]

        return screen

    # ------------------------------------------------------------------------------------------- #
    def transform_screen_data(self, screen):
        """
        # print(f"screen size after crop: {screen.shape}")
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = tf.convert_to_tensor(screen)

        resize = tf.keras.preprocessing.image.array_to_img(screen, data_format=None, scale=True, dtype=None)
        # resize.thumbnail((30, 100), Image.ANTIALIAS)
        resize.thumbnail((screen.shape[1] * 0.15, screen.shape[0] * 0.15), Image.ANTIALIAS)
        resize = tf.keras.preprocessing.image.img_to_array(resize, data_format=None, dtype=None)
        """
        resize = cv2.resize(screen,
                            dsize=(int(screen.shape[1] * 0.2), int(screen.shape[0] * 0.2)),
                            interpolation=cv2.INTER_CUBIC)
        cv2.imshow('resize', resize)
        # print(f"screen size after resize: {resize.shape}")
        # resize = np.expand_dims(resize, axis=0)
        # print(f"screen size after resize + batch dimension: {resize.shape}")

        return resize


# ----------------------------------------------------------------------------------------------- #
def main():
    # for i in list(range(4))[::-1]:
    #     print(i + 1)
    #     time.sleep(1)

    episodes = 25000

    agent = Agent(63 * 206 * 3, 4)
    # (state_space_size, action_space_size)
    # (state_space_size = image size[h*w*c] -> first layer input)
    # (action_space_size = last layer output -> action[0,...,3])

    observation = Observe()
    agent.handle_episode_start()

    paused = False
    for _ in range(episodes):
        observation.done = False
        agent.steps = 0
        while not observation.done:

            if not paused:
                observation.state = observation.get_state()  # maybe move to Agent.step()
                action = agent.step(observation)
                agent.take_action(action)
                observation.reset_check()

                # reward += 1 if X steps were made without ending an episode
                if agent.steps % 500 == 0:
                    observation.reward += 1.
                print(f"steps = {agent.steps}")

            keys = key_check()
            if 'T' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    release_key(W)
                    release_key(A)
                    release_key(D)
                    paused = True
                    time.sleep(1)

            # press 'q' to close the window
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


main()
