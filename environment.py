import os
import random
import pprint
import numpy as np

from PIL import Image


class Environment:
    def __init__(self, random_starting_pos=False, random_horizontal_line=False):
        self.length = 4
        self.actions = np.array([0,1,2,3,4])  # 0 move down, 1 move up, 2 move left, 3 move right, 4 color the cell
        self.source_matrix = np.zeros((self.length, self.length))
        self.canvas = np.zeros((self.length, self.length))
        self.current_state = 0
        self.row = 0
        self.column = 0

        self.step_count = 0
        self.done = False

        self.max_state = (self.length ** 2) - 1
        self.num_states = 2 * (self.length ** 2) + 1
        self.max_steps = 50

        self.num_actions = len(self.actions)

        self.random_starting_pos = random_starting_pos
        self.random_horizontal_line = random_horizontal_line

        self.to_render = False

        self.starting_pos = self.current_state
        # if self.random_starting_pos:
        #     self.current_state = random.randint(0, self.max_state)
        #     self.row = self.current_state // self.length
        #     self.column = self.current_state % self.length

    def reset(self):
        self.source_matrix = np.zeros((self.length, self.length))
        if self.random_horizontal_line:
            self.source_matrix[random.randint(0, self.length - 1)] = 1  # randomize the horizontal line to draw
        else:
            self.source_matrix[1] = 1  # draw a line in the second row
        self.canvas = np.zeros((self.length, self.length))
        self.current_state = 0
        self.row = 0
        self.column = 0

        if self.random_starting_pos:
            self.current_state = random.randint(0, self.max_state)
            self.row = self.current_state // self.length
            self.column = self.current_state % self.length

        self.step_count = 0
        self.done = False

        self.starting_pos = self.current_state

        return self.source_matrix, self.canvas, self.current_state

    def render(self):
        self.to_render = True

    def step(self, action):
        chosen_action_str = ''
        if self.to_render:
            print('source matrix:')
            pprint.pprint(self.source_matrix)
            print('canvas:')
            pprint.pprint(self.canvas)
            print('Agent position:', self.current_state)


        if action == 0:  # down
            if self.row < self.length - 1:  # avoid going out of bounds
                self.current_state += self.length
            chosen_action_str = 'down'
        if action == 1:  # up
            if self.row > 0:
                self.current_state -= self.length
            chosen_action_str = 'up'
        if action == 2:  # move pointer left
            if self.column > 0:
                self.current_state -= 1
            chosen_action_str = 'left'
        if action == 3:  # move pointer right
            if self.column < self.length - 1:  # avoid going out of bounds
                self.current_state += 1
            chosen_action_str = 'right'

        if action < 4:
            # obtain matrix coords for the drawer pointer
            self.row = self.current_state // self.length
            self.column = self.current_state % self.length


        # simple start - working
        '''reward is -1 per step, unless the agent is in a cell that must be colored. Moreover,
        if we colored the correct cell, get +1 reward'''
        reward = -1  # -1 per step

        if self.source_matrix[self.row][self.column] == 1:
            reward = 0  # unless the agent is in a cell that must be colored

        if action == 4:  # if we drew, we have to check whether the drawn cell is the right one
            if self.canvas[self.row][self.column] == 0 and self.source_matrix[self.row][self.column] == 1:
                reward = 1  # if we colored the correct cell, get +1 reward
            self.canvas[self.row][self.column] = 1
            chosen_action_str = 'color cell'

        if self.to_render:
            print('chosen action:', chosen_action_str)
            print('-----------')
            self.to_render = False

        # if all the correct cells are colored, the episode can end
        if np.array_equal(self.source_matrix, self.canvas):
            #reward = 100
            self.done = True
        if self.step_count == self.max_steps:
            self.done = True
            #if np.sum(self.canvas[1] == 1) != self.length:
            #    reward = -100
        self.step_count += 1

        return (self.source_matrix, self.canvas, self.current_state), reward, self.done
