import os
import random
import pprint
import numpy as np
import cv2
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
        self.color_action = False  # True if we colored in that step, False otherwise
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
        self.color_action = False
        self.starting_pos = self.current_state

        return self.source_matrix, self.canvas, self.current_state

    def render(self):
        self.to_render = True


    def mostra(self):
        # INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
        # MOREOVER: We only change the source matrix when reset is called.
        # source matrix to draw, initialized with black pixels
        source = np.zeros((self.length, self.length, 3), dtype=np.uint8)
        # set the correct pixels to white, according to the source matrix
        source[self.source_matrix == 1] = [255, 255, 255]

        canvas = np.zeros((self.length, self.length, 3), dtype=np.uint8)
        canvas[self.canvas == 1] = [255, 255, 255]
        if self.color_action:
            canvas[self.row, self.column] = [255, 255, 0]
        else:
            canvas[self.row, self.column] = [0, 0, 255]

        print(source)
        source_img = Image.fromarray(source)
        canvas_img = Image.fromarray(canvas)
        # img = img.resize((16,16))
        source_img = np.uint8(source_img)
        canvas_img = np.uint8(canvas_img)

        height, width, ch = source_img.shape
        print(height, width, ch)
        new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8
        print(new_width, new_height)

        # Crate a new canvas with new width and height.
        source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125
        canvas_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

        # New replace the center of canvas with original image
        padding_top, padding_left = 1, 0  # 60, 10

        source_background[padding_top:padding_top + height, padding_left:padding_left + width] = source_img
        canvas_background[padding_top:padding_top + height, padding_left:padding_left + width] = canvas_img

        text1 = "Source image"
        text2 = "Canvas"
        text_color_list = np.array([255, 0, 0])
        text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
        img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                           text_color)
        img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                           text_color)

        final = cv2.hconcat((img1, img2))
        print(final.shape)
        # shape[1] is the width, it seems it needs to go first when resizing.
        final = cv2.resize(final, (final.shape[1] * 30, final.shape[0] * 30), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("./debug.png", final)
        cv2.imshow("pr", final)
        cv2.waitKey(300)


    # new_reward
    # def step(self, action):
    #     chosen_action_str = ''
    #     if self.to_render:
    #         print('source matrix:')
    #         pprint.pprint(self.source_matrix)
    #         print('canvas:')
    #         pprint.pprint(self.canvas)
    #         print('Agent position:', self.current_state)
    #
    #     if action == 0:  # down
    #         if self.row < self.length - 1:  # avoid going out of bounds
    #             self.current_state += self.length
    #         chosen_action_str = 'down'
    #     if action == 1:  # up
    #         if self.row > 0:
    #             self.current_state -= self.length
    #         chosen_action_str = 'up'
    #     if action == 2:  # move pointer left
    #         if self.column > 0:
    #             self.current_state -= 1
    #         chosen_action_str = 'left'
    #     if action == 3:  # move pointer right
    #         if self.column < self.length - 1:  # avoid going out of bounds
    #             self.current_state += 1
    #         chosen_action_str = 'right'
    #
    #     if action < 4:
    #         # obtain matrix coords for the drawer pointer
    #         self.row = self.current_state // self.length
    #         self.column = self.current_state % self.length
    #
    #     # simple start - working
    #     '''reward is -1 per step, unless the agent is in a cell that must be colored. Moreover,
    #     if we colored the correct cell, get +1 reward'''
    #     # reward = -1  # -1 per step
    #     reward = -0.01
    #     if self.source_matrix[self.row][self.column] == 1:
    #         reward = 0  # unless the agent is in a cell that must be colored
    #
    #     if action == 4:  # if we drew, we have to check whether the drawn cell is the right one
    #         if self.canvas[self.row][self.column] == 0 and self.source_matrix[self.row][self.column] == 1:
    #             reward = 0.1  # if we colored the correct cell, get +1 reward
    #         self.canvas[self.row][self.column] = 1
    #         chosen_action_str = 'color cell'
    #
    #     if self.to_render:
    #         print('chosen action:', chosen_action_str)
    #         print('-----------')
    #         self.to_render = False
    #
    #     # if all the correct cells are colored, the episode can end
    #     if np.array_equal(self.source_matrix, self.canvas):
    #         reward = 1
    #         self.done = True
    #     if self.step_count == self.max_steps:
    #         self.done = True
    #         # if np.sum(self.canvas[1] == 1) != self.length:
    #         #    reward = -100
    #     self.step_count += 1
    #
    #     return (self.source_matrix, self.canvas, self.current_state), reward, self.done
    #

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
            self.color_action = False


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
            self.color_action = True

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

        if self.done:
            cv2.destroyAllWindows()
        return (self.source_matrix, self.canvas, self.current_state), reward, self.done
