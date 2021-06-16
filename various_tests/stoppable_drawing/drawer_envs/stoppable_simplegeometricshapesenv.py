import cv2
import random
import numpy as np

from PIL import Image
from pprint import pprint


class SimpleGeometricShapesEnv:
    def __init__(self, side_length: int, max_steps, random_starting_pos=False, random_missing_pixel=False):  # , start_on_line=False):
        """
        :param side_length: source and canvas matrices side lengths. An odd size is preferable
        :param max_steps: max number of steps for the simulation to run
        :param random_starting_pos: whether the agent must start in a random position
        """
        self.length = side_length
        self.actions = np.array([0, 1, 2, 3, 4])  # 0 move down, 1 move up, 2 move left, 3 move right, 4 color the cell
        self.source_matrix = np.zeros((self.length, self.length), dtype=np.float32)
        self.canvas = np.zeros((self.length, self.length), dtype=np.float32)
        self.canvas_old = np.zeros((self.length, self.length), dtype=np.float32)
        self.current_state = 0
        self.row = 0
        self.column = 0

        self.step_count = 0
        self.done = False

        self.max_state = (self.length ** 2) - 1
        self.num_states = 2 * (self.length ** 2) + 2  # last + 2 is for x and y coords of agent's position
        # self.num_states = (self.length ** 2) + 2  # last + 2 is for x and y coords of agent's position
        self.max_steps = max_steps

        self.num_actions = len(self.actions)

        self.random_starting_pos = random_starting_pos
        # self.start_on_line=start_on_line
        # if self.start_on_line:
        #     self.random_starting_pos = False

        self.show_debug_info = False
        self.color_action = False  # True if we colored in that step, False otherwise
        self.starting_pos = self.current_state

        self.shapes_list = [self.__create_square, self.__create_circle, self.__create_triangle, self.__create_diamond]
        self.shapes_list_backup = self.shapes_list.copy()
        self.num_completed = 0
        self.shape_n = 0
        self.random_missing_pixel = random_missing_pixel

    def print_debug(self):
        self.show_debug_info = True

    def show_q_values(side_length: int, x, y, values):
        # moving the position by 1 is necessary because we will then need to show extra possible movements TODO: riscrivi
        x += 1
        y += 1
        # TODO: INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
        # MOREOVER: We only change the source matrix when reset is called.
        # initialize the matrix that will contain qvalues
        # +2 because we need to be able to rappresent every movement even when the agent is at the matrix boundaries
        scores_matrix = np.zeros((side_length + 2, side_length + 2, 3), dtype=np.uint8)
        positions = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]

        # scale values in [0,255]
        values = values / values.max() * 255
        for i in range(5):
            x_offset = positions[i][0]
            y_offset = positions[i][1]
            scores_matrix[x + x_offset][y + y_offset] = [128, values[i], 0]

        scores_matrix_img = Image.fromarray(scores_matrix)
        scores_matrix_img = np.uint8(scores_matrix_img)

        height, width, ch = scores_matrix_img.shape
        new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

        # Crate a new canvas with new width and height.
        source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

        # New replace the center of canvas with original image
        padding_top, padding_left = 1, 0  # 60, 10

        source_background[padding_top:padding_top + height, padding_left:padding_left + width] = scores_matrix_img

        text1 = "Source image"
        text2 = "Canvas"
        text_color_list = np.array([255, 0, 0])
        text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
        # img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                   text_color)
        # img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                   text_color)

        # shape[1] is the width, it seems it needs to go first when resizing.
        final = cv2.resize(source_background, (source_background.shape[1] * 30, source_background.shape[0] * 30),
                           interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("./debug.png", final)
        cv2.imshow("pr", final)  # this prevents code from running with wandb after first sweep
        cv2.waitKey(300)

    def render(self, show_q_values=False, side_length: int = None, x: int = None, y: int = None, values = None):
        if show_q_values:
            assert x is not None
            assert y is not None
            assert values is not None
            # moving the position by 1 is necessary because we will then need to show extra possible movements TODO: riscrivi
            x += 1
            y += 1
            # TODO: INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
            # MOREOVER: We only change the source matrix when reset is called.
            # initialize the matrix that will contain qvalues
            # +2 because we need to be able to rappresent every movement even when the agent is at the matrix boundaries
            scores_matrix = np.zeros((side_length + 2, side_length + 2, 3), dtype=np.uint8)
            positions = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]

            # scale values in [0,255]
            values = values / values.max() * 255
            for i in range(5):
                x_offset = positions[i][0]
                y_offset = positions[i][1]
                scores_matrix[x + x_offset][y + y_offset] = [128, values[i], 0]

            scores_matrix_img = Image.fromarray(scores_matrix)
            scores_matrix_img = np.uint8(scores_matrix_img)

            height, width, ch = scores_matrix_img.shape
            new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

            # Crate a new canvas with new width and height.
            source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

            # New replace the center of canvas with original image
            padding_top, padding_left = 1, 0  # 60, 10

            source_background[padding_top:padding_top + height, padding_left:padding_left + width] = scores_matrix_img

            text1 = "Source image"
            text2 = "Canvas"
            text_color_list = np.array([255, 0, 0])
            text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
            # img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            #                   text_color)
            # img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            #                   text_color)

            # shape[1] is the width, it seems it needs to go first when resizing.
            final = cv2.resize(source_background, (source_background.shape[1] * 30, source_background.shape[0] * 30),
                               interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite("./debug.png", final)
            cv2.imshow("pr", final)  # this prevents code from running with wandb after first sweep
            cv2.waitKey(300)

        # TODO: INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
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

        source_img = Image.fromarray(source)
        canvas_img = Image.fromarray(canvas)
        # img = img.resize((16,16))
        source_img = np.uint8(source_img)
        canvas_img = np.uint8(canvas_img)

        height, width, ch = source_img.shape
        new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

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
        # img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                   text_color)
        # img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                   text_color)

        img1 = source_background
        img2 = canvas_background
        final = cv2.hconcat((img1, img2))
        # shape[1] is the width, it seems it needs to go first when resizing.
        final = cv2.resize(final, (final.shape[1] * 30, final.shape[0] * 30), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("./debug.png", final)
        cv2.imshow("pr", final)  # this prevents code from running with wandb after first sweep
        cv2.waitKey(300)

    def render_with_scores_vis(self, show_q_values=False, side_length: int = None, x: int = None, y: int = None, values = None):
        if show_q_values:
            assert x is not None
            assert y is not None
            assert values is not None
            # moving the position by 1 is necessary because we will then need to show extra possible movements TODO: riscrivi
            x += 1
            y += 1
            # TODO: INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
            # MOREOVER: We only change the source matrix when reset is called.
            # initialize the matrix that will contain qvalues
            # +2 because we need to be able to rappresent every movement even when the agent is at the matrix boundaries
            scores_matrix = np.zeros((side_length + 2, side_length + 2, 3), dtype=np.uint8)
            positions = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]

            # scale values in [0,255]
            values = values / values.max() * 255
            for i in range(5):
                x_offset = positions[i][0]
                y_offset = positions[i][1]
                scores_matrix[x + x_offset][y + y_offset] = [128, values[i], 0]

            scores_matrix_img = Image.fromarray(scores_matrix)
            scores_matrix_img = np.uint8(scores_matrix_img)

            height, width, ch = scores_matrix_img.shape
            new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

            # Crate a new canvas with new width and height.
            source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

            # New replace the center of canvas with original image
            padding_top, padding_left = 1, 0  # 60, 10

            source_background[padding_top:padding_top + height, padding_left:padding_left + width] = scores_matrix_img

            text1 = "Source image"
            text2 = "Canvas"
            text_color_list = np.array([255, 0, 0])
            text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
            # img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            #                   text_color)
            # img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            #                   text_color)

            # shape[1] is the width, it seems it needs to go first when resizing.
            final = cv2.resize(source_background, (source_background.shape[1] * 30, source_background.shape[0] * 30),
                               interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite("./debug.png", final)
            cv2.imshow("pr", final)  # this prevents code from running with wandb after first sweep
            cv2.waitKey(300)

        # TODO: INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
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

        source_img = Image.fromarray(source)
        canvas_img = Image.fromarray(canvas)
        # img = img.resize((16,16))
        source_img = np.uint8(source_img)
        canvas_img = np.uint8(canvas_img)

        if show_q_values:
            scores_matrix_img = Image.fromarray(scores_matrix)
            scores_matrix_img = np.uint8(scores_matrix_img)

            height, width, ch = scores_matrix_img.shape
            new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

            # Crate a new canvas with new width and height.
            scores_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125
        else:
            height, width, ch = source_img.shape
            new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

        # Crate a new canvas with new width and height.
        source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125
        canvas_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

        # New replace the center of canvas with original image
        padding_top, padding_left = 1, 0  # 60, 10

        source_background[padding_top:padding_top + height, padding_left:padding_left + width] = source_img
        canvas_background[padding_top:padding_top + height, padding_left:padding_left + width] = canvas_img
        if show_q_values:
            scores_background[padding_top:padding_top + height, padding_left:padding_left + width] = scores_matrix_img



        text1 = "Source image"
        text2 = "Canvas"
        text_color_list = np.array([255, 0, 0])
        text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
        # img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                   text_color)
        # img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                   text_color)

        img1 = source_background
        img2 = canvas_background
        final = cv2.hconcat((img1, img2))
        if show_q_values:
            img3 = scores_background
            final = cv2.hconcat((final, img3))
        # shape[1] is the width, it seems it needs to go first when resizing.
        final = cv2.resize(final, (final.shape[1] * 30, final.shape[0] * 30), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("./debug.png", final)
        cv2.imshow("pr", final)  # this prevents code from running with wandb after first sweep, didn't find a fix yet
        cv2.waitKey(300)

    '''SHAPES CREATION'''

    def __delete_pixel(self):
        found = False
        while not found:
            random_missing_idx = random.randint(0, self.length ** 2 - 1)
            # ''' GENERALIZATION TEST '''
            # random_missing_idx = 2
            # if random_missing_idx == 2:  # ONLY FOR 5x5 matrices!
            #    continue
            row = random_missing_idx // self.length
            col = random_missing_idx % self.length
            if self.source_matrix[row][col] == 1:
                self.source_matrix[row][col] = 0
                found = True

    def __create_diamond(self):
        mid = self.length // 2
        for x in range(mid):
            self.source_matrix[x][mid + x] = 1
            self.source_matrix[x][mid - x] = 1
            self.source_matrix[self.length - 1 - x][mid + x] = 1
            self.source_matrix[self.length - 1 - x][mid - x] = 1
        if self.length % 2 != 0:
            self.source_matrix[mid][0] = 1
            self.source_matrix[mid][self.length - 1] = 1
        if self.random_missing_pixel:
            self.__delete_pixel()

    def __create_triangle(self):
        # TODO: support for multi-dimensional shapes and maybe fix a little
        # TODO: Drawing is a little bit skewed, fix it
        b = self.length
        h = self.length
        mid = round((self.length - 1) / 2)
        self.source_matrix[h - 1] = 1
        for y in range(self.length):
            if y == self.length - 1:
                continue
            val = (b - y) // 2
            val = mid if val < 0 else val
            self.source_matrix[y][h - val] = 1
            self.source_matrix[y][0 + val] = 1
        if self.random_missing_pixel:
            self.__delete_pixel()

    def __create_circle(self):
        # TODO: check that values for a,b,r and conditions to draw are always correct
        # a, b are the coordinates of the center
        r = round(self.length / 2)
        a = b = self.length // 2
        EPSILON = 2.2
        for y in range(self.length):
            for x in range(self.length):
                if abs((x - a) ** 2 + (y - b) ** 2 - r ** 2) <= round(self.length / 2):
                    self.source_matrix[x][y] = 1
        if self.random_missing_pixel:
            self.__delete_pixel()

    def __create_square(self):
        # TODO: support for different shapes and shape size
        self.source_matrix[:, 0] = 1
        self.source_matrix[:, self.length - 1] = 1
        self.source_matrix[0, :] = 1
        self.source_matrix[self.length - 1, :] = 1
        # delete a random pixel
        if self.random_missing_pixel:
            self.__delete_pixel()
        # if self.random_missing_pixel:
        #     random_row = random.randint(0, self.length - 1)
        #     if random_row == 0 or random_row == self.length - 1:
        #         random_col = random.randint(0, self.length - 1)
        #     else:
        #         random_col = random.choice((0, self.length-1))
        #     self.source_matrix[random_row, random_col] = 0


class StoppableSimpleSequentialGeometricNonEpisodicShapeEnv(SimpleGeometricShapesEnv):
    def __init__(self, side_length: int, max_steps, random_starting_pos=False, random_missing_pixel=False, subtract_canvas=False):  # , start_on_line=False):
        """

        :param side_length:
        :param max_steps:
        :param random_starting_pos:
        :param random_missing_pixel:
        :param subtract_canvas: whether the source input states must be provided complete or only as a
        difference with the already drawn canvas
        """
        super().__init__(side_length, max_steps, random_starting_pos, random_missing_pixel)
        self.subtract_canvas = subtract_canvas
        self.complete_source_matrix = self.source_matrix.copy()

    def reset(self):
        self.canvas = np.zeros((self.length, self.length))
        self.current_state = 0
        self.row = 0
        self.column = 0

        self.source_matrix = np.zeros((self.length, self.length))
        # random_shape_n = random.randint(0, len(self.shapes_list) - 1)
        self.shapes_list[self.shape_n]()  # call the function to create a random shape
        self.complete_source_matrix = self.source_matrix.copy()
        self.row = self.current_state // self.length
        self.column = self.current_state % self.length

        self.step_count = 0
        self.done = False
        self.color_action = False
        self.starting_pos = self.current_state

        self.shape_n += 1
        self.shape_n = self.shape_n % len(self.shapes_list)
        return self.shape_n, self.source_matrix, self.canvas, (self.row, self.column)  # self.current_state

    def next_drawing(self):
        self.canvas = np.zeros((self.length, self.length))
        self.current_state = 0
        self.row = 0
        self.column = 0

        self.source_matrix = np.zeros((self.length, self.length))
        # random_shape_n = random.randint(0, len(self.shapes_list) - 1)
        self.shapes_list[self.shape_n]()  # call the function to create a random shape
        self.complete_source_matrix = self.source_matrix.copy()
        self.row = self.current_state // self.length
        self.column = self.current_state % self.length

        self.step_count = 0
        self.color_action = False
        self.starting_pos = self.current_state

        self.shape_n += 1
        self.shape_n = self.shape_n % len(self.shapes_list)

    # new_reward
    def step(self, action):
        is_win = False
        chosen_action_str = ''
        if self.show_debug_info:
            print('source matrix:')
            pprint(self.complete_source_matrix)
            print('canvas:')
            pprint(self.canvas)
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

        # simple reward - working
        '''reward is -1 per step, unless the agent is in a cell that must be colored. Moreover,
        if we colored the correct cell, get +1 reward'''
        # reward = -1  # -1 per step
        reward = 0
        if self.complete_source_matrix[self.row][self.column] == 1:
            reward = 0  # unless the agent is in a cell that must be colored

        if action == 4:  # if we drew, we have to check whether the drawn cell is the right one
            if self.canvas[self.row][self.column] == 0 and self.complete_source_matrix[self.row][self.column] == 1:
                reward = 1  # if we colored the correct cell, get +1 reward
            self.canvas[self.row][self.column] = 1
            chosen_action_str = 'color cell'
            self.color_action = True

        if self.show_debug_info:
            print('chosen action:', chosen_action_str)
            print('-----------')
            self.show_debug_info = False

        # if all the correct cells are colored, the episode can end
        if np.array_equal(self.complete_source_matrix, self.canvas):
            if self.shape_n == 0:
                self.reset()
                self.done = True
                # self.num_completed = 0
                is_win = True
            # reward = 100
            else:
                self.reset()
                # self.num_completed += 1

            # if self.num_completed == 50:
            #     self.done = True
            #     self.num_completed = 0
            # self.done = True
        elif self.step_count == self.max_steps:
            self.done = True
            self.shape_n = 0
            # if np.sum(self.canvas[1] == 1) != self.length:
            #    reward = -100
        self.step_count += 1
        if self.done:
            cv2.destroyAllWindows()
        if self.subtract_canvas:
            self.source_matrix = self.complete_source_matrix - self.canvas
            self.source_matrix[self.source_matrix == -1] = 0
        return (self.shape_n, self.source_matrix, self.canvas, (self.row, self.column)), reward, self.done, is_win
        # return (self.shape_n, self.source_matrix, self.canvas, self.current_state), reward, self.done

    def step_simultaneous(self, action, pen_state):
        is_win = False

        # simple reward - working
        '''reward is -1 per step, unless the agent is in a cell that must be colored. Moreover,
        if we colored the correct cell, get +1 reward'''
        # reward = -1  # -1 per step
        reward = 0
        # if self.complete_source_matrix[self.row][self.column] == 1:
        #     reward = 0  # unless the agent is in a cell that must be colored
        if action == 4:  # with action 4 we go to next env
            if self.shape_n == 0:
                self.done = True
                if np.array_equal(self.complete_source_matrix, self.canvas):
                    # TODO: should we give a bonus reward for correctly completing the drawing?
                    is_win = True
            else:
                self.next_drawing()
            return (self.shape_n, self.source_matrix, self.canvas, (self.row, self.column)), reward, self.done, is_win

        if pen_state == 1:  # if we drew, we have to check whether the drawn cell is the right one
            if self.canvas[self.row][self.column] == 0 and self.complete_source_matrix[self.row][self.column] == 1:
                reward = 1  # if we colored the correct cell, get +1 reward
            # else:
            #     reward = -0.1  # -0.1
                # self.done = True
                # self.shape_n = 0
                # return (self.shape_n, self.source_matrix, self.canvas,
                #         (self.row, self.column)), reward, self.done, False
            self.canvas[self.row][self.column] = 1
            chosen_action_str = 'color cell'
            self.color_action = True
        if pen_state == 0:
            self.color_action = False

        chosen_action_str = ''
        if self.show_debug_info:
            print('source matrix:')
            pprint(self.complete_source_matrix)
            print('canvas:')
            pprint(self.canvas)
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
            # self.color_action = False

        if self.show_debug_info:
            print('chosen action:', chosen_action_str)
            print('-----------')
            self.show_debug_info = False

        # if all the correct cells are colored, the episode can end

        if self.step_count == self.max_steps:
            self.done = True
            self.shape_n = 0
            # if np.sum(self.canvas[1] == 1) != self.length:
            #    reward = -100
        self.step_count += 1
        if self.done:
            cv2.destroyAllWindows()
        if self.subtract_canvas:
            self.source_matrix = self.complete_source_matrix - self.canvas
            self.source_matrix[self.source_matrix == -1] = 0
        return (self.shape_n, self.source_matrix, self.canvas, (self.row, self.column)), reward, self.done, is_win
        # return (self.shape_n, self.source_matrix, self.canvas, self.current_state), reward, self.done


class SimpleRandomGeometricNonEpisodicShapeEnv(SimpleGeometricShapesEnv):
    def __init__(self, side_length: int, max_steps, random_starting_pos=False):  # , start_on_line=False):
        super().__init__(side_length, max_steps, random_starting_pos)

    def reset(self):
        self.canvas = np.zeros((self.length, self.length))
        self.current_state = 0
        self.row = 0
        self.column = 0

        self.source_matrix = np.zeros((self.length, self.length))
        random_shape_n = random.randint(0, len(self.shapes_list) - 1)
        # self.shapes_list.pop(random_shape_n)()  # pop and call function
        self.shapes_list[random_shape_n]()  # call the function to create a random shape
        self.row = self.current_state // self.length
        self.column = self.current_state % self.length

        self.step_count = 0
        self.done = False
        self.color_action = False
        self.starting_pos = self.current_state

        self.shape_n += 1
        self.shape_n = self.shape_n % len(self.shapes_list)
        self.shapes_list.pop(random_shape_n)
        return self.shape_n, self.source_matrix, self.canvas, (self.row, self.column)  # self.current_state

    # new_reward
    def step(self, action):
        is_win = False
        chosen_action_str = ''
        if self.show_debug_info:
            print('source matrix:')
            pprint(self.source_matrix)
            print('canvas:')
            pprint(self.canvas)
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

        # simple reward - working
        '''reward is -1 per step, unless the agent is in a cell that must be colored. Moreover,
        if we colored the correct cell, get +1 reward'''
        # reward = -1  # -1 per step
        reward = 0
        if self.source_matrix[self.row][self.column] == 1:
            reward = 0  # unless the agent is in a cell that must be colored

        if action == 4:  # if we drew, we have to check whether the drawn cell is the right one
            if self.canvas[self.row][self.column] == 0 and self.source_matrix[self.row][self.column] == 1:
                reward = 1  # if we colored the correct cell, get +1 reward
            self.canvas[self.row][self.column] = 1
            chosen_action_str = 'color cell'
            self.color_action = True

        if self.show_debug_info:
            print('chosen action:', chosen_action_str)
            print('-----------')
            self.show_debug_info = False

        # if all the correct cells are colored, the episode can end
        if np.array_equal(self.source_matrix, self.canvas):
            if not self.shapes_list:  # if list is empty
                # self.reset()
                self.shapes_list = self.shapes_list_backup
                self.done = True
                self.num_completed = 0
                is_win = True
            # reward = 100
            else:
                self.reset()
                self.num_completed += 1

            # if self.num_completed == 50:
            #     self.done = True
            #     self.num_completed = 0
            # self.done = True
        elif self.step_count == self.max_steps:
            self.done = True
            self.shape_n = 0
            # if np.sum(self.canvas[1] == 1) != self.length:
            #    reward = -100
        self.step_count += 1
        if self.done:
            cv2.destroyAllWindows()
        return (self.shape_n, self.source_matrix, self.canvas, (self.row, self.column)), reward, self.done, is_win
        # return (self.shape_n, self.source_matrix, self.canvas, self.current_state), reward, self.done
