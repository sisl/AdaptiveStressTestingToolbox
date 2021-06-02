import dill
import numpy as np
from matplotlib import pyplot as plt
import pdb
import time

def debug_print(var_list, global_scope, local_scope):
    print_str = ''
    for var in var_list:
        print_str += (var + ': ' + repr(eval(var, global_scope, local_scope)) + ', ')

    # print(print_str[:-2])

class TrackList(object):
    # Linked list of track or intersection cells

    def __init__(self, has_a_clear_path=False):
        self._first = None
        self._last = None
        self.intersections = []
        self.has_a_clear_path = has_a_clear_path
        self.probability = 1.0

    @property
    def first(self):
        return self._first

    @first.setter
    def first(self, value):
        self._first = value
        if self._last is None:
            self._last = value

    @property
    def last(self):
        return self._last

    @last.setter
    def last(self, value):
        # Append to linked list, shift end of list pointer
        self._last.next = value
        while self._last.next is not None:
            self._last = self._last.next

    @property
    def num_intersections(self):
        return len(self.intersections)

class TrackCell(object):

    def __init__(self, x, y, p_obstruction = 0.0, p_reward=0.0):
        self._x = x
        self._y = y
        self.obstruction = (np.random.rand() <= p_obstruction)
        self.reward = (not self.obstruction and (np.random.rand() <= p_reward))
        if self.obstruction:
            self._probability = p_obstruction
        elif self.reward:
            self._probability = p_reward
        else:
            self._probability = 1.0 - p_obstruction - p_reward
        self._next = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        self._probability = value

class IntersectionCell(TrackCell):

    def __init__(self, x, y, next_track_num=0, has_a_clear_path=False):
        self._next_track_num = next_track_num
        self._next_track_starts = []
        self.has_a_clear_path = has_a_clear_path
        super().__init__(x,y, p_obstruction=0.0, p_reward=0.0)

    @property
    def next_track_num(self):
        return self._next_track_num

    @next_track_num.setter
    def next_track_num(self, value):
        self._next_track_num = value

    @property
    def next(self):
        return self._next[self._next_track_num]

    @next.setter
    def next(self, value):
        self._next = value

    @property
    def num_tracks(self):
        return len(self._next)

    @property
    def next_track_start(self):
        return self._next_track_starts[self._next_track_num]

    @property
    def next_track_list(self):
        return self._next

    @property
    def next_track_starts(self):
        return self._next_track_starts

def print_frame(frame):
    # Helper function to print a frame in one line, with standardized settings
    plt.imshow(frame)
    plt.show()

class CrazyTrolleyHeadlessGame:
    def __init__(self, height=16, width=32, hardest_level=50, init_p_split=0.3, debug=False):

        self.level_bonus = 100
        self.gem_bonus = 50
        self.initial_lives = 1

        # self.background = None
        # reset game tracking variables
        self._update_counter = 0
        self._level = 0
        self._cell = None
        self._lives = self.initial_lives
        self._score = 0
        self.intersection_num = 0
        self.intersection_setting = 0
        self._random_seed = None
        self.debug = debug

        self._height = height
        self._width = width

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # Return 0 if on_clear_path
        self.p_obstruction = lambda level, on_clear_path=False: 0 if on_clear_path else sigmoid(-5 + level / 2) / 2
        self.p_reward = lambda level: sigmoid(-5 + level / 2) / 2
        self.trolley_update_period = lambda level: max([50 - level, 10])
        # probability parameter for geometric distribution
        self.p_split = lambda level: 0.999 - (0.999 - init_p_split)*(max([hardest_level - level, 0])/hardest_level)
        # Scaling parameter for geometric distribution, bigger makes distribution sharper curve (lower values more likely)
        self.scale = 2

        self.num_tracks = lambda max_tracks: 1 + int(np.floor(np.log(1 - np.random.rand() *
                                                                     (1 - (self.p_split(self._level)) **
                                                                      (self.scale * max_tracks)))
                                                              / (self.scale * np.log(self.p_split(self._level)))))
        # Calculate probability of num_tracks
        lower = lambda num_tracks, max_tracks: -(
                np.exp((num_tracks - 1) * self.scale * np.log(self.p_split(self._level))) - 1) / (
                                                       1 - (self.p_split(self._level) ** (self.scale * max_tracks)))
        upper = lambda num_tracks, max_tracks: -(
                np.exp(num_tracks * self.scale * np.log(self.p_split(self._level))) - 1) / (
                                                       1 - (self.p_split(self._level) ** (self.scale * max_tracks)))
        self.prob_num_tracks = lambda num_tracks, max_tracks: upper(num_tracks, max_tracks) - lower(num_tracks,
                                                                                                    max_tracks)
        # pdb.set_trace()

        # self.p_obstruction = lambda x: 0.5 - (0.45/(x ** (1/2)))
        # self.p_reward =  lambda x: 0.5 - (0.45/(x ** (1/2)))

        self.action_dict = {0: self.action_noop,
                            1: self.action_decrement_intersection_setting,
                            2: self.action_increment_intersection_num,
                            3: self.action_increment_intersection_setting,
                            4: self.action_decrement_intersection_num}

        self.action_meaning_dict = {0: "NOOP",
                                    1: "DECREMENT_INTERSECTION_SETTING",
                                    2: "INCREMENT_INTERSECTION_NUM",
                                    3: "INCREMENT_INTERSECTION_SETTING",
                                    4: "DECREMENT_INTERSECTION_NUM",}

        self._pixel_num_from_key_dict = {'background': 0,
                                         'track': 1,
                                         'trolley': 2,
                                         'player': 3,
                                         'unselected_intersection': 4,
                                         'selected_intersection': 5,
                                         'obstruction': 6,
                                         'gem': 7,
                                         }
        # Generate inverted dict, since it is one to one mapping
        self._pixel_key_from_num_dict = {v: k for k, v in self._pixel_num_from_key_dict.items()}
        # Generate a list of pixel types
        self.pixel_keys = [k for k in self._pixel_num_from_key_dict.keys()]



        self.new_game()


        #
        # self.new_frame()
        # self.tick()

    def debug_print(self, *args):
        if self.debug:
            print(*args)

    @property
    def on_clear_path(self):
        return self.check_trolley_path()

    @property
    def game_over(self):
        return self._lives <= 0

    @property
    def score(self):
        return self._score

    @property
    def lives(self):
        return self._lives

    @property
    def level(self):
        return self._level

    @property
    def frame(self):
        return self._frame.copy()

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def end_of_frame(self):
        return (self._cell is None) or (self._cell.next is None)

    @random_seed.setter
    def random_seed(self, value):
        self._random_seed = value

    @property
    def frame_probability(self):
        return self._track_list.probability

    def pixel_num_from_key(self, key):
        return self._pixel_num_from_key_dict[key]

    def pixel_key_from_num(self, num):
        return self._pixel_key_from_num_dict[num]

    def new_game(self):
        self._update_counter = 0
        self._level = 0
        self._cell = None
        self._lives = self.initial_lives
        self._score = 0

        self.new_frame()
        self.tick()

    def new_frame(self):
        # Generate a new frame
        self._base_frame, self._track_list = self.generate_frame(height=self._height, width=self._width, rnd_seed=self._random_seed)
        self.debug_print(self.frame_probability)
        self._frame = self._base_frame.copy()
        self._cell = self._track_list.first

        # Reset player to new default spot
        self.intersection_num = 0
        self.intersection_setting = 0

        # Give reward for new level
        self._score += self.level_bonus * self._level

        # Increment level counter
        self._level += 1
        if self._level > 50:
            self._lives = 0

    def action_noop(self):
        # Do nothing action
        pass

    # if event.key == 'd':
    def action_increment_intersection_num(self):
        if self._track_list.num_intersections >= 1:
            self.intersection_num = (self.intersection_num + 1) % self._track_list.num_intersections
        # print(self.intersection_num)

    # if event.key == 'a':
    def action_decrement_intersection_num(self):
        if self._track_list.num_intersections >= 1:
            self.intersection_num = (self.intersection_num - 1) % self._track_list.num_intersections

    # if event.key == 's':
    def action_increment_intersection_setting(self):
        if self._track_list.num_intersections >= 1:
            self.intersection_setting = ((self.intersection_setting + 1) %
                                         self._track_list.intersections[self.intersection_num].num_tracks)
            self._track_list.intersections[self.intersection_num].next_track_num = self.intersection_setting
        # print(self.intersection_setting)

    # if event.key == 'w':
    def action_decrement_intersection_setting(self):
        if self._track_list.num_intersections >= 1:
            self.intersection_setting = ((self.intersection_setting - 1) %
                                         self._track_list.intersections[self.intersection_num].num_tracks)
            self._track_list.intersections[self.intersection_num].next_track_num = self.intersection_setting
        # print(self.intersection_setting)

    def player_action(self, action_num):
        # Handle player action
        self.action_dict[action_num]()

    def tick(self):
        self._frame = self._base_frame.copy()


        # Update intersections
        for intersection in self._track_list.intersections:
            for track_start in intersection.next_track_starts:
                # First set all tracks to 'off'
                self._frame[track_start.y, track_start.x] = self._pixel_num_from_key_dict['unselected_intersection']
            # Set the selected track to 'on'
            selected_track = intersection.next_track_start
            self._frame[selected_track.y, selected_track.x] = self._pixel_num_from_key_dict['selected_intersection']

        # Update player location
        if self._track_list.num_intersections >= 1:
            selected_intersection = self._track_list.intersections[self.intersection_num]
            self._frame[selected_intersection.y + 1, selected_intersection.x - 1] = self._pixel_num_from_key_dict['player']

        # Update trolley
        self._update_counter += 1
        if self._update_counter >= self.trolley_update_period(self._level):
            self._cell = self._cell.next
            self._update_counter = 0

            if self._cell is None:
                # End of frame, tell game to generate next frame
                self.new_frame()

            if self._cell.obstruction:
                self._lives -= 1
            if self._cell.reward:
                self._score += self.gem_bonus

            # self.debug_print(self.check_trolley_path())

        self._frame[self._cell.y, self._cell.x] = self._pixel_num_from_key_dict['trolley']

    def check_trolley_path(self):
        cell_trace = self._cell
        # Scan to end of frame
        while (cell_trace is not None):
            # Scan to next intersection
            while (type(cell_trace) is not IntersectionCell):
                cell_trace = cell_trace.next

                if cell_trace is None:
                    return True
                if cell_trace.obstruction:
                    return False

            self.debug_print(cell_trace._next_track_num)
            self.debug_print(cell_trace.has_a_clear_path)
            if not cell_trace.has_a_clear_path:
                return False
            # Move onto next track
            cell_trace=cell_trace.next

        # Got to end of frame on clear path
        return True

    def lay_track(self, frame, track_list, frame_start_x, frame_end_x, frame_y, track_start_x, track_y, on_clear_path=False):
        path_is_clear = True
        track_probability = 1.0
        for track_x_idx, frame_x in enumerate(range(frame_start_x, frame_end_x)):
            track_cell = TrackCell(x=track_start_x+track_x_idx, y=track_y, p_obstruction=self.p_obstruction(self._level, on_clear_path), p_reward=self.p_reward(self._level))
            track_probability *= track_cell.probability
            if track_list.first is None:
                track_list.first = track_cell
            else:
                track_list.last = track_cell

            if track_cell.obstruction:
                frame[frame_y, frame_x] = self._pixel_num_from_key_dict['obstruction']
                path_is_clear = False
            elif track_cell.reward:
                frame[frame_y, frame_x] = self._pixel_num_from_key_dict['gem']
            else:
                frame[frame_y, frame_x] = self._pixel_num_from_key_dict['track']

        track_list.has_a_clear_path = path_is_clear
        track_list.probability = track_probability

        return frame, track_list

    def connect_intersection(self, frame, sub_track_lists, frame_x, frame_y, intersection_x, intersection_y):
        # Fill in intersection
        frame[frame_y, frame_x] = self._pixel_num_from_key_dict['track']
        # Build vertical tracks from intersection point to each of the next tracks
        connected_sub_track_list = []
        for sub_track_list in sub_track_lists:
            end_y = sub_track_list.first.y
            delta_y = intersection_y - end_y
            if np.abs(delta_y) <= 1:
                # No vertical track, connect directly to start of sub track
                connected_sub_track_list.append(sub_track_list.first)
            else:
                connected_sub_track = TrackList()
                # Build track list and fill frame vertically up or down to horizontal start of sub_track
                for y_idx in range(np.sign(delta_y), delta_y, np.sign(delta_y)):
                    track_cell = TrackCell(x=intersection_x, y=intersection_y - y_idx, p_obstruction=0.0, p_reward=0.0)
                    if connected_sub_track.first is None:
                        connected_sub_track.first = track_cell
                    else:
                        connected_sub_track.last = track_cell
                    frame[frame_y - y_idx, frame_x] = self._pixel_num_from_key_dict['track']
                # Point end of vertical track to start of horizontal track
                if connected_sub_track.last is None:
                    print('Here!')
                connected_sub_track.last = sub_track_list.first
                # Connect full subtrack to intersection
                connected_sub_track_list.append(connected_sub_track.first)

        # Create and connect intersection
        intersection_cell = IntersectionCell(x=intersection_x, y=intersection_y)
        intersection_cell.next = connected_sub_track_list
        has_a_clear_path = False
        for sub_track_list in sub_track_lists:
            if sub_track_list.has_a_clear_path:
                has_a_clear_path = True
            next_track_start = sub_track_list.first
            # Due to rendering issues, first track cell of each intersection path can't be special
            next_track_start.obstruction = False
            next_track_start.reward = False
            next_track_start.probability = 1.0
            intersection_cell._next_track_starts.append(next_track_start)

        intersection_cell.has_a_clear_path = has_a_clear_path

        return frame, intersection_cell

    def generate_frame(self, height, width, rnd_seed=None):
        if rnd_seed is not None:
            np.random.seed(rnd_seed)
        frame = np.zeros((height, width))
        track_start = (height - 1) // 2
        track_list = TrackList()
        # frame, track_list = self.lay_track(frame, track_list, height=track_start, start_width=0, end_width=5)
        frame, track_list = self.lay_track(frame, track_list, frame_start_x=0, frame_end_x=5, frame_y=track_start, track_start_x=0, track_y=track_start, on_clear_path=True)
        try:
            sub_frame, sub_track_list = self.generate_intersection(height, width-5, start_x=5, start_y=0, on_clear_path=True)
            frame[:, 5:] = sub_frame
            track_list.last = sub_track_list.first
            track_list.intersections = sub_track_list.intersections
            track_list.probability *= sub_track_list.probability
        except Exception as e:
            print(e)
            print_frame(frame)
            pdb.set_trace()
            print()

        return frame, track_list

    def generate_intersection(self, height, width, start_x, start_y, on_clear_path=True):
        # Create an array to hold intersection
        frame = np.zeros((height, width))
        # Track starts at halfway point, rounding down, and keeping top and bottom rows free
        track_start = (height - 1) // 2
        track_list = TrackList()
        # pdb.set_trace()
        # Not enough room for an intersection, return a straight track
        if width <= 10:
            # sub_frame[track_start, :] = 1
            # sub_frame, track_list = self.lay_track(sub_frame, track_list, height=track_start, start_width=0, end_width=width)
            frame_y_ = track_start
            track_y_ = start_y + track_start
            debug_print(
                ['frame_y_',
                 'track_y_', 'start_y'], global_scope=globals(), local_scope=locals())

            frame, track_list = self.lay_track(frame, track_list, frame_start_x=0, frame_end_x=width, frame_y=track_start,
                                          track_start_x=start_x, track_y=start_y + track_start, on_clear_path=on_clear_path)
            # pdb.set_trace()
            return frame, track_list

        # Straight track to the intersection
        # sub_frame[track_start, 0:5] = 1
        # sub_frame, sub_track_list = self.lay_track(sub_frame, track_list, height=track_start, start_width=0, end_width=5)
        frame, track_list = self.lay_track(frame, track_list, frame_start_x=0, frame_end_x=5, frame_y=track_start,
                                          track_start_x=start_x, track_y=start_y + track_start, on_clear_path=on_clear_path)
        # Max number of tracks to split to, keeping free space between rows
        free_height = height - 2
        max_tracks = max([(free_height - 1) // 2, 1])
        num_tracks = self.num_tracks(max_tracks)
        num_tracks_probability = self.prob_num_tracks(num_tracks, max_tracks)
        track_list.probability *= num_tracks_probability
        # if max_tracks > 1:
        #     # Generate random number of tracks
        #     num_tracks = np.random.randint(low=1, high=max_tracks+1)
        # else:
        #     num_tracks = 1

        sub_track_lists = []

        if num_tracks == 1:
            # pdb.set_trace()
            try:
                # Only one track, lay down 5 squares of track and get the next intersextion
                # sub_frame[track_start, 0:5] = 1
                frame_y_ = track_start
                track_y_ = start_y + track_start
                debug_print(['num_tracks',
                             'frame_y_', 'track_y_', 'start_y'], global_scope=globals(), local_scope=locals())
                # pdb.set_trace()
                sub_frame, sub_track_list = self.generate_intersection(height, width-5, start_x=start_x+5, start_y=start_y, on_clear_path=on_clear_path)
                frame[:, 5:] = sub_frame
                track_list.last = sub_track_list.first
                sub_track_lists.append(sub_track_list)
            except Exception as e:
                print(e)
                print_frame(frame)
                pdb.set_trace()
                print()
        else:
            try:
                # Get height for each track, keeping free rows at top, bottom, and at least one between
                free_height = height - 2
                height_per_track = free_height // num_tracks


                sub_track_starts = []
                on_clear_path_list = [False] * num_tracks
                clear_path_index = np.random.randint(low=0, high=num_tracks)
                on_clear_path_list[clear_path_index] = on_clear_path

                for track_num in range(num_tracks):
                    try:
                        # pdb.set_trace()
                        # Get the dimensions and coordinates of the subtrack
                        min_height = 1 + (track_num * height_per_track)
                        max_height = 1 + min((track_num + 1) * height_per_track - 1, height - 1)
                        sub_track_height = (max_height - min_height) + 1
                        sub_track_width = width - 10

                        # Get subtrack location at halfway point of subtrack space, rounding down, and keeping a row free
                        # at the top and bottom
                        sub_track_start = min_height + (max_height - min_height) // 2
                        sub_track_starts.append(sub_track_start)

                        # Lay down track to next intersection and get that intersection
                        # sub_frame[sub_track_start, 5:10] = 1
                        sub_track_list = TrackList()
                        # if min_height <= 1:
                        #     pdb.set_trace()
                        frame_y_ = sub_track_start
                        track_y_ = start_y + sub_track_start
                        start_y_ = min_height
                        debug_print(['num_tracks', 'track_num', 'min_height', 'max_height', 'sub_track_height', 'sub_track_width', 'frame_y_', 'track_y_', 'start_y_'], global_scope = globals(), local_scope=locals())
                        frame, sub_track_list = self.lay_track(frame, sub_track_list, frame_start_x=5, frame_end_x=10,
                                                              frame_y=sub_track_start,
                                                              track_start_x=start_x+5, track_y=start_y + sub_track_start, on_clear_path=on_clear_path_list[track_num])
                        # pdb.set_trace()
                        sub_frame, sub_sub_track_list = self.generate_intersection(sub_track_height, sub_track_width, start_x=start_x+10, start_y=start_y + min_height, on_clear_path=on_clear_path_list[track_num])
                        frame[min_height:max_height + 1, 10:] = sub_frame
                        sub_track_list.last = sub_sub_track_list.first
                        sub_track_list.intersections = sub_sub_track_list.intersections
                        sub_track_lists.append(sub_track_list)
                        # pdb.set_trace()
                    except Exception as e:
                        print(e)
                        print_frame(frame)
                        pdb.set_trace()
                        print()

                # Lay vertical track to splits
                #  = 1
                frame, intersection_cell = self.connect_intersection(frame, sub_track_lists, frame_x=5, frame_y=track_start, intersection_x=start_x+5, intersection_y=track_start+start_y)
                track_list.last = intersection_cell
                track_list.intersections.append(intersection_cell)



            except Exception as e:
                print(e)
                print_frame(frame)
                pdb.set_trace()
                print()

        has_a_clear_path = False
        for sub_track_list in sub_track_lists:
            track_list.probability *= sub_track_list.probability
            if sub_track_list.has_a_clear_path:
                has_a_clear_path = True
            # print(len(sub_track_list.intersections))
            if len(sub_track_list.intersections) > 0:
                track_list.intersections += sub_track_list.intersections

        # pdb.set_trace()
        track_list.has_a_clear_path = has_a_clear_path
        return frame, track_list

    # def __getstate__(self):
    #     return dill.dumps(self.__dict__)
    #
    # def __setstate__(self, state):
    #     self.__dict__ = dill.loads(state)


class CrazyTrolleyRenderedGame:
    def __init__(self, ax, height=16, width=32, rgb=True, debug=False):
        self.game = CrazyTrolleyHeadlessGame(height=height, width=width, debug=debug)

        # Color settings
        # self.colors = {'background': np.array([255, 255, 255]),
        #                'track': np.array([175, 175, 175]),
        #                'trolley': np.array([0, 0, 0]),
        #                'unselected_intersection': np.array([215, 0, 0]),
        #                'selected_intersection': np.array([50, 176, 0]),
        #                'gem': np.array([123, 0, 176]),
        #                'obstruction': np.array([255, 132, 0]),
        #                'player': np.array([25, 0, 255]),}

        self.colors = {'trolley': np.array([0, 0, 0]),  # 0
                       'player': np.array([8, 8, 255]),  # 36.2
                       'unselected_intersection': np.array([244, 0, 0]),  # 72.9
                       'gem': np.array([255, 7, 255]),  # 109.4
                       'selected_intersection': np.array([0, 248, 0]),  # 145.6
                       'obstruction': np.array([255, 180, 2]),  # 182.1
                       'track': np.array([218, 219, 218]),  # 218.6
                        'background': np.array([255, 255, 255]),  # 255
                       }

        self.on = False
        self.inst = True    # show instructions from the beginning
        self.background = None
        self.rgb = rgb

        # Create the axis
        self.ax = ax
        if self.ax is not None:
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            for (_, spine) in self.ax.spines.items():
                # spine.set_color('0.0')
                # spine.set_linewidth(1.0)
                spine.set_color(None)

            self.canvas = self.ax.figure.canvas

            if self.rgb:
                self.disp_frame = self.ax.imshow(np.ones((height+int(0.25*height), width, 3))*255, interpolation='none')
            else:
                self.disp_frame = self.ax.imshow(np.ones((height+int(0.25*height), width))*255, interpolation='none')

            self.title = self.ax.text(0.5, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                                      transform=ax.transAxes, ha="center")

            instructions = """
                            Player:
                            press 'd' -- Move player to next intersection
                            press 'a' -- Move player to previous intersection
                            press 'w' -- Change intersection signal up
                            press 's' -- Change intersection signal down

                            press 't' -- Close these instructions
                                        (animation will be much faster)
                            press 'g' -- Toggle the game on/off
                            press 'n' -- New frame
                            press 'q' -- Quit

                            """

            self.i = self.ax.annotate(instructions, (.5, 0.5),
                                      name='monospace',
                                      verticalalignment='center',
                                      horizontalalignment='center',
                                      multialignment='left',
                                      textcoords='axes fraction',
                                      animated=False)
            self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.display_frame = self.game.frame

        self.game.new_game()




    def rgb_frame(self, frame):
        # frame types
        # 'background': 0,
        # 'track': 1,
        # 'trolley': 2,
        # 'player': 3,
        # 'unselected_intersection': 4,
        # 'selected_intersection': 5,
        # 'obstruction': 6,
        # 'gem': 7,
        rgb_frame = np.ones((frame.shape[0], frame.shape[1], 3), dtype=int) * self.colors['background']
        for pixel_key in self.game.pixel_keys:
            rgb_frame[frame == self.game.pixel_num_from_key(pixel_key)] = self.colors[pixel_key]

        # rgb_frame[frame == self.game.pixel_num_from_key('track')] = self.colors['track']
        # rgb_frame[frame == 2] = self.colors['obstruction']
        # rgb_frame[frame == 3] = self.colors['reward']
        # rgb_frame[frame == 4] = self.colors['unselected_intersection']
        # rgb_frame[frame == 5] = self.colors['selected_intersection']
        # rgb_frame[frame == 6] = self.colors['trolley']
        # rgb_frame[frame == 7] = self.colors['player']

        return rgb_frame

    def new_game(self):
        self.game.new_game()
        self.update_frame()

    def get_display_frame(self, frame):
        display_frame = frame.copy
        if self.rgb:
            display_frame = self.rgb_frame(display_frame)

        return display_frame


    def update_frame(self):
        self.display_frame = self.game.frame
        if self.rgb:
            self.display_frame = self.rgb_frame(self.display_frame)

        if self.game.game_over:
            self.on = False
            if self.rgb:
                # Invert frame colors
                self.display_frame = (np.ones_like(self.display_frame) * 255) - self.display_frame

        self.display_frame_with_header = np.vstack([(np.ones_like(self.display_frame) * 255)[0:int(0.25*self.display_frame.shape[0]),:], self.display_frame])


    def draw(self, event, animation_only=False):

        if self.background is None:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # restore the clean slate background
        # self.canvas.restore_region(self.background)

        if self.on:
            self.canvas.restore_region(self.background)

            if not animation_only:
                # This allows the game state update and the display update to be handled seperately, if needed
                self.game.tick()

            self.update_frame()

            if self.ax is not None:

                self.disp_frame.set_data(self.display_frame_with_header)
                self.ax.draw_artist(self.disp_frame)

                self.title.set_text('Level: {level:03d}          Score: {score:07d}          Lives: {lives}'.format(
                    level=self.game.level,
                    score=self.game.score,
                    lives=self.game.lives))
                self.ax.draw_artist(self.title)

        # just redraw the axes rectangle
        if self.ax is not None:
            self.canvas.blit(self.ax.bbox)
            self.canvas.flush_events()
        # if self.cnt == 50000:
        #     # just so we don't get carried away
        #     print("...and you've been playing for too long!!!")
        #     plt.close()
        #
        # self.cnt += 1
        return True

    def on_key_press(self, event):

        if event.key == 'w':
            self.game.player_action(1)

        if event.key == 'd':
            self.game.player_action(2)

        if event.key == 's':
            self.game.player_action(3)

        if event.key == 'a':
            self.game.player_action(4)


        if event.key == 'n':
            self.game.new_frame()
            self.update_frame()

        if event.key == 'g':
            self.on = not self.on
            if self.game.game_over:
                self.new_game()

        if event.key == 't':
            self.inst = not self.inst
            self.i.set_visible(not self.i.get_visible())
            self.background = None
            self.canvas.draw_idle()

        if event.key == 'q':
            plt.close()

    # def __getstate__(self):
    #     return dill.dumps(self.__dict__)
    #
    # def __setstate__(self, state):
    #     self.__dict__ = dill.loads(state)

def play_game(height=16, width=32, rgb=True, debug=False):
    fig, ax = plt.subplots()
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    canvas = ax.figure.canvas
    animation = CrazyTrolleyRenderedGame(ax, height=height, width=width, rgb=rgb, debug=debug)

    # disable the default key bindings
    if fig.canvas.manager.key_press_handler_id is not None:
        canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    # reset the blitting background on redraw
    def on_redraw(event):
        animation.background = None

    # bootstrap after the first draw
    def start_anim(event):
        canvas.mpl_disconnect(start_anim.cid)

        def local_draw():
            if animation.ax.get_renderer_cache():
                animation.draw(None)

        start_anim.timer.add_callback(local_draw)
        start_anim.timer.start()
        canvas.mpl_connect('draw_event', on_redraw)

    start_anim.cid = canvas.mpl_connect('draw_event', start_anim)
    start_anim.timer = animation.canvas.new_timer(interval=1)

    tstart = time.time()

    plt.show()
    # print('FPS: %f' % (animation.cnt / (time.time() - tstart)))

if __name__ == '__main__':
    # for i in range(18,100):
    #     print(i)
    #     frame = generate_frame(height=16, width=32, rnd_seed=i)
    #     print_frame(frame)
        # pdb.set_trace()

    play_game(height=64, width=64, rgb=True, debug=True)

    # frame, track_list = generate_frame(height=16, width=16, rnd_seed=0)
    # track_cell = track_list.first
    # coords = []
    # intersection_direction_list = [4]
    # intersection_direction_index = 0
    # while track_cell is not None:
    #     coords.append((track_cell.x, track_cell.y))
    #     if type(track_cell) is IntersectionCell:
    #         # pdb.set_trace()
    #         track_cell.next_track_num = intersection_direction_list[intersection_direction_index]
    #         intersection_direction_index += 1
    #     track_cell = track_cell.next
    # print(coords)
    #
    # for intersection_cell in track_list.intersections:
    #     print(intersection_cell.x, intersection_cell.y)
    #
    # print_frame(frame)
    # pdb.set_trace()
