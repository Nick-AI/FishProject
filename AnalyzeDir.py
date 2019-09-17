import os
import pdb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


class BatchAnalyzer:
    def __init__(self, data_dir, radius_list, frame_rate, save_dir, frame_file):
        self.SOURCE_DIR = data_dir
        self.RADIUS_LIST = radius_list
        self._radius = None
        self.FRAME_RATE = frame_rate
        self.TARGET_DIR = save_dir
        # Image petri diameter = 181px
        # Real petri diameter = 94mm
        self.SIZE_RATIO = 1.93
        self.CONF_THRESHOLD = 0.92
        self.FRAME_FILE = frame_file
        if self.FRAME_FILE is not None:
            self.FRAME_SNIPS = {}
            self._parse_frames(self.FRAME_FILE)

    @property
    def approach_radius(self):
        return self._radius

    @approach_radius.setter
    def approach_radius(self, value):
        self._radius = value

    def _parse_frames(self, source):
        with open(source, 'r') as f:
            for idx, line in enumerate(f):
                # row specifying video name
                if idx%3 == 0:
                    vname = line.strip()
                    self.FRAME_SNIPS[vname] = {'l_fish': None,
                                               'r_fish': None}
                # row specifying left fish frames
                elif (idx-1)%3 == 0:
                    delin = line.split(' ')
                    self.FRAME_SNIPS[vname]['l_fish'] = [int(delin[0]), int(delin[1])]

                elif (idx-2)%3 == 0:
                    delin = line.split(' ')
                    self.FRAME_SNIPS[vname]['r_fish'] = [int(delin[0]), int(delin[1])]

    def _get_distance(self, x1, y1, x2, y2):
        return np.sqrt(np.square(x1-x2) + np.square(y1-y2)) / self.SIZE_RATIO

    def _is_left_facing(self, row, side_prefix):
        # which side of fish faces rod
        left_d = self._get_distance(row[f'{side_prefix}head_l'][0], row[f'{side_prefix}head_l'][1],
                                    row[f'{side_prefix}rod'][0], row[f'{side_prefix}rod'][1])
        right_d = self._get_distance(row[f'{side_prefix}head_r'][0], row[f'{side_prefix}head_r'][1],
                                     row[f'{side_prefix}rod'][0], row[f'{side_prefix}rod'][1])

        if left_d < right_d:
            return 1
        return 0

    def _is_in_radius(self, row, side_prefix):
        # which side of fish (if any) crosses approach-radius
        left_d = self._get_distance(row[f'{side_prefix}head_l'][0], row[f'{side_prefix}head_l'][1],
                                    row[f'{side_prefix}rod'][0], row[f'{side_prefix}rod'][1])
        right_d = self._get_distance(row[f'{side_prefix}head_r'][0], row[f'{side_prefix}head_r'][1],
                                     row[f'{side_prefix}rod'][0], row[f'{side_prefix}rod'][1])

        if left_d < self.approach_radius or right_d < self.approach_radius:
            return 1
        return 0

    @staticmethod
    def _coord_fill_helper(relative_coords, frame_idx, fill_length, start_x, start_y):
        x = start_x + ((fill_length-frame_idx)/(fill_length+1) * (relative_coords[0]-start_x))
        y = start_y + ((fill_length-frame_idx)/(fill_length+1) * (relative_coords[1]-start_y))
        return [x, y]

    def _fill_missing_frames(self, row_buffer, curr_row, backtrack_counter, start_row):
        buffer_end = len(row_buffer)-1
        for row_idx in range(backtrack_counter):
            tmp = row_buffer[buffer_end-row_idx]
            for col_idx in range(-1, -6, -1):
                s_x = start_row[col_idx][0]
                s_y = start_row[col_idx][1]
                tmp[col_idx] = self._coord_fill_helper(curr_row[col_idx], row_idx, backtrack_counter, s_x, s_y)
            row_buffer[buffer_end - row_idx] = tmp
        return row_buffer

    def _replace_low_conf(self, frame_lbls):
        l_row_buffer = []
        r_row_buffer = []
        r_start_row = [[0, 0]]*5  # 3 head markers, 1 tail marker, 1 rod marker
        l_start_row = [[0, 0]]*5  # 3 head markers, 1 tail marker, 1 rod marker
        l_conf_counter = 0
        r_conf_counter = 0
        no_detect = [-1, -1]

        out_df = pd.DataFrame(columns=['frame_idx', 'l_filled', 'lhead_l', 'lhead_r', 'lhead_c', 'ltail', 'lrod',
                                       'r_filled', 'rhead_l', 'rhead_r', 'rhead_c', 'rtail', 'rrod'])

        for idx, frame in enumerate(frame_lbls.values[:][2:]):
            conf_flag = True
            l_row = [idx]
            r_row = []
            # labels: HL 0:3, HR 3:6, HC 6:9, T 9:12, R 12:15
            petri1 = frame[1:16].astype(np.float32)
            petri2 = frame[16:].astype(np.float32)
            if np.min(petri1) > self.CONF_THRESHOLD:
                l_row.append(0)  # not filled
                l_row.append(petri1[0:2])  # head l
                l_row.append(petri1[3:5])  # head r
                l_row.append(petri1[6:8])  # head c
                l_row.append(petri1[9:11])  # tail
                l_row.append(petri1[12:14])  # rod
                l_row_buffer = self._fill_missing_frames(l_row_buffer, l_row, l_conf_counter, l_start_row)
                l_start_row = l_row[-5:]
                l_conf_counter = 0
            else:
                conf_flag = False
                l_conf_counter += 1
                l_row.append(1)  # filled in coordinates
                l_row += [no_detect]*5
            l_row_buffer.append(l_row)

            # if any labels are missing, frame will be disregarded
            if np.min(petri2) > self.CONF_THRESHOLD:
                r_row.append(0)
                r_row.append(petri2[0:2])
                r_row.append(petri2[3:5])
                r_row.append(petri2[6:8])
                r_row.append(petri2[9:11])
                r_row.append(petri2[12:14])
                r_row_buffer = self._fill_missing_frames(r_row_buffer, r_row, r_conf_counter, r_start_row)
                r_start_row = r_row[-5:]
                r_conf_counter = 0
            else:
                conf_flag = False
                r_conf_counter += 1
                r_row.append(1)
                r_row += [no_detect] * 5
            r_row_buffer.append(r_row)

            if conf_flag or len(frame_lbls.values[:])-1 == idx:
                assert len(l_row_buffer) == len(r_row_buffer)
                for i in range(len(l_row_buffer)):
                    tmp_row = l_row_buffer[i] + r_row_buffer[i]
                    out_df.loc[tmp_row[0]] = np.array(tmp_row)
                l_row_buffer = []
                r_row_buffer = []
        return out_df

    def _get_metrics(self, filled_df, vid_title=None):
        left_in_radius = False
        left_fish = {'in_time': 0,
                     'out_time': 0,
                     'in_facing_left': 0,
                     'in_facing_right': 0,
                     'out_facing_left': 0,
                     'out_facing_right': 0,
                     'left_approach': 0,
                     'right_approach': 0}

        right_in_radius = False
        right_fish = {'in_time': 0,
                      'out_time': 0,
                      'in_facing_left': 0,
                      'in_facing_right': 0,
                      'out_facing_left': 0,
                      'out_facing_right': 0,
                      'left_approach': 0,
                      'right_approach': 0}

        l_approach_duration = 0
        l_side_buffer = None
        r_approach_duration = 0
        r_side_buffer = None

        left_cols = [col for col in filled_df.columns if col.startswith('l')]
        right_cols = [col for col in filled_df.columns if col.startswith('r')]

        if self.FRAME_FILE is not None:
            l_snip = self.FRAME_SNIPS[vid_title]['l_fish']
            l_idx_list = [i for i in range(l_snip[0], l_snip[1]+1)]
            r_snip = self.FRAME_SNIPS[vid_title]['r_fish']
            r_idx_list = [i for i in range(r_snip[0], r_snip[1]+1)]

        for idx, frame in filled_df.iterrows():
            petri1 = frame.loc[left_cols]
            petri2 = frame[right_cols]

            # LEFT FISH
            # fish was in radius in previous frame
            if self.FRAME_FILE is None or idx in l_idx_list:
                if left_in_radius:
                    if self._is_in_radius(petri1, 'l') == 0:
                        left_in_radius = False
                        l_approach_duration = 0
                        left_fish['out_time'] += 1
                        if self._is_left_facing(petri1, 'l') == 1:
                            left_fish['out_facing_left'] += 1
                        else:
                            left_fish['out_facing_right'] += 1
                    else:
                        left_fish['in_time'] += 1
                        l_approach_duration += 1
                        if l_approach_duration == self.FRAME_RATE//2:
                            if l_side_buffer == 'left':
                                left_fish['left_approach'] += 1
                            elif l_side_buffer == 'right':
                                left_fish['right_approach'] += 1
                            l_side_buffer = None
                        if self._is_left_facing(petri1, 'l') == 1:
                            left_fish['in_facing_left'] += 1
                        else:
                            left_fish['in_facing_right'] += 1

                # fish wasn't in radius in previous frame
                else:
                    if self._is_in_radius(petri1, 'l') == 1:
                        if self._is_left_facing(petri1, 'l') == 1:
                            l_side_buffer = 'left'
                            l_approach_duration += 1
                            left_fish['in_time'] += 1
                            left_fish['in_facing_left'] += 1
                            left_in_radius = True
                        else:
                            l_side_buffer = 'right'
                            left_fish['in_time'] += 1
                            left_fish['in_facing_right'] += 1
                            left_in_radius = True
                    else:
                        left_fish['out_time'] += 1
                        l_approach_duration = 0
                        if self._is_left_facing(petri1, 'l') == 1:
                            left_fish['out_facing_left'] += 1
                        else:
                            left_fish['out_facing_right'] += 1

            # RIGHT FISH
            # fish was in radius in previous frame
            if self.FRAME_FILE is None or idx in r_idx_list:
                if right_in_radius:
                    if self._is_in_radius(petri2, 'r') == 0:
                        right_in_radius = False
                        r_approach_duration = 0
                        right_fish['out_time'] += 1
                        if self._is_left_facing(petri2, 'r') == 1:
                            right_fish['out_facing_left'] += 1
                        else:
                            right_fish['out_facing_right'] += 1
                    else:
                        right_fish['in_time'] += 1
                        r_approach_duration += 1
                        if r_approach_duration == self.FRAME_RATE//2:
                            if r_side_buffer == 'left':
                                right_fish['left_approach'] += 1
                            elif r_side_buffer == 'right':
                                right_fish['right_approach'] += 1
                            r_side_buffer = None
                        if self._is_left_facing(petri2, 'r') == 1:
                            right_fish['in_facing_left'] += 1
                        else:
                            right_fish['in_facing_right'] += 1

                # fish wasn't in radius in previous frame
                else:
                    if self._is_in_radius(petri2, 'r') == 1:
                        if self._is_left_facing(petri2, 'r') == 1:
                            r_side_buffer = 'left'
                            r_approach_duration += 1
                            right_fish['in_time'] += 1
                            right_fish['in_facing_left'] += 1
                            right_in_radius = True
                        else:
                            r_side_buffer = 'right'
                            right_fish['in_time'] += 1
                            right_fish['in_facing_right'] += 1
                            right_in_radius = True
                    else:
                        right_fish['out_time'] += 1
                        r_approach_duration = 0
                        if self._is_left_facing(petri2, 'r') == 1:
                            right_fish['out_facing_left'] += 1
                        else:
                            right_fish['out_facing_right'] += 1
        return left_fish, right_fish

    def walk_dir(self):
        vid_dirs = [os.path.join(self.SOURCE_DIR, p) for p in os.listdir(self.SOURCE_DIR) if
                    os.path.isdir(os.path.join(self.SOURCE_DIR, p))]
        sum_df = None
        for rad in self.RADIUS_LIST:
            print(f'Analysis for radius: {rad}mm')
            self.approach_radius = rad
            rad_results = pd.DataFrame(columns=[f'vid_fish',
                                                f'left_approaches({self.approach_radius})',
                                                f'right_approaches({self.approach_radius})'])
            for vid_data in tqdm(vid_dirs):
                vdata = ""
                vname = os.path.basename(vid_data).split('_')[0]
                vid_files = [f for f in os.listdir(vid_data)]
                for f in vid_files:
                    if f.startswith(vname) and f.endswith('.csv'):
                        vdata = os.path.join(vid_data, f)

                assert len(vdata) > 0  # check if some result file was found

                if self.FRAME_FILE is None or os.path.basename(vid_data).split('.')[0] in self.FRAME_SNIPS.keys():
                    vdata = pd.read_csv(vdata)
                    vdata = self._replace_low_conf(vdata)
                    l_results, r_results = self._get_metrics(vdata, os.path.basename(vid_data).split('.')[0])

                    l_row = {f'vid_fish': f'{vname}: Fish 1',
                             f'left_approaches({self.approach_radius})': l_results['left_approach'],
                             f'right_approaches({self.approach_radius})': l_results['right_approach']}
                    r_row = {f'vid_fish': f'{vname}: Fish 2',
                             f'left_approaches({self.approach_radius})': r_results['left_approach'],
                             f'right_approaches({self.approach_radius})': r_results['right_approach']}

                rad_results.loc[rad_results['vid_fish'].count()] = l_row
                rad_results.loc[rad_results['vid_fish'].count()] = r_row

            rad_results.set_index('vid_fish', inplace=True)

            if sum_df is None:
                sum_df = rad_results.copy()
            else:
                sum_df = sum_df.merge(rad_results.copy(), left_index=True, right_index=True, how='left', sort=False)
            assert not sum_df.isnull().values.any()  # check that same files were analyzed

            del rad_results

        if self.FRAME_FILE is not None:
            sname = 'radius_frame_analysis.csv'
        else:
            sname = 'radius_analysis.csv'
        sum_df.to_csv(os.path.join(self.TARGET_DIR, sname))
        return os.path.join(self.TARGET_DIR, sname)


if __name__ == '__main__':
    # conf_file, save_dir, approach_radius
    parser = argparse.ArgumentParser(description='Analyze directory containing subdirectories of deeplabcut result '
                                                 'files')

    parser.add_argument(
        '--radius-list', '-r', type=int, nargs='+',
        help='List of radii to check (in mm). Specify as space-separated integers (e.g. -r 10 15 20)'
    )

    parser.add_argument(
        '--frame_rate', '-f', type=int, required=False, default=10,
        help='Frame rate at which videos were recorded (default is 10).'
    )

    parser.add_argument(
        '--data_dir', '-d', type=str, required=False, default=os.getcwd(),
        help='Directory containing video analysis subdirectories (default is current directory).'
    )

    parser.add_argument(
        '--save_dir', '-s', type=str, required=False, default=os.getcwd(),
        help='Directory where results will be stored (default is current directory).'
    )

    parser.add_argument(
        '--frame_file', '-x', type=str, required=False, default=None,
        help='File containing subsections of videos to analyze.'
    )

    usr_args = vars(parser.parse_args())

    walker = BatchAnalyzer(**usr_args)
    save_file = walker.walk_dir()
    print(f'Success! Results are saved at {save_file}.')


# (venv) C:\Users\yanni\Desktop\tm\FishProject>python AnalyzeDir.py -r 1 30 -f 10 -d C:/Users/yanni/Desktop/tm/BulkResults -s ./











