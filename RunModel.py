import os
import argparse
import subprocess
import deeplabcut
import numpy as np
import pandas as pd


class Model:
    def __init__(self, conf_file, save_dir, approach_radius, video_folder, frame_rate):
        self.LABELS = ['Head1_L', 'Head1_R', 'Head1_C', 'Tail1', 'Rod1',
                       'Head2_L', 'Head2_R', 'Head2_C', 'Tail2', 'Rod2']
        self.RESULT_POSTFIX = 'DeepCut_resnet50_FishApproachJun24shuffle1_900000.h5'
        self.NEW_DIMS = {'width':  480,
                         'height': 270}
        # After resize:
        # Image petri diameter = 181px
        # Real petri diameter = 94mm
        self.SIZE_RATIO = 1.93
        self.CONF_THRESHOLD = 0.15
        self.FRAME_RATE = frame_rate
        self.conf = conf_file
        self.save_dir = save_dir
        self.approach_radius = approach_radius
        # Duration in radius, left - facing
        # duration in radius, right - facing
        # duration in radius, left - facing
        # duration
        # outside
        # of
        # radius, right - facing
        # duration
        # outside
        # of
        # radius, Times
        # approaching
        # with left side, Times approaching with right side
        self.sum_file= pd.DataFrame(columns=['Subject',
                                             'Duration in radius - left-facing',
                                             'Duration in radius = right-facing',
                                             'Duration outside radius - left-facing',
                                             'Duration outside radius - right-facing',
                                             'Left approaches',
                                             'Right approaches'
                                             ])

    def _resize_avi(self, avi_file, dest_folder):
        if not (avi_file.startswith('/') or avi_file[1]==':'):  # not an absolute path
            pwd = os.getcwd()
            avi_file = pwd + '/' + avi_file

        if not (dest_folder.endswith('\\') or dest_folder.endswith('/')):
            dest_folder += '/'
        avi_file = avi_file.replace('\\', '/')
        vid_name = f'{dest_folder + "".join(avi_file.split("/")[-1].split(".")[:-1])}_resized.avi'
        subprocess.call(['ffmpeg',
                         '-i', avi_file,
                         '-vf', f'scale={self.NEW_DIMS["width"]}:{self.NEW_DIMS["height"]}',
                         vid_name])
        return vid_name

    def _get_distance(self, x1, y1, x2, y2):
        return np.sqrt(np.square(x1-x2) + np.square(y1-y2)) / self.SIZE_RATIO

    def _facing_side(self, frame_lbls):
        # which side of fish faces rod
        left_d = self._get_distance(frame_lbls[0], frame_lbls[1],
                                    frame_lbls[12], frame_lbls[13])
        right_d = self._get_distance(frame_lbls[3], frame_lbls[4],
                                     frame_lbls[12], frame_lbls[13])

        if left_d < right_d:
            return 1
        elif left_d > right_d:
            return -1
        else:
            return 0

    def _approach_side(self, frame_lbls):
        # which side of fish (if any) crosses approach-radius
        left_d = self._get_distance(frame_lbls[0], frame_lbls[1],
                                    frame_lbls[12], frame_lbls[13])
        right_d = self._get_distance(frame_lbls[3], frame_lbls[4],
                                     frame_lbls[12], frame_lbls[13])

        if left_d < self.approach_radius:
            return 1
        elif right_d < self.approach_radius:
            return -1
        else:
            return 0

    def _get_metrics(self, frame_lbls, save_dir):
        out_df = pd.DataFrame(columns=['frame_idx',
                                       'l_in_radius', 'l_left_approaches', 'l_right_approaches', 'l_in_time',
                                       'l_out_time',
                                       'l_in_left_time', 'l_in_right_time',
                                       'l_out_left_time', 'l_out_right_time',
                                       'l_left_head', 'l_right_head', 'l_center_head', 'l_rod',
                                       'r_in_radius', 'r_left_approaches', 'r_right_approaches', 'r_in_time',
                                       'r_out_time',
                                       'r_in_left_time', 'r_in_right_time',
                                       'r_out_left_time', 'r_out_right_time',
                                       'r_left_head', 'r_right_head', 'r_center_head', 'r_rod'])
        no_detect = [-1, -1]

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


        for idx, frame in enumerate(frame_lbls.values[:]):
            row = [idx]
            # labels: HL 0:3, HR 3:6, HC 6:9, T 9:12, R 12:15
            petri1 = frame[:15]
            petri2 = frame[15:]

            # if any labels are missing, frame will be disregarded
            if np.min(petri1) > self.CONF_THRESHOLD:
                # fish was in radius in previous frame
                if left_in_radius:
                    if self._approach_side(petri1) == 0:
                        left_in_radius = False
                        left_fish['out_time'] += 1
                        if self._facing_side(petri1) == 1:
                            left_fish['out_facing_left'] += 1
                        if self._facing_side(petri1) == -1:
                            left_fish['out_facing_right'] += 1
                    else:
                        left_fish['in_time'] += 1
                        if self._facing_side(petri1) == 1:
                            left_fish['in_facing_left'] += 1
                        if self._facing_side(petri1) == -1:
                            left_fish['in_facing_right'] += 1

                # fish wasn't in radius in previous frame
                else:
                    if self._approach_side(petri1) == 1:
                        left_fish['left_approach'] += 1
                        left_fish['in_time'] += 1
                        left_fish['in_facing_left'] += 1
                        left_in_radius = True
                    elif self._approach_side(petri1) == -1:
                        left_fish['right_approach'] += 1
                        left_fish['in_time'] += 1
                        left_fish['in_facing_right'] += 1
                        left_in_radius = True
                    else:
                        left_fish['out_time'] += 1
                        if self._facing_side(petri1) == 1:
                            left_fish['out_facing_left'] += 1
                        if self._facing_side(petri1) == -1:
                            left_fish['out_facing_right'] += 1
                row.append(int(left_in_radius))
                row.append(left_fish['left_approach'])
                row.append(left_fish['right_approach'])
                row.append(round(left_fish['in_time']/self.FRAME_RATE, 2))
                row.append(round(left_fish['out_time']/self.FRAME_RATE, 2))
                row.append(round(left_fish['in_facing_left']/self.FRAME_RATE, 2))
                row.append(round(left_fish['in_facing_right']/self.FRAME_RATE, 2))
                row.append(round(left_fish['out_facing_left']/self.FRAME_RATE, 2))
                row.append(round(left_fish['out_facing_right']/self.FRAME_RATE, 2))
                row.append(petri1[0:2])
                row.append(petri1[3:5])
                row.append(petri1[6:8])
                row.append(petri1[12:14])
            else:
                row.append(-1)
                row.append(left_fish['left_approach'])
                row.append(left_fish['right_approach'])
                row.append(round(left_fish['in_time']/self.FRAME_RATE, 2))
                row.append(round(left_fish['out_time']/self.FRAME_RATE, 2))
                row.append(round(left_fish['in_facing_left']/self.FRAME_RATE, 2))
                row.append(round(left_fish['in_facing_right']/self.FRAME_RATE, 2))
                row.append(round(left_fish['out_facing_left']/self.FRAME_RATE, 2))
                row.append(round(left_fish['out_facing_right']/self.FRAME_RATE, 2))
                row += [no_detect]*4

            # if any labels are missing, frame will be disregarded
            if np.min(petri2) > self.CONF_THRESHOLD:
                # fish was in radius in previous frame
                if right_in_radius:
                    if self._approach_side(petri2) == 0:
                        right_in_radius = False
                        right_fish['out_time'] += 1
                        if self._facing_side(petri2) == 1:
                            right_fish['out_facing_left'] += 1
                        if self._facing_side(petri2) == -1:
                            right_fish['out_facing_right'] += 1
                    else:
                        right_fish['in_time'] += 1
                        if self._facing_side(petri2) == 1:
                            right_fish['in_facing_left'] += 1
                        if self._facing_side(petri2) == -1:
                            right_fish['in_facing_right'] += 1

                # fish wasn't in radius in previous frame
                else:
                    if self._approach_side(petri2) == 1:
                        right_fish['left_approach'] += 1
                        right_fish['in_time'] += 1
                        right_fish['in_facing_left'] += 1
                        right_in_radius = True
                    elif self._approach_side(petri2) == -1:
                        right_fish['right_approach'] += 1
                        right_fish['in_time'] += 1
                        right_fish['in_facing_right'] += 1
                        right_in_radius = True
                    else:
                        right_fish['out_time'] += 1
                        if self._facing_side(petri2) == 1:
                            right_fish['out_facing_left'] += 1
                        if self._facing_side(petri2) == -1:
                            right_fish['out_facing_right'] += 1

                row.append(int(right_in_radius))
                row.append(right_fish['left_approach'])
                row.append(right_fish['right_approach'])
                row.append(round(right_fish['in_time']/self.FRAME_RATE, 2))
                row.append(round(right_fish['out_time']/self.FRAME_RATE, 2))
                row.append(round(right_fish['in_facing_left']/self.FRAME_RATE, 2))
                row.append(round(right_fish['in_facing_right']/self.FRAME_RATE, 2))
                row.append(round(right_fish['out_facing_left']/self.FRAME_RATE, 2))
                row.append(round(right_fish['out_facing_right']/self.FRAME_RATE, 2))
                row.append(petri2[0:2])
                row.append(petri2[3:5])
                row.append(petri2[6:8])
                row.append(petri2[12:14])
            else:
                row.append(-1)
                row.append(right_fish['left_approach'])
                row.append(right_fish['right_approach'])
                row.append(round(right_fish['in_time']/self.FRAME_RATE, 2))
                row.append(round(right_fish['out_time']/self.FRAME_RATE, 2))
                row.append(round(right_fish['in_facing_left']/self.FRAME_RATE, 2))
                row.append(round(right_fish['in_facing_right']/self.FRAME_RATE, 2))
                row.append(round(right_fish['out_facing_left']/self.FRAME_RATE, 2))
                row.append(round(right_fish['out_facing_right']/self.FRAME_RATE, 2))
                row += [no_detect] * 4
            out_df.loc[idx] = np.array(row)
        out_df.to_csv(save_dir + 'approach_results.csv')
        return left_fish, right_fish

    def _quick_results(self, left_dict, right_dict):
        print('Left Petri Dish:')
        print(f'\tTime spent in radius:\t\t\t{left_dict["in_time"]/self.FRAME_RATE}')
        print(f'\tTime spent inside with left size facing rod:\t{left_dict["in_facing_left"]/self.FRAME_RATE}')
        print(f'\tTime spent inside with right size facing rod:\t{left_dict["in_facing_right"]/self.FRAME_RATE}')
        print(f'\tTime spent outside of radius:\t\t{left_dict["out_time"]/self.FRAME_RATE}')
        print(f'\tTime spent outside with left size facing rod:\t{left_dict["out_facing_left"]/self.FRAME_RATE}')
        print(f'\tTime spent outside with right size facing rod:\t{left_dict["out_facing_right"]/self.FRAME_RATE}')
        print(f'\tTimes approaching rod with left side:\t{left_dict["left_approach"]}')
        print(f'\tTimes approaching rod with right side:\t{left_dict["right_approach"]}')

        print('Right Petri Dish:')
        print(f'\tTime spent in radius:\t\t\t{right_dict["in_time"]/self.FRAME_RATE}')
        print(f'\tTime spent inside with left size facing rod:\t{right_dict["in_facing_left"]/self.FRAME_RATE}')
        print(f'\tTime spent inside with right size facing rod:\t{right_dict["in_facing_right"]/self.FRAME_RATE}')
        print(f'\tTime spent outside of radius:\t\t{right_dict["out_time"]/self.FRAME_RATE}')
        print(f'\tTime spent outside with left size facing rod:\t{right_dict["out_facing_left"]/self.FRAME_RATE}')
        print(f'\tTime spent outside with right size facing rod:\t{right_dict["out_facing_right"]/self.FRAME_RATE}')
        print(f'\tTimes approaching rod with left side:\t{right_dict["left_approach"]}')
        print(f'\tTimes approaching rod with right side:\t{right_dict["right_approach"]}')


    def _add_to_summary_file(self, left_dict, right_dict, fname):
        lrow = {'Subject': f'{fname}_left',
               'Duration in radius - left-facing': left_dict["in_facing_left"]/self.FRAME_RATE,
               'Duration in radius = right-facing': left_dict["in_facing_right"]/self.FRAME_RATE,
               'Duration outside radius - left-facing': left_dict["out_facing_left"]/self.FRAME_RATE,
               'Duration outside radius - right-facing': left_dict["out_facing_right"]/self.FRAME_RATE,
               'Left approaches': left_dict["left_approach"],
               'Right approaches': left_dict["right_approach"]}
        rrow = {'Subject': f'{fname}_right',
               'Duration in radius - left-facing': right_dict["in_facing_left"]/self.FRAME_RATE,
               'Duration in radius = right-facing': right_dict["in_facing_right"]/self.FRAME_RATE,
               'Duration outside radius - left-facing': right_dict["out_facing_left"]/self.FRAME_RATE,
               'Duration outside radius - right-facing': right_dict["out_facing_right"]/self.FRAME_RATE,
               'Left approaches': right_dict["left_approach"],
               'Right approaches': right_dict["right_approach"]}

        self.sum_file = self.sum_file.append([lrow, rrow], ignore_index=True)

    def analyze_video(self, avi_file, del_video=False, del_results=True):
        if self.save_dir.endswith('/'):
            result_dir = self.save_dir + ''.join(os.path.basename(avi_file).split('.')[:-1]) + '/'
        else:
            result_dir = self.save_dir + '/' + ''.join(os.path.basename(avi_file).split('.')[:-1]) + '/'

        try:
            os.mkdir(result_dir)
        except:
            pass

        if avi_file.endswith('.avi'):
            vid_file = self._resize_avi(avi_file, result_dir)
        else:
            raise Exception('Unknown video format. Support is limited to tif or avi')

        try:
            deeplabcut.analyze_videos(self.conf, [vid_file], destfolder=result_dir, save_as_csv=True)
        except:
            print("Problem analyzing video. Check if config.yaml file was adjusted properly.")
        deeplabcut.create_labeled_video(self.conf, [vid_file], destfolder=result_dir)

        result_file = result_dir + ''.join(os.path.basename(vid_file).split('.')[:-1]) + self.RESULT_POSTFIX
        results = pd.read_hdf(result_file, 'df_with_missing')

        left_results, right_results = self._get_metrics(results, result_dir)
        self._quick_results(left_results, right_results)
        self._add_to_summary_file(left_results, right_results, ''.join(os.path.basename(avi_file).split('.')[:-1]))

        if del_video:
            os.remove(vid_file)
        if del_results:
            os.remove(result_file)
        os.remove(result_file.replace('.h5', 'includingmetadata.pickle'))


if __name__ == '__main__':
    # conf_file, save_dir, approach_radius
    parser = argparse.ArgumentParser(description='Analyze cavefish behavior in tif files')

    parser.add_argument(
        'approach_radius', type=int,
        help='Radius around rod that will be considered as an approach if crossed (in mm)'
    )

    parser.add_argument(
        '--frame_rate', '-f', type=int, required=False, default=10,
        help='Frame rate at which videos were recorded. Default is 10'
    )

    parser.add_argument(
        '--video_folder', '-v', type=str, required=False, default=None,
        help='Folder containing videos to be analyzed. If not specified, user will be prompted for individual videos'
    )

    parser.add_argument(
        '--save_dir', '-s', type=str, required=False, default=os.getcwd(),
        help='Directory where results will be stored (default is current directory)'
    )

    parser.add_argument(
        '--conf_file', '-c', type=str, required=False,
        default=os.getcwd() + '/DeepLabCutModel/FishApproach-Nick-2019-06-24/config.yaml',
        help='Config file, it is recommended to leave this unchanged for now'
    )

    usr_args = vars(parser.parse_args())

    model = Model(**usr_args)
    if usr_args['video_folder']:
        for vid_path in [f for f in os.listdir(usr_args['video_folder'])]:
            if usr_args['video_folder'].endswith('/') or usr_args['video_folder'].endswith('\\'):
                model.analyze_video(usr_args['video_folder'][:-1] + '/' + vid_path)
            else:
                model.analyze_video(usr_args['video_folder'] + '/' + vid_path)
        if usr_args['save_dir'].endswith('\\') or usr_args['save_dir'].endswith('/'):
            model.sum_file.to_csv(usr_args['save_dir']+'summary_results.csv', index=False)
        else:
            model.sum_file.to_csv(usr_args['save_dir'] + '/summary_results.csv', index=False)
    else:
        while True:
            vid_path = input('Enter video path or press Ctrl+C to quit:')
            model.analyze_video(vid_path)
            if usr_args['save_dir'].endswith('\\') or usr_args['save_dir'].endswith('/'):
                model.sum_file.to_csv(usr_args['save_dir']+'summary_results.csv', index=False)
            else:
                model.sum_file.to_csv(usr_args['save_dir'] + '/summary_results.csv', index=False)
