import os
import cv2
import argparse
import deeplabcut
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence


class Model:
    def __init__(self, conf_file, save_dir, approach_radius, video_folder):
        self.LABELS = ['Head1_L', 'Head1_R', 'Head1_C', 'Tail1', 'Rod1',
                       'Head2_L', 'Head2_R', 'Head2_C', 'Tail2', 'Rod2']
        self.RESULT_POSTFIX = 'DeepCut_resnet50_FishApproachMay7shuffle1_650000.h5'
        self.FRAME_RATE = 10
        # Image petri diameter = 188px
        # Real petri diameter = 94mm
        self.SIZE_RATIO = 2
        self.CONF_THRESHOLD = 0.15
        self.conf = conf_file
        self.save_dir = save_dir
        self.approach_radius = approach_radius


    def _format_tif(self, tif_file, dest_folder):
        im_stack = Image.open(tif_file)
        vid_name = dest_folder + '/' + ''.join(os.path.basename(tif_file).split('.')[:-1]) + '.mp4'
        for idx, img in enumerate(ImageSequence.Iterator(im_stack)):
            test = img
            img_data = np.array(test)
            if idx == 0:
                h, w = img_data.shape
                out_vid = cv2.VideoWriter(vid_name, -1, self.FRAME_RATE, (w, h))
            out_vid.write(img_data)

        cv2.destroyAllWindows()
        out_vid.release()
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

    def _get_metrics(self, frame_lbls):
        left_in_radius = False
        left_fish = {'in_time': 0,
                     'out_time': 0,
                     'facing_left': 0,
                     'facing_right': 0,
                     'left_approach': 0,
                     'right_approach': 0}

        right_in_radius = False
        right_fish = {'in_time': 0,
                     'out_time': 0,
                     'facing_left': 0,
                     'facing_right': 0,
                     'left_approach': 0,
                     'right_approach': 0}


        for frame in frame_lbls.values[:]:
            # labels: HL 0:3, HR 3:6, HC 6:9, T 9:12, R 12:15
            petri1 = frame[:15]
            petri2 = frame[15:]

            # if any labels are missing, frame will be disregarded
            if np.min(petri1) > self.CONF_THRESHOLD:
                if self._facing_side(petri1) == 1:
                    left_fish['facing_left'] += 1
                if self._facing_side(petri1) == -1:
                    left_fish['facing_right'] += 1

                # fish was in radius in previous frame
                if left_in_radius:
                    if self._approach_side(petri1) == 0:
                        left_in_radius = False
                        left_fish['out_time'] += 1
                    else:
                        left_fish['in_time'] += 1

                # fish wasn't in radius in previous frame
                else:
                    if self._approach_side(petri1) == 1:
                        left_fish['left_approach'] += 1
                        left_fish['in_time'] += 1
                        left_in_radius = True
                    elif self._approach_side(petri1) == -1:
                        left_fish['right_approach'] += 1
                        left_fish['in_time'] += 1
                        left_in_radius = True
                    else:
                        left_fish['out_time'] += 1

            # if any labels are missing, frame will be disregarded
            if np.min(petri2) > self.CONF_THRESHOLD:
                if self._facing_side(petri2) == 1:
                    right_fish['facing_left'] += 1
                if self._facing_side(petri2) == -1:
                    right_fish['facing_right'] += 1

                # fish was in radius in previous frame
                if right_in_radius:
                    if self._approach_side(petri1) == 0:
                        right_in_radius = False
                        right_fish['out_time'] += 1
                    else:
                        right_fish['in_time'] += 1

                # fish wasn't in radius in previous frame
                else:
                    if self._approach_side(petri1) == 1:
                        right_fish['left_approach'] += 1
                        right_fish['in_time'] += 1
                        right_in_radius = True
                    elif self._approach_side(petri1) == -1:
                        right_fish['right_approach'] += 1
                        right_fish['in_time'] += 1
                        right_in_radius = True
                    else:
                        right_fish['out_time'] += 1
        return left_fish, right_fish

    def _print_results(self, left_dict, right_dict):

        print('Left Petri Dish:')
        print(f'\tTime spent in radius:\t\t\t{left_dict["in_time"]/self.FRAME_RATE}')
        print(f'\tTime spent outside of radius:\t\t{left_dict["out_time"]/self.FRAME_RATE}')
        print(f'\tTime spent with left size facing rod:\t{left_dict["facing_left"]/self.FRAME_RATE}')
        print(f'\tTime spent with right size facing rod:\t{left_dict["facing_right"]/self.FRAME_RATE}')
        print(f'\tTimes approaching rod with left side:\t{left_dict["left_approach"]}')
        print(f'\tTimes approaching rod with right side:\t{left_dict["right_approach"]}')

        print('Right Petri Dish:')
        print(f'\tTime spent in radius:\t\t\t{right_dict["in_time"]/self.FRAME_RATE}')
        print(f'\tTime spent outside of radius:\t\t{right_dict["out_time"]/self.FRAME_RATE}')
        print(f'\tTime spent with left size facing rod:\t{right_dict["facing_left"]/self.FRAME_RATE}')
        print(f'\tTime spent with right size facing rod:\t{right_dict["facing_right"]/self.FRAME_RATE}')
        print(f'\tTimes approaching rod with left side:\t{right_dict["left_approach"]}')
        print(f'\tTimes approaching rod with right side:\t{right_dict["right_approach"]}')


    def analyze_video(self, tif_file, del_video=True, del_results=True):
        result_dir = self.save_dir + '/' + ''.join(os.path.basename(tif_file).split('.')[:-1]) + '/'
        try:
            os.mkdir(result_dir)
        except:
            pass
        vid_file = self._format_tif(tif_file, result_dir)
        try:
            deeplabcut.analyze_videos(self.conf, [vid_file], destfolder=result_dir, save_as_csv=True)
        except:
            import pdb
            pdb.set_trace()
        deeplabcut.create_labeled_video(self.conf, [vid_file], destfolder=result_dir)

        result_file = result_dir + ''.join(os.path.basename(tif_file).split('.')[:-1]) + self.RESULT_POSTFIX
        results = pd.read_hdf(result_file, 'df_with_missing')

        left_results, right_results = self._get_metrics(results)
        self._print_results(left_results, right_results)

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
        '--video_folder', '-v', type=str, required=False, default=None,
        help='Folder containing videos to be analyzed. If not specified, user will be prompted for individual videos'
    )

    parser.add_argument(
        '--save_dir', '-s', type=str, required=False, default=os.getcwd(),
        help='Directory where results will be stored (default is current directory)'
    )

    parser.add_argument(
        '--conf_file', '-c', type=str, required=False,
        default=os.getcwd() + '/DeepLabCutModel/FishApproach-Nick-2019-05-07/config.yaml',
        help='Config file, it is recommended to leave this unchanged for now'
    )

    usr_args = vars(parser.parse_args())

    model = Model(**usr_args)
    if usr_args['video_folder']:
        for vid_path in [f for f in os.listdir(usr_args['video_folder'])]:
            model.analyze_video(usr_args['video_folder'] + '/' + vid_path)
    else:
        while True:
            vid_path = input('Enter video path or press Ctrl+C to quit:')
            model.analyze_video(vid_path)