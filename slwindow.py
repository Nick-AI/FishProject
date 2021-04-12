import os
import re
import argparse
import numpy as np
import pandas as pd


class BackFiller:
    def __init__(self, approach_radius, frame_rate, result_folder, dims_x, ratio_pixmm,
                 frame_file=None, window_size=10):
        self.size_ratio = ratio_pixmm * (480 / dims_x)

        self.approach_radius = approach_radius
        self.frame_rate = frame_rate
        self.window_size = window_size

        self.frame_file = frame_file
        if self.frame_file is not None:
            self.frame_snips = {}
            self._parse_frames(self.frame_file)

        self.sum_file = pd.DataFrame(
            columns=[
                "Subject",
                "Duration in radius - left-facing",
                "Duration in radius - right-facing",
                "Duration outside radius - left-facing",
                "Duration outside radius - right-facing",
                "Left approaches",
                "Right approaches",
            ]
        )

    @staticmethod
    def _comp_helper(vals, targ):
        for v in vals:
            if v != targ:
                return False
        return True
    
    def _parse_frames(self, source):
        with open(source, "r") as f:
            
            for idx, line in enumerate(f):
                
                # row specifying video name
                if idx % 3 == 0:
                    vname = line.strip()
                    self.frame_snips[vname] = {"l_fish": None, "r_fish": None}

                # row specifying left fish frames
                elif (idx - 1) % 3 == 0:
                    delin = line.split(" ")
                    self.frame_snips[vname]["l_fish"] = [int(delin[0]), int(delin[1])]

                elif (idx - 2) % 3 == 0:
                    delin = line.split(" ")
                    self.frame_snips[vname]["r_fish"] = [int(delin[0]), int(delin[1])]

    def _get_distance(self, x1, y1, x2, y2):
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)) / self.size_ratio

    def _is_left_facing(self, row, side_prefix):

        # which side of fish faces rod
        left_d = self._get_distance(
            row[f"{side_prefix}head_l"][0],
            row[f"{side_prefix}head_l"][1],
            row[f"{side_prefix}rod"][0],
            row[f"{side_prefix}rod"][1],
        )

        right_d = self._get_distance(
            row[f"{side_prefix}head_r"][0],
            row[f"{side_prefix}head_r"][1],
            row[f"{side_prefix}rod"][0],
            row[f"{side_prefix}rod"][1],
        )

        if left_d < right_d:
            return 1

        return 0

    def _is_in_radius(self, row, side_prefix):

        # which side of fish (if any) crosses approach-radius
        left_d = self._get_distance(
            row[f"{side_prefix}head_l"][0],
            row[f"{side_prefix}head_l"][1],
            row[f"{side_prefix}rod"][0],
            row[f"{side_prefix}rod"][1],
        )

        right_d = self._get_distance(
            row[f"{side_prefix}head_r"][0],
            row[f"{side_prefix}head_r"][1],
            row[f"{side_prefix}rod"][0],
            row[f"{side_prefix}rod"][1],
        )

        if left_d < self.approach_radius or right_d < self.approach_radius:
            return 1

        return 0

    def _get_metrics(self, filled_df, save_dir):

        out_df = pd.DataFrame(
            columns=[
                "frame_idx",
                "l_in_radius",
                "l_left_approaches",
                "l_right_approaches",
                "l_in_time",
                "l_out_time",
                "l_in_left_time",
                "l_in_right_time",
                "l_out_left_time",
                "l_out_right_time",
                "l_left_head",
                "l_right_head",
                "l_center_head",
                "l_rod",
                "l_coords_filled",
                "r_in_radius",
                "r_left_approaches",
                "r_right_approaches",
                "r_in_time",
                "r_out_time",
                "r_in_left_time",
                "r_in_right_time",
                "r_out_left_time",
                "r_out_right_time",
                "r_left_head",
                "r_right_head",
                "r_center_head",
                "r_rod",
                "r_coords_filled",
            ]
        )

        left_in_radius = False

        left_fish = {
            "in_time": 0,
            "out_time": 0,
            "in_facing_left": 0,
            "in_facing_right": 0,
            "out_facing_left": 0,
            "out_facing_right": 0,
            "left_approach": 0,
            "right_approach": 0,
            "distance": 0,
        }

        right_in_radius = False

        right_fish = {
            "in_time": 0,
            "out_time": 0,
            "in_facing_left": 0,
            "in_facing_right": 0,
            "out_facing_left": 0,
            "out_facing_right": 0,
            "left_approach": 0,
            "right_approach": 0,
            "distance": 0,
        }

        l_approach_duration = 0
        l_side_buffer = None
        r_approach_duration = 0
        r_side_buffer = None
        left_cols = [col for col in filled_df.columns if col.startswith("l")]
        right_cols = [col for col in filled_df.columns if col.startswith("r")]
        vid_key = save_dir.split("/")[-2]

        if self.frame_file is not None:
            l_snip = self.frame_snips[vid_key]["l_fish"]
            l_idx_list = [i for i in range(l_snip[0], l_snip[1] + 1)]
            r_snip = self.frame_snips[vid_key]["r_fish"]
            r_idx_list = [i for i in range(r_snip[0], r_snip[1] + 1)]

        fill_row = [-1] * 9 + [[-1, -1]] * 4 + [-1]
        df_idx = -1

        for idx, frame in filled_df.iterrows():
            if self.frame_file is None or idx in l_idx_list + r_idx_list:
                df_idx += 1
                row = [idx]
                petri1 = frame.loc[left_cols]
                petri2 = frame[right_cols]

                # LEFT FISH
                # fish was in radius in previous frame
                if self.frame_file is None or idx in l_idx_list:
                    if left_in_radius:
                        if self._is_in_radius(petri1, "l") == 0:
                            left_in_radius = False
                            l_approach_duration = 0
                            left_fish["out_time"] += 1
                            if self._is_left_facing(petri1, "l") == 1:
                                left_fish["out_facing_left"] += 1
                            else:
                                left_fish["out_facing_right"] += 1

                        else:
                            left_fish["in_time"] += 1
                            l_approach_duration += 1
                            
                            if l_approach_duration == self.frame_rate // 2:
                                if l_side_buffer == "left":
                                    left_fish["left_approach"] += 1
                                elif l_side_buffer == "right":
                                    left_fish["right_approach"] += 1
                                l_side_buffer = None
                                
                            if self._is_left_facing(petri1, "l") == 1:
                                left_fish["in_facing_left"] += 1
                            else:
                                left_fish["in_facing_right"] += 1

                    # fish wasn't in radius in previous frame
                    else:
                        if self._is_in_radius(petri1, "l") == 1:
                            if self._is_left_facing(petri1, "l") == 1:
                                l_side_buffer = "left"
                                l_approach_duration += 1
                                left_fish["in_time"] += 1
                                left_fish["in_facing_left"] += 1
                                left_in_radius = True

                            else:
                                l_side_buffer = "right"
                                left_fish["in_time"] += 1
                                left_fish["in_facing_right"] += 1
                                left_in_radius = True

                        else:
                            left_fish["out_time"] += 1
                            l_approach_duration = 0
                            if self._is_left_facing(petri1, "l") == 1:
                                left_fish["out_facing_left"] += 1
                            else:
                                left_fish["out_facing_right"] += 1

                    row.append(int(left_in_radius))
                    row.append(left_fish["left_approach"])
                    row.append(left_fish["right_approach"])
                    row.append(round(left_fish["in_time"] / self.frame_rate, 2))
                    row.append(round(left_fish["out_time"] / self.frame_rate, 2))
                    row.append(round(left_fish["in_facing_left"] / self.frame_rate, 2))
                    row.append(round(left_fish["in_facing_right"] / self.frame_rate, 2))
                    row.append(round(left_fish["out_facing_left"] / self.frame_rate, 2))
                    row.append(
                        round(left_fish["out_facing_right"] / self.frame_rate, 2)
                    )
                    row.append(petri1.loc["lhead_l"])  # head l
                    row.append(petri1.loc["lhead_r"])  # head r
                    row.append(petri1.loc["lhead_c"])  # head c
                    row.append(petri1.loc["lrod"])  # rod
                    row.append(petri1.loc["l_filled"])

                else:
                    row += fill_row

                # RIGHT FISH
                # fish was in radius in previous frame
                if self.frame_file is None or idx in r_idx_list:
                    if right_in_radius:
                        if self._is_in_radius(petri2, "r") == 0:
                            right_in_radius = False
                            r_approach_duration = 0
                            right_fish["out_time"] += 1
                            
                            if self._is_left_facing(petri2, "r") == 1:
                                right_fish["out_facing_left"] += 1
                            else:
                                right_fish["out_facing_right"] += 1

                        else:
                            right_fish["in_time"] += 1
                            r_approach_duration += 1

                            if r_approach_duration == self.frame_rate // 2:
                                if r_side_buffer == "left":
                                    right_fish["left_approach"] += 1
                                elif r_side_buffer == "right":
                                    right_fish["right_approach"] += 1
                                r_side_buffer = None

                            if self._is_left_facing(petri2, "r") == 1:
                                right_fish["in_facing_left"] += 1
                                
                            else:
                                right_fish["in_facing_right"] += 1
                        if (
                            self._comp_helper(petri2.rhead_l, -1)
                            or self._comp_helper(petri2.rhead_r, -1)
                            or self._comp_helper(petri2.rhead_c, -1)
                            or self._comp_helper(petri2.rtail, -1)
                            or self._comp_helper(petri2.rrod, -1)
                        ):
                            r_approach_duration -= 1
                            right_fish["in_time"] -= 1
                            right_fish["in_facing_right"] -= 1

                    # fish wasn't in radius in previous frame
                    else:
                        if self._is_in_radius(petri2, "r") == 1:
                            if self._is_left_facing(petri2, "r") == 1:
                                r_side_buffer = "left"
                                r_approach_duration += 1
                                right_fish["in_time"] += 1
                                right_fish["in_facing_left"] += 1
                                right_in_radius = True

                            else:
                                r_side_buffer = "right"
                                right_fish["in_time"] += 1
                                right_fish["in_facing_right"] += 1
                                right_in_radius = True

                        else:
                            right_fish["out_time"] += 1
                            r_approach_duration = 0
                            if self._is_left_facing(petri2, "r") == 1:
                                right_fish["out_facing_left"] += 1
                            else:
                                right_fish["out_facing_right"] += 1

                        if (
                            self._comp_helper(petri2.rhead_l, -1)
                            or self._comp_helper(petri2.rhead_r, -1)
                            or self._comp_helper(petri2.rhead_c, -1)
                            or self._comp_helper(petri2.rtail, -1)
                            or self._comp_helper(petri2.rrod, -1)
                        ):
                            r_approach_duration -= 1
                            right_fish["in_time"] -= 1
                            right_fish["right_approach"] -= 1
                            right_fish["in_facing_right"] -= 1

                    row.append(int(right_in_radius))
                    row.append(right_fish["left_approach"])
                    row.append(right_fish["right_approach"])
                    row.append(round(right_fish["in_time"] / self.frame_rate, 2))
                    row.append(round(right_fish["out_time"] / self.frame_rate, 2))
                    row.append(round(right_fish["in_facing_left"] / self.frame_rate, 2))
                    row.append(
                        round(right_fish["in_facing_right"] / self.frame_rate, 2)
                    )
                    row.append(
                        round(right_fish["out_facing_left"] / self.frame_rate, 2)
                    )
                    row.append(
                        round(right_fish["out_facing_right"] / self.frame_rate, 2)
                    )
                    row.append(petri2.loc["rhead_l"])  # head l
                    row.append(petri2.loc["rhead_r"])  # head r
                    row.append(petri2.loc["rhead_c"])  # head c
                    row.append(petri2.loc["rrod"])  # rod
                    row.append(petri2.loc["r_filled"])
                    
                else:
                    row += fill_row
                    
                out_df.loc[df_idx] = row

        l_dist = 0
        r_dist = 0
        l_pos = None
        r_pos = None

        for row_idx in range(out_df.shape[0]):
            if row_idx == 0:
                l_pos = out_df.loc[row_idx, "l_center_head"]
                r_pos = out_df.loc[row_idx, "r_center_head"]

            else:
                tmp = out_df.loc[row_idx, "l_center_head"]
                l_dist += self._get_distance(l_pos[0], l_pos[1], tmp[0], tmp[1])
                l_pos = tmp.copy()
                tmp = out_df.loc[row_idx, "r_center_head"]
                r_dist += self._get_distance(r_pos[0], r_pos[1], tmp[0], tmp[1])
                r_pos = tmp.copy()

        left_fish["distance"] = l_dist
        right_fish["distance"] = r_dist
        out_df.to_csv(save_dir + "backfilled_results.csv", index=False)
        return left_fish, right_fish

    def _roll_mean(self, col):
        results = []
        assert col.shape[0] > self.window_size + 1
        win_range = (self.window_size // 2)
        for i in range(col.shape[0]):
            # start
            if i == 0:
                x_coord = np.mean(col[1:i + 1 + win_range, 0])
                y_coord = np.mean(col[1:i + 1 + win_range, 1])
                results.append([x_coord, y_coord])

            # not start but still too close to start for full backwards sliding window
            elif i < win_range:
                x_coord = (np.mean(col[:i, 0]) + np.mean(col[i + 1:i + 1 + win_range, 0])) / 2
                y_coord = (np.mean(col[:i, 1]) + np.mean(col[i + 1:i + 1 + win_range, 1])) / 2
                results.append([x_coord, y_coord])

            # end
            elif i == col.shape[0] - 1:
                x_coord = np.mean(col[i - win_range:, 0])
                y_coord = np.mean(col[i - win_range:, 1])
                results.append([x_coord, y_coord])

            # not end but too close for full forward sliding window
            # need to add +1 here because right side of list slice is exclusive index
            elif i + 1 + win_range > col.shape[0] - 1:
                x_coord = (np.mean(col[i - win_range:i, 0]) + np.mean(col[i + 1:, 0])) / 2
                y_coord = (np.mean(col[i - win_range:i, 1]) + np.mean(col[i + 1:, 1])) / 2
                results.append([x_coord, y_coord])

            else:
                x_coord = (np.mean(col[i - win_range:i, 0]) + np.mean(col[i + 1:i + 1 + win_range, 0])) / 2
                y_coord = (np.mean(col[i - win_range:i, 1]) + np.mean(col[i + 1:i + 1 + win_range, 1])) / 2
                results.append([x_coord, y_coord])

        return results

    @staticmethod
    def _form_helper(val):
        val = re.sub(r'[\s\[\]]', '', val)
        return val

    def _form_str_cell(self, val):
        assert type(val) == str

        if len(val.split(',')) > 1:
            tmp = val.split(',')
            return np.array([float(self._form_helper(tmp[0])), float(self._form_helper(tmp[1]))])

        else:
            form_val = []
            tmp = val.split(' ')
            for v in tmp:
                try:
                    form_val.append(float(self._form_helper(v)))
                except ValueError:
                    pass
            assert len(form_val) == 2

            return np.array(form_val)

    def _convert_source_df(self, in_df):
        out_cols = {"frame_idx": 'frame_idx',
                    "l_filled": 'l_coords_filled', 
                    "lhead_l": 'l_left_head', 
                    "lhead_r": 'l_right_head', 
                    "lhead_c": 'l_center_head', 
                    "ltail": None, 
                    "lrod": 'l_rod', 
                    "r_filled": 'r_coords_filled', 
                    "rhead_l": 'r_left_head', 
                    "rhead_r": 'r_right_head',
                    "rhead_c": 'r_center_head', 
                    "rtail": None, 
                    "rrod": 'r_rod'}
        
        # initialize dictionary holding values for out df
        out_dict = {}
        for col in out_cols.keys():
            # tail coordinates are irrelevant
            if 'tail' in col:
                out_dict[col] = [[0,0]] * in_df.shape[0]
            
            # these columns can be copied directly
            elif col in ['frame_idx', 'l_filled', 'r_filled']:
                out_dict[col] = in_df[out_cols[col]].tolist()
                
            # everything else needs to be filled in
            else:
                # might need to work with legacy pandas version, can't call .to_numpy()
                col_array = np.array([self._form_str_cell(item) for item in in_df[out_cols[col]].tolist()])
                out_dict[col] = self._roll_mean(col_array)

        return pd.DataFrame.from_dict(out_dict)

    def _add_to_summary_file(self, left_dict, right_dict, fname):
        lrow = {
            "Subject": f"{fname}_left",
            "Duration in radius - left-facing": left_dict["in_facing_left"]
            / self.frame_rate,
            "Duration in radius - right-facing": left_dict["in_facing_right"]
            / self.frame_rate,
            "Duration outside radius - left-facing": left_dict["out_facing_left"]
            / self.frame_rate,
            "Duration outside radius - right-facing": left_dict["out_facing_right"]
            / self.frame_rate,
            "Left approaches": left_dict["left_approach"],
            "Right approaches": left_dict["right_approach"],
            "Distance traveled (mm)": left_dict["distance"],
        }

        rrow = {
            "Subject": f"{fname}_right",
            "Duration in radius - left-facing": right_dict["in_facing_left"]
            / self.frame_rate,
            "Duration in radius - right-facing": right_dict["in_facing_right"]
            / self.frame_rate,
            "Duration outside radius - left-facing": right_dict["out_facing_left"]
            / self.frame_rate,
            "Duration outside radius - right-facing": right_dict["out_facing_right"]
            / self.frame_rate,
            "Left approaches": right_dict["left_approach"],
            "Right approaches": right_dict["right_approach"],
            "Distance traveled (mm)": right_dict["distance"],
        }

        self.sum_file = self.sum_file.append([lrow, rrow], ignore_index=True)

    def analyze_video(self, result_dir):
        vid_name = result_dir.split('/')[-2]
        if self.frame_file is not None and vid_name not in self.frame_snips.keys():
            print(f'Skipped video: {vid_name}')

        else:
            src_df = pd.read_csv(result_dir + 'approach_results.csv')
            conv_df = self._convert_source_df(src_df)
            left_results, right_results = self._get_metrics(conv_df, result_dir)

            self._add_to_summary_file(
                left_results,
                right_results,
                vid_name,
            )

            print(f'Done: {result_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill& reanalyze cavefish results.")

    parser.add_argument("approach_radius",
                        type=int,
                        help="Radius around rod that will be considered as an approach if crossed (in mm).")

    parser.add_argument("--result_folder",
                        "-r",
                        type=str,
                        required=True,
                        default=None,
                        help="Folder containing results from previously analyzed videos.")

    parser.add_argument("--frame_rate",
                        "-f",
                        type=int,
                        required=False,
                        default=10,
                        help="Frame rate at which videos were recorded. Default is 10.")

    parser.add_argument("--frame_file",
                        "-x",
                        type=str,
                        required=False,
                        default=None,
                        help="File containing subsections of videos to analyze.")

    parser.add_argument("--window_size",
                        "-w",
                        type=int,
                        required=False,
                        default=10,
                        help="Rolling window size for filling coordinates. Default is 10.")

    parser.add_argument("--ratio_pixmm",
                        "-r",
                        type=float,
                        required=True,
                        default=5.287,
                        help="Pixel to millimeter ratio (px/mm).")

    parser.add_argument("--dims_x",
                        "-d",
                        type=int,
                        required=True,
                        default=1280,
                        help="Original video x-dimensionality in pixels.")

    usr_args = vars(parser.parse_args())

    filler = BackFiller(**usr_args)

    for vid_path in [f for f in os.listdir(usr_args["result_folder"])]:
        if usr_args["result_folder"].endswith("/") or usr_args["result_folder"].endswith("\\"):
            if os.path.isdir(usr_args["result_folder"][:-1] + "/" + vid_path + "/"):
                filler.analyze_video(usr_args["result_folder"][:-1] + "/" + vid_path + "/")

        else:
            if os.path.isdir(usr_args["result_folder"][:-1] + "/" + vid_path + "/"):
                filler.analyze_video(usr_args["result_folder"] + "/" + vid_path + "/")

    if usr_args["result_folder"].endswith("\\") or usr_args["result_folder"].endswith("/"):
        filler.sum_file.to_csv(usr_args["result_folder"] + "roll_avg_summary_results.csv", index=False)

    else:
        filler.sum_file.to_csv(usr_args["result_folder"] + "/roll_avg_summary_results.csv", index=False)

