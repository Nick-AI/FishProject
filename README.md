# FishProject
Project analyzing videos of cave fish approaching vibrating rod in a petri dish for handedness.

## To analyze videos
- After installing all dependencies, change the DeepLabCut/FishApproach-Nick-2019-05-07/config.yaml file as follows:
    - Line 7: change the project_path to the absolute path to FishProject/DeepLabCutModel/FishApproach-Nick-2019-05-07/
- Then you can run the model:
    - call `python RunModel.py [radius] -f [frame rate] -s [destination dir] -v [video dir]`
    - replace [radius] with the radius around the rod in mm that you want to count as an approach
    - replace [destination dir] with the location where results will be stored (you don't have to specify this argument,
    it will default to your current working directory, however, it is recommended, especially for batches of videos)
    - replace [frame rate] with the frame rate at which your video is recorded (default is 10)
    - you can call `python RunModel.py -h` for help
- To analyze batches of videos, you need to have all of them in a folder (not containing anything else) and specify that
folder in the -v flag
- Use absolute pathnames with no trailing slash (e.g. /home/user/folder)
- If you don't specify the -v flag you will be prompted to enter the path to a tif video that you want to analyze
    - To quit, press Ctrl+C

## Dependencies
- Python3.6
- deeplabcut
- wxPython
  - With pip, on Ubuntu use: 
  `$pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl`
- tensorflow
- numpy==1.16.3
- pandas==0.21.0
