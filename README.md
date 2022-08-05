# FishProject
Project analyzing videos of cave fish approaching vibrating rod in a petri dish for handedness.

## To analyze videos
- After installing all dependencies, change the DeepLabCut/FishApproach-Nick-2019-05-07/config.yaml file as follows:
    - Line 7: change the project_path to the absolute path to FishProject/DeepLabCutModel/FishApproach-Nick-2019-05-07/
- Then you can run the model:
    - call `python RunModel.py [radius] -s [dir]`
    - replace [radius] with the radius around the rod in mm that you want to count as an approach
    - replace dir with the location where results will be stored (you don't have to specify this argument, it will default to your current working directory
    - you can call `python RunModel.py -h` for help
- It will prompt you to enter the path to a tif video that you want to analyze
    - For now only one video at a time is supported
    - Results cannot yet be stored to a file but will just be printed out
- To quit, press Ctrl+C

## Dependencies
- Python3.6
- deeplabcut
- wxPython
- tensorflow
- numpy==1.16.3

[![DOI](https://zenodo.org/badge/186560317.svg)](https://zenodo.org/badge/latestdoi/186560317)
