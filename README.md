# phd

## main.py
### inputs
  pathsname = "aug" ## aug or no_aug
  hhhh = "h1"
  names = ["test", "valid", "train"]
  folders = ["h1_asis", "h2_warp", "h3_ahe", "h7_combined"] ## USED FOR no_aug
  sizes = 32
  color_mode = "LAB"  # Could also be "LAB" or "H"


## helpers_opencv.py
Its a support file for main.py, it handles the actual processing of individual annotation files.



## ml.py
Its trains the ML models from the data above.

