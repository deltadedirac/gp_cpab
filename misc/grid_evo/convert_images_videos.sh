#!/bin/bash

#ls linear/*.png | sort -V | xargs -I {} echo "file '{}'" > list.txt
ffmpeg -i "linear/%d.png" outputlinear.mp4
#ffmpeg -r 1/5 -f concat -i list.txt -vf scale=540:-2 -t 15 out.mp4
