# How To

## Capture Video

I used the manual exposure mode of [ProCam 8](https://apps.apple.com/in/app/procam-8/id730712409) on iPhone to capture 1080p 30fps video of the exgizmo. The exposure settings I used were a shutter speed of 1/810 of a second and ISO 400. These setting correctly expose the exgizmo's LCD for the image processing used by this software, I recommend using them to capture video. If video has been captured in an automatic mode, the `--enhance-contrast` option may allow the image to be processed correctly.

Video should be free of artifacts like scratches, reflections, etc. The software must have an unobstructed view of the blue bounding box in each frame. Here is a sample image I have captured:

![A sample image of the exgizmo](/assets/sample.jpg)

## Install
```
# create virtualenv
virtualenv .venv
# activate virtualenv
. .venv/bin/activate
# install requirements
pip3 install -r requirements.txt
```

## Prep Data

Place your data file in a directory, such as `/data` so the extracted images stay in one spot. Run ffmpeg to extract images of video frames. I recommend using 15fps. I tried at 30fps but saw some errant data read out, 15 worked better.

```
cd data/
ffmpeg -i <video file> -vf fps=15 -qscale:v 2 video-%06d.jpg
```

If there are frames that aren't of the exgizmo at the beginning or end, delete those jpegs.

## Process a Test Frame

Process a test frame of the video before running OCR on the whole video to see if it will work without waiting for the whole video to process.

```
python3 test_frame.py data/video-000001.jpg
```

Here is the output for sample.jpg
```
  PM2.5: 1978
  PM1.0: 204
   PM10: 5193
   >0.3: 65535
   >0.5: 25190
   >1.0: 21005
   >2.5: 10055
   >5.0: 3567
    >10: 2264
```

In addition, the following png artifacts from image processing are created:
* contrast.png: if `--enhance-contrast` was set, the contrast enhanced image is output
* blue_thresh.png: the threshold image for finding the blue border around the PM data
* warpim.png: perspective corrected image of just the blue box contents
* y_thresh.png: yellow thresholded image for finding the PM text
* ocrboxes.png: warpim.png overlaid with green boxes denoting the positions of the values to read
* pm2_5bbox.png ... gt10bbox.png: yellow-thresholded, gaussian-blurred, inverted, cropped image of the value -- this is fed to OCR

If there are problems with the OCR, these images can be examined to understand the issue.

## Get Results

```
# enter virtualenv
. .venv/bin/activate
# run ocr program
python3 read_exgizmo.py data/
```

This will generate a csv file, `data/output.csv` containing the PM1.0, PM2.5, and PM10 results. To extract the right-hand column values (>0.3, >0.5, ...) use the `--details` option.

The program will run much faster if CUDA is enabled. On a system with an Nvidia 3080 ti, 32 seconds of video processed in 44 seconds without `--details` and 1m25s with `--details`. On the same system with the GPU disabled, the video processed in 4m6s without `--details` and 8m53s with `--details`.

# Details

An issue that had to be solved was that an extracted frame may catch a value during the LCD's refresh cycle. These values are garbage. They present in one of two ways. One, part of a value has changed but part has not. OCR will usually read a number in this case but it will be trash. Two, a completely blank value. In this case OCR will read nothing.

To deal with the possibility of mid-refresh values the program keeps track of the state of each value and only records a new value when two consecutive frames produce the same result. This means data is only recorded for a value when there is a change so for a particular time, not all PM categories have a value. e.g.
```
	pm2.5	pm1.0	pm10
time			
0.000000	2042.0	187.0	6029.0
0.266667	1978.0	NaN	NaN
0.333333	NaN	204.0	5193.0
1.066667	1888.0	NaN	NaN
1.133333	NaN	229.0	4370.0
...	...	...	...
30.133333	61.0	45.0	63.0
31.066667	54.0	40.0	56.0
32.000000	48.0	36.0	50.0
32.866667	40.0	NaN	NaN
32.933333	NaN	29.0	43.0
```
To smooth out the data, the blank values are filled forward from the previous value. This is the result:
```
	pm2.5	pm1.0	pm10
time			
0.000000	2042.0	187.0	6029.0
0.266667	1978.0	187.0	6029.0
0.333333	1978.0	204.0	5193.0
1.066667	1888.0	204.0	5193.0
1.133333	1888.0	229.0	4370.0
...	...	...	...
30.133333	61.0	45.0	63.0
31.066667	54.0	40.0	56.0
32.000000	48.0	36.0	50.0
32.866667	40.0	36.0	50.0
32.933333	40.0	29.0	43.0
```

To disable the fill forward, use the `--no-fill` option.

In addition to the issue with single values being caught mid-refresh, 15 fps is fast enough that the values don't appear to refresh all at once. Rather, they refresh one at a time in the order PM2.5, PM1.0, PM10, >0.3, >0.5, and so on. The program captures the frame number at which each individual change occurs which means there are many slightly different time values in the resulting data. It is probably possible to post-process the data to detect a refresh and group all the values to a particular time value. That was not attempted here.
