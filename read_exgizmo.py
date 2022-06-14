import cv2 as cv
import numpy as np
import easyocr
import argparse
import os
import pandas as pd

reader = easyocr.Reader(['en'])

class FrameData:
    def __init__(self, pm2_5, pm1_0, pm10, gt0_3, gt0_5, gt1_0, gt2_5, gt5_0, gt10):
        self.pm2_5 = pm2_5
        self.pm1_0 = pm1_0
        self.pm10  = pm10
        self.gt0_3 = gt0_3
        self.gt0_5 = gt0_5
        self.gt1_0 = gt1_0
        self.gt2_5 = gt2_5
        self.gt5_0 = gt5_0
        self.gt10  = gt10
    
    @staticmethod
    def empty() -> 'FrameData':
        return FrameData(None, None, None, None, None, None, None, None, None)

def getFrameData(image_path: str, contrast: bool = False, details: bool = False, debug: bool = False) -> FrameData:
    im = cv.imread(image_path)

    # if contrast was set, increase contrast of the image
    if contrast:
        xp = [0, 64, 128, 192, 255]
        fp = [0, 0, 100, 255, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        imcontrast = cv.LUT(im, table)
        cv.imwrite("contrast.png", imcontrast)
    else:
        imcontrast = im.copy()

    # convert the image to hsv to aid in thresholding on the blue color of the border
    imhsv = cv.cvtColor(imcontrast, cv.COLOR_BGR2HSV)
    low_hsv = (88, 0, 70)
    high_hsv = (255, 255, 255)
    thresh = cv.inRange(imhsv, low_hsv, high_hsv)
    if debug:
        cv.imwrite("blue_thresh.png", thresh)

    # get contours associated with blue box
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # select the largest contour area, this is the largest blue rectangle
    contour = None
    max_area = 0
    for idx, area in enumerate([cv.contourArea(x) for x in contours]):
        if area > max_area:
            contour = contours[idx]
            max_area = area

    perimeter = cv.arcLength(contour, True)
    # get 4 corners of the largest blue rectangle
    poly = cv.approxPolyDP(contour, 0.01 * perimeter, True)
    # define the size of the straight on blue rectangle
    pts2 = np.float32([(1456, 0), (0, 0), (0, 1080), (1456, 1080)])

    # warp image to square up face of the exgizmo
    M = cv.getPerspectiveTransform(np.float32(poly), pts2)
    warpim = cv.warpPerspective(imcontrast, M, (1456,1080))
    if debug:
        cv.imwrite("warpim.png", warpim)

    # threshold on the yellow so we can select the text
    yellowim = cv.inRange(warpim, (0, 0, 70), (255,255,255))
    if debug:
        cv.imwrite("y_thresh.png", yellowim)
    # gaussian blur it
    imblur = cv.GaussianBlur(yellowim, (17,17), 0)
    # and invert it so it's black text on white bg
    impmdata = cv.bitwise_not(imblur)

    # find bboxes of values to extract relative to bbox from perspective xform
    ocrboxim = warpim.copy()
    max_x = 491
    min_x = 62
    max_y = 513
    min_y = 303
    pm2_5bbox = np.int32([(max_x, min_y), (min_x,min_y), (min_x,max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [pm2_5bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 69 + 427
    min_x = 69
    max_y = 798
    min_y = 798 - 200
    pm1_0bbox = np.int32([(max_x, min_y), (min_x,min_y), (min_x,max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [pm1_0bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 71 + 412
    min_x = 71
    max_y = 1063
    min_y = 1063 - 183
    pm10bbox = np.int32([(max_x, min_y), (min_x,min_y), (min_x,max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [pm10bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 1006 + 348
    min_x = 1006
    max_y = 363
    min_y = 363 - 149
    gt0_3bbox = np.int32([(max_x, min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [gt0_3bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 1006 + 348
    min_x = 1006
    max_y = 504
    min_y = 504 - 141
    gt0_5bbox = np.int32([(max_x, min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [gt0_5bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 1006 + 348
    min_x = 1006
    max_y = 651
    min_y = 651 - 148
    gt1_0bbox = np.int32([(max_x, min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [gt1_0bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 1006 + 348
    min_x = 1006
    max_y = 789
    min_y = 789 - 139
    gt2_5bbox = np.int32([(max_x, min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [gt2_5bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 1006 + 348
    min_x = 1006
    max_y = 937
    min_y = 937 - 149
    gt5_0bbox = np.int32([(max_x, min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [gt5_0bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    max_x = 1006 + 348
    min_x = 1006
    max_y = 1070
    min_y = 1070 - 134
    gt10bbox = np.int32([(max_x, min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y)])
    cv.polylines(ocrboxim, [gt10bbox], True, (0, 255, 0), 1, cv.LINE_AA)

    if debug:
        cv.imwrite("ocrboxes.png", ocrboxim)

    pm2_5im = impmdata[pm2_5bbox[0][1]:pm2_5bbox[2][1], pm2_5bbox[1][0]:pm2_5bbox[0][0]]
    if debug:
        cv.imwrite("pm2_5bbox.png", pm2_5im)

    try:
        pm2_5 = reader.readtext(pm2_5im)[0][1]
    except IndexError:
        pm2_5 = None

    pm1_0im = impmdata[pm1_0bbox[0][1]:pm1_0bbox[2][1], pm1_0bbox[1][0]:pm1_0bbox[0][0]]
    if debug:
        cv.imwrite("pm1_0bbox.png", pm1_0im)

    try:
        pm1_0 = reader.readtext(pm1_0im)[0][1]
    except IndexError:
        pm1_0 = None

    pm10im = impmdata[pm10bbox[0][1]:pm10bbox[2][1], pm10bbox[1][0]:pm10bbox[0][0]]
    if debug:
        cv.imwrite("pm10bbox.png", pm10im)

    try:
        pm10 = reader.readtext(pm10im)[0][1]
    except IndexError:
        pm10 = None
    
    if details:
        gt0_3im = impmdata[gt0_3bbox[0][1]:gt0_3bbox[2][1], gt0_3bbox[1][0]:gt0_3bbox[0][0]]
        if debug:
            cv.imwrite("gt0_3bbox.png", gt0_3im)

        try:
            gt0_3 = reader.readtext(gt0_3im)[0][1]
        except IndexError:
            gt0_3 = None

        gt0_5im = impmdata[gt0_5bbox[0][1]:gt0_5bbox[2][1], gt0_5bbox[1][0]:gt0_5bbox[0][0]]
        if debug:
            cv.imwrite("gt0_5bbox.png", gt0_5im)

        try:
            gt0_5 = reader.readtext(gt0_5im)[0][1]
        except IndexError:
            gt0_5 = None

        gt1_0im = impmdata[gt1_0bbox[0][1]:gt1_0bbox[2][1], gt1_0bbox[1][0]:gt1_0bbox[0][0]]
        if debug:
            cv.imwrite("gt1_0bbox.png", gt1_0im)

        try:
            gt1_0 = reader.readtext(gt1_0im)[0][1]
        except IndexError:
            gt1_0 = None

        gt2_5im = impmdata[gt2_5bbox[0][1]:gt2_5bbox[2][1], gt2_5bbox[1][0]:gt2_5bbox[0][0]]
        if debug:
            cv.imwrite("gt2_5bbox.png", gt2_5im)

        try:
            gt2_5 = reader.readtext(gt2_5im)[0][1]
        except IndexError:
            gt2_5 = None

        gt5_0im = impmdata[gt5_0bbox[0][1]:gt5_0bbox[2][1], gt5_0bbox[1][0]:gt5_0bbox[0][0]]
        if debug:
            cv.imwrite("gt5_0bbox.png", gt5_0im)

        try:
            gt5_0 = reader.readtext(gt5_0im)[0][1]
        except IndexError:
            gt5_0 = None

        gt10im = impmdata[gt10bbox[0][1]:gt10bbox[2][1], gt10bbox[1][0]:gt10bbox[0][0]]
        if debug:
            cv.imwrite("gt10bbox.png", gt10im)

        try:
            gt10 = reader.readtext(gt10im)[0][1]
        except IndexError:
            gt10 = None
    else:
        gt0_3 = None
        gt0_5 = None
        gt1_0 = None
        gt2_5 = None
        gt5_0 = None
        gt10  = None

    f = FrameData(pm2_5, pm1_0, pm10, gt0_3, gt0_5, gt1_0, gt2_5, gt5_0, gt10)

    return f

def read_frames(dir: str, contrast: bool = False, details: bool = False) -> pd.DataFrame:
    contents = sorted(filter(lambda x: x.endswith('.jpg'), os.listdir(dir)))

    class ValueCategory:
        def __init__(self, change_action):
            self.value = None
            self.state = None
            self.change_action = change_action

        def next_value(self, value: int):
            if self.value != value:
                self.state = 'changed'
                self.value = value
            else:
                if self.state == 'changed':
                    self.change_action(value)
                self.state = 'stable'

    framenum = 0
    pm2_5_list = []
    pm1_0_list = []
    pm10_list  = []
    gt0_3_list = []
    gt0_5_list = []
    gt1_0_list = []
    gt2_5_list = []
    gt5_0_list = []
    gt10_list  = []

    state = {
        'pm2_5': ValueCategory(lambda x: pm2_5_list.append((framenum - 1, x))),
        'pm1_0': ValueCategory(lambda x: pm1_0_list.append((framenum - 1, x))),
        'pm10':  ValueCategory(lambda x: pm10_list.append(( framenum - 1, x))),
        'gt0_3': ValueCategory(lambda x: gt0_3_list.append((framenum - 1, x))),
        'gt0_5': ValueCategory(lambda x: gt0_5_list.append((framenum - 1, x))),
        'gt1_0': ValueCategory(lambda x: gt1_0_list.append((framenum - 1, x))),
        'gt2_5': ValueCategory(lambda x: gt2_5_list.append((framenum - 1, x))),
        'gt5_0': ValueCategory(lambda x: gt5_0_list.append((framenum - 1, x))),
        'gt10':  ValueCategory(lambda x: gt10_list.append(( framenum - 1, x)))
    }

    print('processing framedata', end='...')
    length = len(contents)
    mod_pct_complete_last, mod_pct_complete = 0, 0
    for framenum, path in enumerate(contents):
        framedata = getFrameData(os.path.join(dir, path), contrast, details)
        # transition value to 'changed' & store value for comparison with next frame
        # on next frame see if value is the same as last, if so store it and set state to 'stable'
        for k, valuecategory in state.items():
            value = getattr(framedata, k)
            valuecategory.next_value(value)
        mod_pct_complete = framenum * 100.0 / length % 10
        if mod_pct_complete >= 0 and mod_pct_complete < mod_pct_complete_last:
            print('{}%'.format(int(framenum * 100 / length)), end='...', flush=True)
        mod_pct_complete_last = mod_pct_complete
    print('100%')

    unzip_pm2_5 = [x for x in zip(*pm2_5_list)]
    unzip_pm1_0 = [x for x in zip(*pm1_0_list)]
    unzip_pm10  = [x for x in zip(*pm10_list)]
    if details:
        unzip_gt0_3 = [x for x in zip(*gt0_3_list)]
        unzip_gt0_5 = [x for x in zip(*gt0_5_list)]
        unzip_gt1_0 = [x for x in zip(*gt1_0_list)]
        unzip_gt2_5 = [x for x in zip(*gt2_5_list)]
        unzip_gt5_0 = [x for x in zip(*gt5_0_list)]
        unzip_gt10  = [x for x in zip(*gt10_list)]
    s_pm2_5 = pd.Series(unzip_pm2_5[1], index=unzip_pm2_5[0])
    s_pm1_0 = pd.Series(unzip_pm1_0[1], index=unzip_pm1_0[0])
    s_pm10  = pd.Series(unzip_pm10[1], index=unzip_pm10[0])
    if details:
        s_gt0_3 = pd.Series(unzip_gt0_3[1], index=unzip_gt0_3[0])
        s_gt0_5 = pd.Series(unzip_gt0_5[1], index=unzip_gt0_5[0])
        s_gt1_0 = pd.Series(unzip_gt1_0[1], index=unzip_gt1_0[0])
        s_gt2_5 = pd.Series(unzip_gt2_5[1], index=unzip_gt2_5[0])
        s_gt5_0 = pd.Series(unzip_gt5_0[1], index=unzip_gt5_0[0])
        s_gt10  = pd.Series(unzip_gt10[1],  index=unzip_gt10[0])

    if not details:
        df = pd.DataFrame({'pm2.5': s_pm2_5, 'pm1.0': s_pm1_0, 'pm10': s_pm10})
    else:
        df = pd.DataFrame({'pm2.5': s_pm2_5, 'pm1.0': s_pm1_0, 'pm10': s_pm10, '>0.3': s_gt0_3,
            '>0.5': s_gt0_5, '>1.0': s_gt1_0, '>2.5': s_gt2_5, '>5.0': s_gt5_0, '>10': s_gt10})

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default='.', help='directory containing frames of video')
    parser.add_argument('--enhance-contrast', action='store_true')
    parser.add_argument('--details', action='store_true', help='read the rhs column with the > values in it')
    parser.add_argument('--fps', type=int, default=15, help='fps of source images, the frame numbers are divided by this value to generate the time value written to the csv')
    parser.add_argument('--no-fill', action='store_true', help='do not fill in data gaps')
    parser.add_argument('--output', type=str, help='name of csv file to output')

    args = parser.parse_args()

    if args.output is None:
        output = os.path.join(args.dir, 'output.csv')
    else:
        output = args.output

    df = read_frames(args.dir, args.enhance_contrast, args.details)
    if not args.no_fill:
        # use pandas to fill in all the gaps in each column
        df.fillna(method='ffill', inplace=True)
    df['time'] = df.index / args.fps

    df.to_csv(output)

if __name__ == '__main__':
    main()
