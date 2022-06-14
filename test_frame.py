import read_exgizmo
import argparse

def run(args):
    framedata = read_exgizmo.getFrameData(args.IMAGE, contrast=args.enhance_contrast, details=True, debug=True)
    print('  PM2.5: {}'.format(framedata.pm2_5))
    print('  PM1.0: {}'.format(framedata.pm1_0))
    print('   PM10: {}'.format(framedata.pm10))
    print('   >0.3: {}'.format(framedata.gt0_3))
    print('   >0.5: {}'.format(framedata.gt0_5))
    print('   >1.0: {}'.format(framedata.gt1_0))
    print('   >2.5: {}'.format(framedata.gt2_5))
    print('   >5.0: {}'.format(framedata.gt5_0))
    print('    >10: {}'.format(framedata.gt10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('IMAGE', type=str, help='path to sample image to test')
    parser.add_argument('--enhance-contrast', action='store_true')

    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
