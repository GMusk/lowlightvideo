import numpy as np
import cv2
import os
from MovingAverage import MovingAverage
from FrameEditor import FrameEditor
from Stable import Stabiliser
from tqdm import tqdm
from plots import *


class LowLightEnhancer:
    """A class for enhancing low light video files

    The LowLightEnhancer class is used to enhance low light video using a logarithmic expansion of frames averaged over some buffer size

    """
    def __init__(self, args):

        input_path = args['input']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        # get capture object
        self.cap = self.getVideo(input_path)
        # get total number of frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # get buffer size
        self.buffer_size = args['buffer']
        # handle size parameter
        self.handle_size(args['size'])
        # get name of file
        self.filename = self.get_filename(input_path)
        # get output
        self.output = args['output']
        # create dictionary of video writers
        self.video_writers = {}
        # determine options for program
        self.handle_option(args['option'])
        # initiate average and edit class
        self.stabiliser = Stabiliser(self.total_frames, self.size)
        self.ma = MovingAverage(self.size, False, args['buffer'])
        self.fe = FrameEditor()
        # creates background model class using mixture of gradients
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def release_video(self):
        for writer in self.video_writers.values():
            writer.release()

    def handle_size(self, size):
        if size != None:
            self.size = tuple(args.size)
        else:
            # get caps original size
            self.size = (int(self.cap.get(3)), int(self.cap.get(4)))

    def get_filename(self, path):
        path_split = path.split('/')
        filename = path_split[-1].split('.')[0]
        return filename

    def handle_option(self, opt):
        for char in opt:
            if char == 's':
                self.video_writers["stable"] = self.create_vwriter("stable")
            elif char == 'e':
                self.video_writers["expand"] = self.create_vwriter("expand")
            elif char == 'a':
                self.video_writers["average"] = self.create_vwriter("average")
            elif char == 'm':
                self.video_writers["motion"] = self.create_vwriter("motion")
            elif char == 'c':
                self.video_writers["closure"] = self.create_vwriter("closure")

    def create_vwriter(self, out_type):
        # set codec for written video
        filename = self.output + self.filename + '-' + out_type + str(self.buffer_size) + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(filename, fourcc, fps, self.size)
        return video_writer

    def getVideo(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("No Video Found")
            return None
        else:
            return cap

    def enhance(self):
        """Read video, perform enhancement, & write enhanced video to file
        """

        # first pass
        print("first pass")
        for i in tqdm(range(self.total_frames)):

            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # check capture
            if not ret:
                print("video finished")
                break

            self.stabiliser.get_transform(frame, i)

        # stabilise
        traj, smooth = self.stabiliser.get_trajectory()

        # plot trajectorys
        plot_trajectory(traj, smooth)

        # Reset stream to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # default mask value
        mask_fg = np.zeros((*self.size, 3))

        print("second pass")
        for i in tqdm(range(self.total_frames)):

            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # check capture
            if not ret:
                print("video finished")
                break

            # stablise video
            if "stable" in self.video_writers:
                stable = self.stabiliser.get_stable_frame(frame, i)
                frame = stable
                self.video_writers["stable"].write(stable)

            # get foreground mask
            if "motion" in self.video_writers:
                fg = self.backSub.apply(frame)
                mask_fg = fg
                mask_fg_3 = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                self.video_writers["motion"].write(mask_fg_3)

            if "closure" in self.video_writers:
                # threshold and inverse
                _, thresh = cv2.threshold(mask_fg, 127, 255, cv2.THRESH_BINARY_INV)
                close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                close_3 = cv2.cvtColor(close, cv2.COLOR_GRAY2BGR)
                self.video_writers["closure"].write(close_3)
                mask_fg = close

            # Call image operations here passes uint8 gets uint8
            if "expand" in self.video_writers:
                edit_frame = self.fe.doOperation(frame)
                self.video_writers["expand"].write(edit_frame)
                frame = edit_frame

            # store frame in moving average
            if "average" in self.video_writers:
                av_frame = self.ma.add(frame, mask_fg)
                if av_frame is not None:
                    av_frame = np.uint8(av_frame)
                    if i % (self.buffer_size - 1) == 0:
                        print("saving image")
                        cv2.imwrite(self.output + "picture" + str(i) + ".png", av_frame)
                    self.video_writers["average"].write(av_frame)

            # key to break playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        self.release_video()
        self.cap.release()
        cv2.destroyAllWindows()

