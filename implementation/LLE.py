import numpy as np
import cv2
import os
from MovingAverage import MovingAverage
from FrameEditor import FrameEditor
from Stabiliser import Stabiliser
from Plotter import Plotter
from tqdm import tqdm


class LowLightEnhancer:
    """A class for enhancing low light video files

    The LowLightEnhancer class is used to enhance low light video using a logarithmic expansion of frames averaged over some buffer size

    """
    def __init__(self, args):

        input_path = args['input']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        # get capture object
        self.cap = self.get_video(input_path)
        # get total number of frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # get buffer size
        self.buffer_size = args['buffer']
        # handle size parameter
        self.handle_size(args['size'])
        # get name of file
        filename = self.get_filename(input_path)
        # get output
        output = args['output']
        # get final dir
        self.output_dir = output + filename + '/'
        # create dictionary of video writers
        self.input_videos = {}
        self.video_writers = {}
        # determine options for program
        self.handle_option(args['option'], args['read'])
        # initiate average and edit class
        self.stabiliser = Stabiliser(self.total_frames, self.size)
        self.ma = MovingAverage(self.size, False, False, args['buffer'])
        self.fe = FrameEditor()
        self.plt = Plotter(self.output_dir)
        # creates background model class using mixture of gradients
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def release_video(self):
        for writer in self.video_writers.values():
            writer.release()

    def handle_size(self, size):
        if size != None:
            self.size = tuple(args.size[1], args.size[0])
        else:
            # get caps original size
            self.size = (int(self.cap.get(4)), int(self.cap.get(3)))

    def get_filename(self, path):
        path_split = path.split('/')
        filename = path_split[-1].split('.')[0]
        return filename

    def handle_option(self, opt, read):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        for char in opt:
            if char == 's':
                cur_filename = self.output_dir + "stable.mp4"
                if read:
                    cap = self.get_video(cur_filename)
                    if cap:
                        print("input found")
                        self.input_videos["stable"] = cap
                        continue
                self.video_writers["stable"] = self.create_vwriter(cur_filename)
            elif char == 'e':
                cur_filename = self.output_dir + "expand.mp4"
                if read:
                    cap = self.get_video(cur_filename)
                    if cap:
                        print("input found")
                        self.input_videos["expand"] = cap
                        continue
                self.video_writers["expand"] = self.create_vwriter(cur_filename)
            elif char == 'm':
                cur_filename = self.output_dir + "motion.mp4"
                if read:
                    cap = self.get_video(cur_filename)
                    if cap:
                        print("input found")
                        self.input_videos["motion"] = cap
                        continue
                self.video_writers["motion"] = self.create_vwriter(cur_filename)
            elif char == 'c':
                cur_filename = self.output_dir + "closure.mp4"
                if read:
                    cap = self.get_video(cur_filename)
                    if cap:
                        print("input found")
                        self.input_videos["closure"] = cap
                        continue
                self.video_writers["closure"] = self.create_vwriter(cur_filename)
            elif char == 'a':
                cur_filename = self.output_dir + "average.mp4"
                self.video_writers["average"] = self.create_vwriter(cur_filename)
                cur_filename = self.output_dir + "contribution.mp4"
                self.video_writers["contribution"] = self.create_vwriter(cur_filename)

    def create_vwriter(self, filename):
        # set codec for written video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(filename, fourcc, fps, self.size[::-1])
        return video_writer

    def get_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("No Video Found")
            return None
        else:
            return cap

    def enough_frames(self):
        return False

    def enhance(self):
        """Read video, perform enhancement, & write enhanced video to file
        """

        # first pass
        print("first pass")
        if "stable" not in self.input_videos and "stable" in self.video_writers:
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
        self.plt.plot_trajectory(traj, smooth)

        # Reset stream to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # default mask value
        mask_fg = np.ones((self.size), dtype=np.uint8) * 255

        finished = False
        i = 0

        print("second pass")
        for i in tqdm(range(self.total_frames)):

            # Capture frame-by-frame
            ret, frame = self.cap.read()
            initial = frame


            # check capture
            if not ret:
                print("video finished")
                break

            # stablise video
            if "stable" in self.video_writers:
                stable = self.stabiliser.get_stable_frame(frame, i)
                frame = stable
                self.video_writers["stable"].write(stable)
            # check if reading from file
            elif "stable" in self.input_videos:
                ret, frame = self.input_videos["stable"].read()

            # get foreground mask
            if "motion" in self.video_writers:
                fg = self.backSub.apply(frame)
                _, thresh = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY_INV)
                mask_fg = thresh
                # convert to 3 channels for output
                mask_fg_3 = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                self.video_writers["motion"].write(mask_fg_3)

                if "closure" in self.video_writers:
                    # threshold and inverse
                    close = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
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

                # get stationary contribution from frame
                contribution = cv2.bitwise_and(frame, frame, mask=mask_fg)

                av_frame = self.ma.add(contribution, mask_fg)
                if av_frame is not None:


                    av_frame = np.uint8(av_frame)
                    contribution = np.uint8(contribution)

                    self.video_writers["contribution"].write(contribution)
                    self.video_writers["average"].write(av_frame)

                    if self.enough_frames() and i >= 311:
                        self.plt.plot_histogram(initial, "input", True)
                        cv2.imwrite(self.output_dir + "input_frame" + ".png", initial)
                        cv2.imwrite(self.output_dir + "expanded_frame" + ".png", edit_frame)
                        self.plt.plot_histogram(av_frame, "final", True)
                        cv2.imwrite(self.output_dir + "final_frame" + ".png", av_frame)
                        if "motion" in self.video_writers:
                            cv2.imwrite(self.output_dir + "mask" + ".png", close_3)
                        finished = True
            i += 1


            # key to break playback
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        self.release_video()
        self.cap.release()
        cv2.destroyAllWindows()

