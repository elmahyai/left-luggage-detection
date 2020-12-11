import copy
from imutils.video import FPS
import time
from static_object import *
from intensity_processing import *
from pathlib import Path
from distance_between import *
import cv2
gamma = 2


import threading


class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, key, value):
        self.cap.set(key, value)

    def start(self):
        if self.started:
            print('[Warning] Asynchronous video capturing is already started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())

        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()





from playsound import playsound

def left_luggage_detection():
    cap = VideoCaptureAsync('rtsp://admin:Admin123456@@192.168.1.13/1').start()

    fps = FPS().start()
    first_run = True
    (ret, frame) = cap.read()
    while not ret:
        (ret, frame) = cap.read()
    frame = imutils.resize(frame, width=450)
    (height, width, channel) = frame.shape
    image_shape = (height, width)
    rgb = IntensityProcessing(image_shape)

    bbox_last_frame_proposals = []
    static_objects = []
    count = 0
    while True:
        (ret, frame) = cap.read()
        if not ret:
            break
        else:
            #adjusted = adjust_gamma(frame, gamma=gamma) # gamma correction
            #frame = adjusted
            frame = imutils.resize(frame, width=450)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.dstack([frame, frame, frame])
            rgb.current_frame = frame  # .getNumpy()

            if first_run:
                old_rgb_frame = copy.copy(rgb.current_frame) # old frame is the new frame
                first_run = False


            rgb.compute_foreground_masks(rgb.current_frame) # compute foreground masks
            rgb.update_detection_aggregator()               # detect if new object proposed

            rgb_proposal_bbox = rgb.extract_proposal_bbox()     # bounding boxes of the areas proposed
            for box in rgb_proposal_bbox:
                print('box', box)

                width = box[2] - box[0]
                height = box[3] - box[1]
                print("width",width)
                print("height", height)
                area = width * height
                print("area", area)
                print('frame shape', frame.shape)
                area_ratio = area / (253 * 450)
                print("area_ratio", area_ratio)
                if area_ratio > .0001:
                    playsound("woop.mp3")

            foreground_rgb_proposal = rgb.proposal_foreground   # rgb proposals

            bbox_current_frame_proposals = rgb_proposal_bbox
            final_result_image = rgb.current_frame.copy()

            old_bbox_still_present = check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals,
                                                          old_rgb_frame, rgb.current_frame.copy())

            # add the old bbox still present in the current frame to the bbox detected
            bbox_last_frame_proposals = bbox_current_frame_proposals + old_bbox_still_present
            old_rgb_frame = rgb.current_frame.copy()
            #
            # # static object ######################
            # owner_frame = rgb.current_frame.copy()
            # added = 0
            # length = len(bbox_current_frame_proposals)
            # if len(bbox_last_frame_proposals) > 0:  # not on first frame of video
            #     for old in bbox_last_frame_proposals:
            #         old_drawn = False
            #         for curr in static_objects:
            #             if rect_similarity2(curr.bbox_info, old):
            #                 old_drawn = True
            #                 length -=1
            #                 break
            #         if not old_drawn:
            #             drawed_new = True
            #             owner_frame = dim_image2(owner_frame, old)
            #             draw_bounding_box2(owner_frame, old)
            #             dim_image2(owner_frame, old)
            #             cv2.imshow("dimmed", owner_frame)
            #             static_objects.append(StaticObject(old, owner_frame, 0))
            #             added += 1
            #     if drawed_new == True:
            #         reverse_image(rgb.current_frame, owner_frame)
            #         cv2.imshow("reverse", owner_frame)
            #         count += 1
            #
            # ##################################
            # print(len(static_objects))
            # cv2.imshow("owner frame dimmed", owner_frame)
            # print(added)

            draw_bounding_box(final_result_image, bbox_current_frame_proposals)
            draw_bounding_box(foreground_rgb_proposal, rgb_proposal_bbox)


            #mask_lg = rgb.foreground_mask_long_term
            #mask_sh = rgb.foreground_mask_short_term
            #long = cv2.bitwise_and(img, rgb.current_frame, mask=mask_lg)
            #cv2.imshow("long", long)
            #short = cv2.bitwise_and(img, rgb.current_frame, mask=mask_sh)
            #cv2.imshow("short", short)

            cv2.imshow('background modeling result', final_result_image)
            cv2.imshow('static object proposals, foreground_rgb_proposal', foreground_rgb_proposal)
            #cv2.imshow('frame', frame)
        cv2.waitKey(27)
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print(count)
    print(len(static_objects))
    cap.stop()
    cv2.destroyAllWindows()


def check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals, old_frame, current_frame):
    bbox_to_add = []
    if len(bbox_last_frame_proposals) > 0:  # not on first frame of video
        for old in bbox_last_frame_proposals:
            old_drawn = False
            for curr in bbox_current_frame_proposals:
                if rect_similarity2(old, curr):
                    old_drawn = True
                    break
            if not old_drawn:
                # Check if the area defined by the bounding box in the old frame and in the new one is still the same
                old_section = old_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                new_section = current_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                if norm_correlate(old_section, new_section)[0] > 0.9:
                    bbox_to_add.append(old)
    return bbox_to_add

left_luggage_detection()

