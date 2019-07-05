#!/usr/bin/env python
# coding=utf-8

"""
    File name: train_homer_new.py
    Author: Anatoli Eckert, Enrico Piazza
    Date created: 3/30/2018
    Date last modified: 4/23/2018
    Python Version: 3.5
"""

from __future__ import print_function
import os
import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
from config import Config
import utils
import visualize
from glob import glob
import json
from skimage import draw

from random import randint
import pickle
import time
import scipy.ndimage
from os.path import expanduser

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def get_list(base_folder, pattern='*.jpg', only_labeled=False):
    list = []

    for path in os.walk(base_folder):
        for image_path in glob(os.path.join(path[0], pattern)):
            label_path = image_path[:-(len(pattern)-1)] + "__labels.json"
            if not only_labeled:
                list += [image_path]
            else:
                if not os.path.exists(label_path):
                    print("Image %s not labeled. Ignoring." % image_path)
                else:
                    list += [image_path]

    old_list = [y for x in os.walk(base_folder) for y in glob(os.path.join(x[0], pattern))]

    if old_list != list:
        print("IMAGES NOT LABELED WILL BE IGNORED!")

    return list


def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        print(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
    else:
        print(json.dumps(json_thing, sort_keys=sort, indent=indents))
    return None


class MyConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "SocRob"

    # class_names = ["knife", "fork", "spoon", "box_yellow", "box_pink", "juice_small", "juice_big", "frame_erl",
    # "frame_spark", "mug_black", "mug_gray", "mug_yellow_dots", "mug_yellow_strips", "cocacola", "pringles"]
    class_names = ["pringles", "juice_big"]  # TODO load from unique config file, and generate yolo configuration

    NUM_CLASSES = 1 + len(class_names)
    IMAGE_MIN_DIM = 416
    IMAGE_MAX_DIM = 416

    class_ids = [class_id for class_id in range(1, len(class_names) + 1)]

    generate_master_informations = True

    max_masks_per_image = 3

    # starting value, previously generated images will be overwritten
    first_augmented_image_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    num_augmented_images = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    num_augmented_images_test = 0

    # intersection over union of bounding boxes of object instances:
    # if this value is > 0, objects will overlap. This value is increased
    # automatically if no valid object position is found
    iou_threshold = 0.25

    allow_scaling = True
    min_relative_size = 0.1
    max_relative_size = 0.5

    allow_rotating = True
    min_angle = 0
    max_angle = 359

    allow_random_brightness = True
    min_brightness = 0.5
    # max_brightness = 1.5 TODO

    allow_random_erasing = True
    random_erasing_p = 0.5
    random_erasing_s_l = 0.02
    random_erasing_s_h = 0.2
    random_erasing_r_1 = 0.3
    random_erasing_r_2 = 1 / random_erasing_r_1
    random_erasing_v_l = 0
    random_erasing_v_h = 255
    random_erasing_pixel_level = False
    random_erasing_erase_mask = True

    image_filename_pattern = '*.jpg'
    scale_factor_resolution = 1.0
    scale_factor_distance = 1.0
    visualize_loaded_objects = False

    base_folder = expanduser(sys.argv[1]+'/') if len(sys.argv) > 1 else os.getcwd()+'/'

    train_images_folder = base_folder + "train_source/"
    test_images_folder = base_folder + "test_source/"
    background_images_folder = base_folder + "negative/"
    output_folder = base_folder + "train/"

    pickled_objects_path = train_images_folder + "pickled_objects.pkl"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Folder", output_folder, "was created")

    background_image_path_list = get_list(background_images_folder, pattern='*.jpg')

    if len(background_image_path_list) < 1:
        print("Please add some background images into", background_images_folder)
        sys.exit()
    else:
        print("Found", len(background_image_path_list), "background images")


config = MyConfig()


def compute_bbox_centroid(start_row, end_row, start_col, end_col):
    center_row = start_row + ((end_row - start_row) / 2.0)
    center_col = start_col + ((end_col - start_col) / 2.0)
    return center_row, center_col


def compute_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    print(img.shape)
    print( rmin, rmax, cmin, cmax)
    return rmin, rmax, cmin, cmax


def rescale_image(image, scale_factor):
    rescaled_image = scipy.ndimage.zoom(image, (scale_factor, scale_factor, 1), order=1)
    return rescaled_image


def export_image_to_yolo(image_id, image):

    output_folder = config.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_filename = output_folder + "%06d.jpg" % image_id
    print("Writing", image_filename)
    plt.imsave(image_filename, image)


def export_labels_to_yolo(image_id, masks, class_ids):
    num_masks = len(masks)
    image_height = masks[0].shape[0]
    image_width = masks[0].shape[1]
    label_folder = config.output_folder
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    label_filename = "%06d.txt" % image_id

    object_list = []  # fill with labels per line
    for index in range(0, num_masks):
        current_object_line = ""
        current_object_class = class_ids[index] - 1  # mask rcnn class ids start at 1, yolo ids start at 0
        current_object_line += str(current_object_class)
        current_object_line += " "
        current_object_mask = masks[index]
        rmin, rmax, cmin, cmax = compute_bbox(current_object_mask)
        bbox_width = cmax - cmin
        bbox_height = rmax - rmin
        center_row, center_col = compute_bbox_centroid(rmin, rmax, cmin, cmax)

        current_object_line += str(center_col / (image_width * 1.0))
        current_object_line += " "

        current_object_line += str(center_row / (image_height * 1.0))
        current_object_line += " "

        current_object_line += str(bbox_width / (image_width * 1.0))
        current_object_line += " "

        current_object_line += str(bbox_height / (image_height * 1.0))
        current_object_line += " "

        object_list.append(current_object_line)

    output_path = label_folder + label_filename
    print("Writing", output_path)
    output_file = open(output_path, 'w')
    for line in object_list:
        output_file.write("%s\n" % line)
    output_file.close()


def export_source_labels_to_yolo(output_path, mask, class_ids):
    num_masks = mask.shape[2] if len(mask.shape) == 3 else 1
    image_height = mask.shape[0]
    image_width = mask.shape[1]

    object_list = []  # fill with labels per line
    for index in range(0, num_masks):
        current_object_line = ""
        current_object_class = class_ids[index] - 1  # mask rcnn class ids start at 1, yolo ids start at 0
        current_object_line += str(current_object_class)
        current_object_line += " "
        current_object_mask = mask[:, :, index]
        rmin, rmax, cmin, cmax = compute_bbox(current_object_mask)
        bbox_width = cmax - cmin
        bbox_height = rmax - rmin
        center_row, center_col = compute_bbox_centroid(rmin, rmax, cmin, cmax)

        current_object_line += str(center_col / (image_width * 1.0))
        current_object_line += " "

        current_object_line += str(center_row / (image_height * 1.0))
        current_object_line += " "

        current_object_line += str(bbox_width / (image_width * 1.0))
        current_object_line += " "

        current_object_line += str(bbox_height / (image_height * 1.0))
        current_object_line += " "

        object_list.append(current_object_line)
        print(center_col, center_row, image_width, image_height, bbox_width, bbox_height)

    print("Writing", output_path)
    output_file = open(output_path, 'w')
    for line in object_list:
        output_file.write("%s\n" % line)
    output_file.close()


def apply_random_brightness(image, random_brightness_factor):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)

    image1[:, :, 2] = image1[:, :, 2] * random_brightness_factor
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def compute_random_position(image_shape, object_shape):
    num_rows = object_shape[0]  # rows
    num_cols = object_shape[1]  # cols
    min_x = 0
    max_x = image_shape[0] - object_shape[0]
    if max_x <= 0:
        max_x = 1
    min_y = 0
    max_y = image_shape[1] - object_shape[1]
    if max_y <= 0:
        max_y = 1
    start_x = randint(min_x, max_x)
    start_y = randint(min_y, max_y)
    return start_x, start_x + num_rows, start_y, start_y + num_cols


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    if iou < 0:
        print("iou =", iou, "< 0")
    if iou > 1:
        print("iou =", iou, "> 1")
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def rescale_masks(masks, scale_factor):
    """
    Rescales incoming masks which can be of shape (H,W), (H,W,1) or (H,W,NUM_OBJECTS)
    """
    incoming_shape_len = len(masks.shape)
    num_objects = 1
    if incoming_shape_len == 3:
        num_objects = masks.shape[2]
    rescaled_masks = None
    for object_index in range(0, num_objects):
        if num_objects > 1:
            single_mask = masks[:, :, object_index]
        else:
            single_mask = masks
        single_mask = scipy.ndimage.zoom(single_mask, (scale_factor, scale_factor), order=0)
        if rescaled_masks is None:
            rescaled_masks = single_mask
        else:
            rescaled_masks = np.dstack((rescaled_masks, single_mask))
    if len(rescaled_masks.shape) == 2 and incoming_shape_len == 3:
        rescaled_masks = np.reshape(rescaled_masks, (rescaled_masks.shape[0], rescaled_masks.shape[1], 1))
    # print("masks shape", masks.shape, "->", rescaled_masks.shape)
    return rescaled_masks


def visualize_patches(image_patch, mask_patch, class_name):
    fig, axarr = plt.subplots(1, 3)
    axarr[0].set_title(class_name)
    axarr[1].set_title("mask")
    axarr[2].set_title("union")
    axarr[0].imshow(image_patch / 255.0)
    axarr[1].imshow(mask_patch)
    mask_rgb = np.dstack([mask_patch] * 3)
    axarr[2].imshow(image_patch * mask_rgb)
    # self.fig = fig
    # self.axarr = axarr


class MyDataset(utils.Dataset):

    def __init__(self):
        super(MyDataset, self).__init__(class_map=None)

        self.class_id_dict = None
        self.class_names_dict = None
        self.background_images_dict = None
        self.source_image_path_list = None
        self.object_dict = None

        print("Loading background images")
        self.load_background_images()
        print("Done.")

        self.init_class_names()

        self.base_folder = config.train_images_folder
        print(config.num_augmented_images, "images will be generated for training")

        # Find all images in source folder

        self.source_image_path_list = get_list(self.base_folder, pattern=config.image_filename_pattern, only_labeled=True)
        num_source_images = len(self.source_image_path_list)
        if num_source_images < 1:
            print("Could not find any images in", self.base_folder, "with the pattern", config.image_filename_pattern)
            sys.exit()

        num_source_images = len(self.source_image_path_list)

        for i in range(0, num_source_images):
            image_path = self.source_image_path_list[i]
            json_path = image_path[:-4] + "__labels.json"
            self.add_image(config.NAME, image_id=i, path=image_path, json_path=json_path)

        for i in range(num_source_images, num_source_images + config.num_augmented_images):
            self.add_image(config.NAME, image_id=i, path="")

        self.extract_masks_and_image_patches(config.pickled_objects_path)

        self.compute_source_images_stats()
        self.compute_test_images_stats()

    def compute_source_images_stats(self):
        print("\n### source images stats:")

        background_images_widths = map(lambda i: i["image"].shape[1], self.background_images_dict.values())
        min_background_width = np.min(background_images_widths)
        max_background_width = np.max(background_images_widths)
        avg_background_width = np.average(background_images_widths)

        print("\n num background images: %i" % len(self.background_images_dict.values()))
        print("min background width: %i" % min_background_width)
        print("min background width: %i" % avg_background_width)
        print("min background width: %i" % max_background_width)

        for class_id in self.object_dict.keys():
            patches_list = self.object_dict[class_id]
            min_relative_width = None
            avg_relative_width = None
            max_relative_width = None

            class_mask_widths = map(lambda p: p["mask"].shape[1], patches_list)

            try:
                min_relative_width = float(np.min(class_mask_widths))/max_background_width
                max_relative_width = float(np.max(class_mask_widths))/min_background_width
                avg_relative_width = float(np.average(class_mask_widths))/avg_background_width
            except ValueError as e:
                print(e)

            print("\n class: %s id: %i" % (self.class_names_dict[class_id], class_id))
            print("num masks:          %i" % len(patches_list))
            if min_relative_width is not None and avg_relative_width is not None and max_relative_width is not None:
                print("min relative width: %0.3f" % min_relative_width)
                print("avg relative width: %0.3f" % avg_relative_width)
                print("max relative width: %0.3f" % max_relative_width)

    @staticmethod
    def compute_test_images_stats():
        print("\n### test images stats:")

        test_labels_path_list = get_list(config.test_images_folder, "*.txt")

        classes = []
        widths = []
        heights = []

        for path in test_labels_path_list:
            with open(path, 'r') as f:
                bbs = f.read().splitlines()
                classes += map(lambda l: int(l.split(' ')[0]), bbs)
                widths += map(lambda l: float(l.split(' ')[3]), bbs)
                heights += map(lambda l: float(l.split(' ')[4]), bbs)

        min_relative_width = np.min(widths)
        max_relative_width = np.max(widths)
        avg_relative_width = np.average(widths)

        print("num bounding boxes:          %i" % len(classes))
        print("min relative width: %0.3f" % min_relative_width)
        print("avg relative width: %0.3f" % avg_relative_width)
        print("max relative width: %0.3f" % max_relative_width)

    def init_class_names(self):
        class_id_dict = {}
        class_id_counter = 1
        for class_name in config.class_names:
            class_id_dict[class_name] = class_id_counter
            self.add_class(config.NAME, class_id_counter, class_name)
            class_id_counter += 1

        self.class_id_dict = class_id_dict
        class_names_dict = {v: k for k, v in self.class_id_dict.items()}
        self.class_names_dict = class_names_dict
        for key in class_names_dict:
            print("id", key, "name", class_names_dict[key])

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this method to load instance masks
        and return them in the form of am array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """

        # A bug in the web-UI prevents the user from deleting lines (polygons with
        # just 2 vertices). The result is a null class label in the json file
        # ("label_class": null). We have to catch this case here.

        image = self.load_image(image_id)
        image_shape = (image.shape[0], image.shape[1])

        file_path = self.image_info[image_id]['json_path']
        if os.path.exists(file_path):
            with open(file_path) as input_file:
                json_object = json.load(input_file)
        else:
            print("Labels file not found for image", self.image_info[image_id])
            return None, np.asarray([], dtype=np.int)

        labels_dicts = json_object['labels']

        num_object_instances = 0
        for current_dict in labels_dicts:
            if current_dict['label_class'] is not None and len(current_dict['vertices']) > 2:
                num_object_instances += 1

        mask = np.zeros((image_shape[0], image_shape[1], num_object_instances), dtype=np.bool)
        class_id_list = []
        index = 0
        for current_dict in labels_dicts:

            if current_dict['label_class'] is None or len(current_dict['vertices']) <= 2:
                print("Invalid label in image id: %i" % image_id)
                continue

            class_name = current_dict['label_class']

            if class_name in self.class_id_dict.keys():
                class_id = self.class_id_dict[class_name]
                class_id_list.append(class_id)

                try:
                    vertices = current_dict['vertices']
                except ValueError:
                    print("\nERROR: Could not extract vertices for image", json_object['image_filename'])
                    continue

                vertex_row_coords = []
                vertex_col_coords = []
                for i in range(0, len(vertices)):
                    vertex_row_coords.append(vertices[i]['y'])
                    vertex_col_coords.append(vertices[i]['x'])

                # noinspection PyArgumentList
                fill_row_coords, fill_col_coords = draw.polygon(np.array(vertex_row_coords),
                                                                np.array(vertex_col_coords), image_shape)
                mask[fill_row_coords, fill_col_coords, index] = True
                index += 1

        class_ids = np.asarray(class_id_list, dtype=np.int)
        return mask, class_ids

    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,3] Numpy array.
        """
        image = None
        image_path = self.image_info[image_id]['path']

        if os.path.exists(image_path):
            image = skimage.io.imread(image_path)
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
        else:
            print("Image ", image_path, " not found")

        return image

    # def image_reference(self, image_id):
    #     return ""
    #
    # def visualize_objects(self):
    #     for current_class_id in self.object_dict:
    #         patches_list = self.object_dict[current_class_id]
    #         for i in range(0, len(patches_list)):
    #             current_dict = patches_list[i]
    #             image_patch = current_dict["image"]
    #             mask_patch = current_dict["mask"]
    #             visualize_patches(image_patch, mask_patch, self.class_names_dict[current_class_id])

    def compute_object_patches(self, image, mask, class_ids):
        print("Extracting object labels from image.")
        num_objects = mask.shape[2]

        scale_factor = config.scale_factor_resolution * config.scale_factor_distance
        if np.abs(1.0 - scale_factor) > 0.01:
            image = rescale_image(image, scale_factor)
            mask = rescale_masks(mask, scale_factor)

        for i in range(0, num_objects):
            object_instance = mask[:, :, i]
            rmin, rmax, cmin, cmax = compute_bbox(object_instance)

            # patch_shape = (rmax - rmin, cmax - cmin, 3)
            # image_patch = np.zeros(patch_shape)
            image_patch = image[rmin:rmax, cmin:cmax, :]
            mask_patch = object_instance[rmin:rmax, cmin:cmax]
            current_class_id = class_ids[i]
            current_dict = {"image": image_patch, "mask": mask_patch}
            self.object_dict[current_class_id].append(current_dict)
            # visualize_patches(image_patch, mask_patch, self.class_names_dict[current_class_id])

    def extract_masks_and_image_patches(self, pickled_objects_path="object_dict.pkl"):
        """
        To augment the data, we need the isolated object masks with corresponding image patch
        object_dict['label_class'] = {image_patches : [image_patch1, ...], masks : [mask_1, ...]}
        """

        if os.path.isfile(pickled_objects_path):
            print("\nLoading", pickled_objects_path)
            with open(pickled_objects_path, 'rb') as f:
                self.object_dict = pickle.load(f)

        else:
            print("\nExtracted objects will be stored in", pickled_objects_path, "for future usage")
            object_dict = {}

            for key in self.class_names_dict.keys():
                object_dict[key] = []

            self.object_dict = object_dict
            for image_id in range(0, len(self.source_image_path_list)):
                print("Extracting objects from image ", image_id, "  path: ", self.source_image_path_list[image_id])
                image = self.load_image(image_id)
                mask, class_ids = self.load_mask(image_id)
                if len(class_ids) == 0:
                    continue
                self.compute_object_patches(image, mask, class_ids)

            if len(self.source_image_path_list) > 0:
                print("\nSaving", pickled_objects_path)
                with open(pickled_objects_path, 'wb') as f:
                    pickle.dump(self.object_dict, f, pickle.HIGHEST_PROTOCOL)
                print("Done.")

    def get_valid_position(self, background_image, mask_patch, iou_threshold=0.0,
                           occupied_regions_list=None, recursion_counter=0):

        if occupied_regions_list is None:
            occupied_regions_list = []

        start_x, end_x, start_y, end_y = compute_random_position(background_image.shape, mask_patch.shape)
        bb1 = {'x1': start_x, 'x2': end_x, 'y1': start_y, 'y2': end_y}

        valid_position = True
        if len(occupied_regions_list) == 0:
            occupied_regions_list.append([start_x, end_x, start_y, end_y])
            return valid_position, start_x, end_x, start_y, end_y

        else:
            for i in range(0, len(occupied_regions_list)):
                current_start_x, current_end_x, current_start_y, current_end_y = occupied_regions_list[i]
                bb2 = {'x1': current_start_x, 'x2': current_end_x, 'y1': current_start_y, 'y2': current_end_y}
                iou = get_iou(bb1, bb2)
                if iou > iou_threshold:
                    valid_position = False

        if valid_position:
            occupied_regions_list.append([start_x, end_x, start_y, end_y])
            return valid_position, start_x, end_x, start_y, end_y

        elif recursion_counter <= 10:
            recursion_counter += 1
            print("No free region found with iou_threshold =", iou_threshold,
                  ". Increasing to", min(iou_threshold + 0.1, 1.0))
            return self.get_valid_position(background_image, mask_patch, min(iou_threshold + 0.1, 1.0),
                                           occupied_regions_list, recursion_counter)
        else:
            print("WARNING No free region found with iou_threshold =", iou_threshold)
            return valid_position, start_x, end_x, start_y, end_y

    def load_background_images(self):
        background_images_dict = {}
        for i in range(0, len(config.background_image_path_list)):
            background_images_dict[i] = {}
            image_path = config.background_image_path_list[i]
            image = skimage.io.imread(image_path)
            background_images_dict[i]["image"] = image

        self.background_images_dict = background_images_dict

    def get_random_background_image(self):

        i = randint(0, len(self.background_images_dict) - 1)
        return self.background_images_dict[i]["image"].copy()

        # background_image_id = randint(0, len(config.background_image_path_list) - 1)
        # image_path = config.background_image_path_list[background_image_id]
        # background_image = skimage.io.imread(image_path)
        # return background_image

    def generate_augmented_image(self):
        """
        Creates an augmented image with random background, random objects and random
        object positions. With iou_threshold > 0.0 objects are allowed to overlap.
        """
        print("\nGenerating augmented image")

        background_image = self.get_random_background_image()
        selected_class_ids = []
        inserted_class_ids = []
        background_masks = []
        occupied_regions_list = []
        num_masks = randint(1, config.max_masks_per_image)

        for i in range(num_masks):
            selected_class_ids.append(random.choice(self.object_dict.keys()))

        for class_id in selected_class_ids:
            print("  Selected class:", self.class_names_dict[class_id])

            patches_list = self.object_dict[class_id]
            num_patches = len(patches_list)

            assert num_patches > 0
            random_index = randint(0, num_patches - 1)
            current_dict = patches_list[random_index]
            image_patch = current_dict["image"].copy()
            mask_patch = current_dict["mask"].copy()

            if config.allow_scaling:
                print("    Rescaling mask patch")
                random_relative_size = round(random.uniform(config.min_relative_size, config.max_relative_size), 3)
                relative_mask_height = float(mask_patch.shape[0]) / background_image.shape[0]
                relative_mask_width = float(mask_patch.shape[1]) / background_image.shape[1]
                scale_factor = min(random_relative_size / relative_mask_width,
                                   random_relative_size / relative_mask_height)

                image_patch = scipy.ndimage.zoom(image_patch, (scale_factor, scale_factor, 1), order=1)
                mask_patch = scipy.ndimage.zoom(mask_patch, (scale_factor, scale_factor), order=0)

            if config.allow_rotating:
                print("    Rotating mask patch")
                random_angle = randint(config.min_angle, config.max_angle)
                image_patch = scipy.ndimage.rotate(image_patch, random_angle)
                mask_patch = scipy.ndimage.rotate(mask_patch, random_angle, order=0)

            if config.allow_random_erasing:
                print("    Random erasing mask patch")

                img_h, img_w, img_c = image_patch.shape
                p_1 = np.random.rand()

                if p_1 < config.random_erasing_p:

                    while True:
                        s = np.random.uniform(config.random_erasing_s_l, config.random_erasing_s_h) * img_h * img_w
                        r = np.random.uniform(config.random_erasing_r_1, config.random_erasing_r_2)
                        w = int(np.sqrt(s / r))
                        h = int(np.sqrt(s * r))
                        left = np.random.randint(0, img_w)
                        top = np.random.randint(0, img_h)

                        if left + w <= img_w and top + h <= img_h:
                            break

                    if config.random_erasing_erase_mask:
                        mask_patch[top:top + h, left:left + w] = 0
                    else:

                        if config.random_erasing_pixel_level:
                            c = np.random.uniform(config.random_erasing_v_l, config.random_erasing_v_h, (h, w, img_c))
                        else:
                            c = (np.random.uniform(config.random_erasing_v_l, config.random_erasing_v_h),
                                 np.random.uniform(config.random_erasing_v_l, config.random_erasing_v_h),
                                 np.random.uniform(config.random_erasing_v_l, config.random_erasing_v_h))

                        image_patch[top:top + h, left:left + w, :] = c

            mask_rgb = np.dstack([mask_patch] * 3)

            print("    Finding position in image")
            valid_position, start_x, end_x, start_y, end_y = self.get_valid_position(background_image,
                                                                                     mask_patch,
                                                                                     config.iou_threshold,
                                                                                     occupied_regions_list)

            if not valid_position:
                return None

            # make sure object mask is not bigger than background image
            assert(mask_patch.shape[0] <= background_image.shape[0] and
                   mask_patch.shape[1] <= background_image.shape[1])

            print("    Applying patch")
            background_image_patch = background_image[start_x:end_x, start_y:end_y, :]
            background_image_patch[mask_rgb == True] = image_patch[mask_rgb == True]
            background_image[start_x:end_x, start_y:end_y, :] = background_image_patch

            current_mask = np.zeros((background_image.shape[0], background_image.shape[1]))
            current_mask[start_x:end_x, start_y:end_y] = mask_patch
            background_masks.append(current_mask)

            inserted_class_ids.append(class_id)

        if config.allow_random_brightness:
            print("  Applying random brightness")
            random_brightness_factor = np.random.uniform() + config.min_brightness
            background_image = apply_random_brightness(background_image, random_brightness_factor)
            # TODO:
            # random brightness for patches and background should be separate, to emulate
            # different lighting of objects and environment

        return background_image, background_masks, np.asarray(inserted_class_ids, dtype=np.int)

    def generate_data(self):
        """
        Exports augmented images and YOLO labels.
        """

        if config.generate_master_informations:
            for i in range(0, len(self.source_image_path_list)):
                path = self.source_image_path_list[i][:-4]+'.txt'
                mask, class_ids = self.load_mask(i)

                print(i, path)

                export_source_labels_to_yolo(path, mask, class_ids)

        output_image_index = config.first_augmented_image_id

        for current_image_id in range(0, config.num_augmented_images):
            augmented_object = None

            while augmented_object is None:
                augmented_object = self.generate_augmented_image()

            image, masks, class_ids = augmented_object
            export_image_to_yolo(output_image_index, image)
            export_labels_to_yolo(output_image_index, masks, class_ids)
            output_image_index += 1


# Load and display random samples
def visualize_mask_samples(dataset_train):
    image_ids = [0, 1, 2, 3]
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


def generate_data():
    loading_start_time = time.time()
    dataset_train = MyDataset()
    generating_start_time = loading_end_time = time.time()
    dataset_train.generate_data()
    generating_end_time = time.time()

    print("Loading data took", loading_end_time - loading_start_time, "seconds")
    print("Generating data took", generating_end_time - generating_start_time, "seconds")


if __name__ == "__main__":
    generate_data()
