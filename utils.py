# coding=utf-8

from __future__ import print_function
import codecs
import os
import numpy as np
from glob import glob
import json
from skimage import draw
import cv2
from geometry_msgs.msg import Point


def mkdir(path):
    if not os.path.exists(path):
        print("mkdir %s" % path)
        os.mkdir(path)


def get_list(base_folder, pattern='.png', *args):
    """
    Returns the list of paths for each image matching the pattern (postfix), starting from the base_folder path.
    If additional args are specified, the returned list will contain tuples with the image path and additional files
    with same prefix of the image and matching the arg postfix.
    E.g.:
    Given these arguments:
     base_folder='.'
     pattern='.png'
     arg1='.txt'
     arg2='.yaml'
    And these files in the cwd: foo.png, foo.txt, foo.yaml, asd.png, asd.txt, asd.yaml, bar.png, lol.txt
    Returned list: [('foo.png', 'foo.txt', 'foo.yaml'), ('asd.png', 'asd.txt', 'asd.yaml')]

    :param base_folder: path from which images are recursively listed
    :param pattern: postfix of the filename of the images
    :param args: postfix of the filenames to be included in the returned list, with filenames matching the ones of the images
    :return: list of filenames with matching patterns
    """
    paths_list = []

    for path in os.walk(base_folder):
        for image_path in sorted(glob(os.path.join(path[0], '*'+pattern))):
            label_paths = map(lambda arg: image_path[:-(len(pattern))] + arg, args)

            if len(args) == 0:
                paths_list += [image_path]
            else:
                if all(map(os.path.exists, label_paths)):
                    paths_list.append((image_path, ) + tuple(label_paths))
                else:
                    print("Ignoring image %s. One or more corresponding files are missing (%s)." % (image_path, ', '.join(args)))

    return paths_list


def template_tokens_replacer(num_classes, num_filters, dataset_name):
    return lambda s: s.replace(u'€1', str(num_classes)).replace(u'€2', str(num_filters)).replace(u'€3', str(dataset_name))


def file_tokens_replacer(replacer, input_filename, output_filename):
    with codecs.open(input_filename, 'r', 'utf-8') as fr:
        s = fr.read()
        with codecs.open(output_filename, 'w', 'utf-8') as fw:
            fw.write(replacer(s))


def get_mask_from_json(json_filename, image_shape, class_id_dict):
    # A bug in the web-UI prevents the user from deleting lines (polygons with
    # just 2 vertices). The result is a null class label in the json file
    # ("label_class": null). We have to catch this case here.

    if os.path.exists(json_filename):
        with open(json_filename) as f:
            json_object = json.load(f)
    else:
        print("Labels file not found:", json_filename)
        return None, np.asarray([], dtype=np.int)

    labels_dicts = json_object['labels']

    num_object_instances = sum(map(lambda l: l['label_class'] is not None and len(l['vertices']) > 2, labels_dicts))

    mask = np.zeros((image_shape[0], image_shape[1], num_object_instances), dtype=np.bool)

    class_id_list = []
    index = 0
    for current_dict in labels_dicts:

        if current_dict['label_class'] is None or len(current_dict['vertices']) <= 2:
            print("Invalid label in:", json_filename)
            continue

        class_name_original = current_dict['label_class']
        
        # replace the spaces with underscore in label names
        class_name = class_name_original.replace(' ', '_')
        
        if class_name in class_id_dict.keys():
            class_id = class_id_dict[class_name]
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


def compute_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    xmin, xmax = np.where(rows)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]
    return xmin, xmax, ymin, ymax


def yolo_boundingbox_string(class_id, (xmin, xmax, ymin, ymax), image_shape):
    ih, iw = float(image_shape[0]), float(image_shape[1])
    w = ymax - ymin
    h = xmax - xmin
    xmid = xmin + (xmax - xmin) / 2.0
    ymid = ymin + (ymax - ymin) / 2.0
    return "%i %f %f %f %f\n" % (class_id, ymid/iw, xmid/ih, w/iw, h/ih)


def rescale_image(image, scale_factor):
    return cv2.resize(image, (int(image.shape[1]*scale_factor), int(image.shape[0]*scale_factor)))


def rescale_mask(mask, scale_factor):
    return cv2.resize(mask.astype(np.float), (int(mask.shape[1]*scale_factor), int(mask.shape[0]*scale_factor))) >= 0.5


def elevation(r, n):
    """
    Computes the angle between two vectors.
    Angle is 0 when the camera is in the surface plane and
    positive when the camera is above the plane.

    :param r: ray vector, from the object's position on the surface to the camera focal point (camera frame's origin).
    :param n: surface normal vector, pointing up.
    :return: angle between the ray and normal vectors.
    """
    return np.pi / 2 - np.arccos(np.dot(n, r) / (np.linalg.norm(n) * np.linalg.norm(r)))


def distance(p):
    """
    :type p: Union[numpy.array, geometry_msgs.msg.Point]
    """
    if isinstance(p, np.ndarray):
        return np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    elif isinstance(p, Point):
        return np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
    else:
        raise TypeError()
