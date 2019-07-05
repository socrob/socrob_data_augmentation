# coding=utf-8

from __future__ import print_function
import numpy as np
import sys
from os import path
import yaml
import image_geometry
import imageio
from random import randint, choice
import pickle
from collections import defaultdict
from geometry_msgs.msg import PointStamped

import utils
from support_surface_saver_ros import support_surface_object


class NotFound(Exception):
    pass


class BackgroundImage:
    def __init__(self, filename=None):
        if filename is None:
            return

        self.filename = filename

        with open(self.filename, 'rb') as f:
            support_surface = pickle.load(f)
            assert(isinstance(support_surface, support_surface_object.SupportSurface))

            self.image = support_surface.image
            self.depth = support_surface.depth
            self.camera = image_geometry.PinholeCameraModel()
            self.camera.fromCameraInfo(support_surface.camera_info)
            self.surface_normal = np.array(support_surface.surface_normal)

    def has_support_surface(self):
        return np.sum(np.logical_not(np.isnan(self.depth))) > 0

    def copy(self):
        b = BackgroundImage()
        b.filename = self.filename
        b.image = self.image.copy()
        b.depth = self.depth.copy()
        b.camera = self.camera
        b.surface_normal = self.surface_normal
        return b


class SourceImage:
    def __init__(self, filename, label_path, info_path, class_name_to_id, class_id_to_name):
        self.filename = filename
        self.image = imageio.imread(self.filename)

        # compute mask from JSON label
        self.masks, self.class_ids = utils.get_mask_from_json(label_path, self.image.shape, class_name_to_id)
        self.patches = list()

        with open(info_path, 'r') as f:
            image_info_object = yaml.load(f)

        # get reference point and surface normal from info file
        self.surface_normals_dict = dict()
        self.reference_points_dict = image_info_object[path.basename(filename)]['reference_points']
        reference_up_points_dict = image_info_object[path.basename(filename)]['reference_up_points']

        for reference_point_name in self.reference_points_dict.keys():
            point_stamped = self.reference_points_dict[reference_point_name]
            up_point_stamped = reference_up_points_dict[reference_point_name]
            point = np.array((point_stamped.point.x, point_stamped.point.y, point_stamped.point.z))
            up_point = np.array((up_point_stamped.point.x, up_point_stamped.point.y, up_point_stamped.point.z))
            self.surface_normals_dict[reference_point_name] = up_point - point

        # extract patches
        for i in range(len(self.class_ids)):
            xmin, xmax, ymin, ymax = utils.compute_bbox(self.masks[:, :, i])
            image_patch = self.image[xmin:xmax, ymin:ymax, :].copy()
            mask_patch = self.masks[xmin:xmax, ymin:ymax, i].copy()
            class_id = self.class_ids[i]
            class_name = class_id_to_name[class_id]

            if class_name not in self.reference_points_dict.keys():
                print("\tmissing reference point for labeled class %s" % class_name)
                continue

            self.patches.append(Patch(image_patch, mask_patch, class_id,
                                      self.surface_normals_dict[class_name],
                                      self.reference_points_dict[class_name]))

    def get_patches(self):
        return self.patches


class Patch:
    def __init__(self, image, mask, class_id, surface_normal, reference_point):
        assert(isinstance(image, np.ndarray) and image.shape[2] == 3)
        assert(isinstance(mask, np.ndarray))
        assert(isinstance(class_id, int))
        assert(isinstance(reference_point, PointStamped))

        self.original = None
        self.image = image
        self.mask = mask
        self.class_id = class_id
        self.surface_normal = surface_normal
        self.reference_point = reference_point
        self.usages = 0

        # compute elevation and distance
        point = np.array((reference_point.point.x, reference_point.point.y, reference_point.point.z))
        self.elevation = utils.elevation(-point, self.surface_normal)
        self.distance = utils.distance(point)

    def rescale(self, target_distance):
        # rescale to fit a distance in meters
        self.image = utils.rescale_image(self.image, self.distance / target_distance)
        self.mask = utils.rescale_mask(self.mask, self.distance / target_distance)
        self.distance = target_distance
        self.reference_point = None

    def get_elevation(self):
        return self.elevation

    def get_class_id(self):
        return self.class_id

    def is_original(self):
        return self.original is None

    def increase_usages(self):
        if self.is_original():
            self.usages += 1
        else:
            self.original.increase_usages()

    def get_usages(self):
        if self.is_original():
            return self.usages
        else:
            return self.original.get_usages()

    def copy(self):
        if not self.is_original():
            print("\n\n\n WARNING: copying non-original patch \n\n\n")

        p = Patch(self.image.copy(), self.mask.copy(), self.class_id, self.surface_normal, self.reference_point)
        p.usages = None
        p.original = self
        return p


class AugmentedImage:
    def __init__(self, background_image=None):
        assert(background_image is None or isinstance(background_image, BackgroundImage))

        self.background_image = None
        self.class_ids = list()
        self.boundingboxes = list()
        self.masks = list()
        self.masks_union = None

        self.set_background_image(background_image)

    # Static values set from configuration
    elevation_bucket_size = None
    max_attempts = None

    def set_background_image(self, background_image):
        if self.background_image is not None:
            raise Exception("background image already set")

        assert(isinstance(background_image, BackgroundImage))

        self.background_image = background_image.copy()
        self.masks_union = np.zeros(self.background_image.depth.shape, dtype=np.bool)

    def apply_patch(self, patches_by_elevation):
        if self.background_image is None:
            raise Exception("background image not set")

        assert(isinstance(patches_by_elevation, defaultdict))

        try:
            xmin, xmax, ymin, ymax, patch = self.find_valid_position(patches_by_elevation)
        except NotFound as e:
            print("\tAugmentedImage.apply_patch:", e)
            return

        # add patch's rgb image
        mask_rgb = np.dstack([patch.mask]*3)

        background_image_patch = self.background_image.image[xmin:xmax, ymin:ymax, :]
        background_image_patch[mask_rgb] = patch.image[mask_rgb]
        self.background_image.image[xmin:xmax, ymin:ymax, :] = background_image_patch

        # add patch's mask
        new_mask = np.zeros(self.background_image.depth.shape, dtype=np.bool)
        new_mask[xmin:xmax, ymin:ymax] = patch.mask
        self.masks.append(new_mask)
        self.add_to_masks_union(new_mask)

        # add patch's class id and patch position
        self.class_ids.append(patch.class_id)
        self.boundingboxes.append((xmin, xmax, ymin, ymax))

        # increase patch's usage statistics
        patch.increase_usages()

    def find_valid_position(self, patches_by_elevation):
        assert(AugmentedImage.elevation_bucket_size is not None)
        assert(AugmentedImage.max_attempts is not None)

        # support surface constraint: only place the patch with its bottom on the support surface
        support_surface_constraint = np.logical_not(np.isnan(self.background_image.depth))

        # occupancy constraint: do not consider positions already occupied by other patches
        occupancy_constraint = np.logical_not(self.masks_union)

        # all constraints must hold
        constraints = np.logical_and(support_surface_constraint, occupancy_constraint)

        # i_mask is the array of indices where the constraints hold
        i_mask = np.array(np.where(constraints))
        num_choices = i_mask.shape[1]

        for _ in range(AugmentedImage.max_attempts):
            # choose a random index where to position the bottom of the patch (bottom in the image is max y)
            u, v = i_mask[:, np.random.choice(range(num_choices))]

            # ray_xyz is the ray from the camera focal point (origin of camera frame), going through pixels u,v
            ray_x = (v - self.background_image.camera.cx()) / self.background_image.camera.fx()
            ray_y = (u - self.background_image.camera.cy()) / self.background_image.camera.fy()

            # p is the intersection point between the ray and the support surface.
            # p.z is initially 1 so all elements must be multiplied by the z coord of the intersection point.
            p = np.array((ray_x, ray_y, 1.)) * self.background_image.depth[u, v]

            # n is the surface normal in camera frame, pointing toward the z axis in fixed frame
            n = self.background_image.surface_normal

            # rescale the patch considering the distance of the support surface
            distance = utils.distance(p)

            # randomly choose a patch from the list of patches with elevation closest to the point p (elevations are bucketed)
            elevation_buckets = np.array(patches_by_elevation.keys())

            if elevation_buckets.size == 0:
                continue

            point_elevation_bucket = int(utils.elevation(-p, n) / AugmentedImage.elevation_bucket_size)
            closest_elevation_index = np.argmin(np.fabs(elevation_buckets - point_elevation_bucket))
            patches = patches_by_elevation[elevation_buckets[closest_elevation_index]]
            original_patch = choice(patches)
            assert(isinstance(original_patch, Patch))

            patch = original_patch.copy()

            # rescale the patch to fit the distance
            patch.rescale(distance)

            # compute the new mask for this patch
            patch_h, patch_w = patch.mask.shape
            xmin = u - patch_h
            xmax = u
            ymin = v - patch_w/2
            ymax = ymin + patch_w

            new_mask = np.zeros(self.background_image.depth.shape, dtype=np.bool)

            # check that the rescaled patch is inside image bounds
            if xmin < 0 or xmax > new_mask.shape[0] or ymin < 0 or ymax > new_mask.shape[1]:
                continue

            new_mask[xmin:xmax, ymin:ymax] = patch.mask

            # check that the new mask does not intersect other masks
            if np.sum(self.get_masks_intersection(new_mask)) == 0:
                return xmin, xmax, ymin, ymax, patch

        raise NotFound("Could not find a valid position after %i attempts" % AugmentedImage.max_attempts)

    def get_masks_intersection(self, mask):
        if self.background_image is None:
            raise Exception("image not set")
        if self.masks_union is None:
            raise Exception("masks_union not initialised")

        return np.logical_and(self.masks_union, mask)

    def add_to_masks_union(self, mask):
        if self.masks_union is None:
            raise Exception("masks_union not set")

        self.masks_union = np.logical_or(self.masks_union, mask)

    def get_masks_union(self, mask):
        if self.background_image is None:
            raise Exception("image not set")

        return np.logical_or(self.masks_union, mask)

    def export(self, filename):

        # save image and labels
        imageio.imwrite(filename, self.background_image.image)

        with open(path.splitext(filename)[0] + '.txt', 'w') as f:
            for class_id, boundingbox in zip(self.class_ids, self.boundingboxes):
                f.write(utils.yolo_boundingbox_string(class_id, boundingbox, self.background_image.image.shape))


class DataAugmentation:

    def __init__(self, base_path='.'):
        self.base_path = base_path
        self.background_images_list = list()
        self.background_images_dict = dict()
        self.source_images_list = list()
        self.patches_dict = defaultdict(lambda: defaultdict(list))
        self.patches_list = list()

        # load data augmentation config
        config_filename = path.join(self.base_path, "config/data_augmentation.yaml")
        with open(config_filename, 'r') as f:
            config = yaml.load(f)

        if isinstance(config['class_id_dict'], dict):
            self.class_name_to_id = config['class_id_dict']
        elif isinstance(config['class_id_dict'], list):
            self.class_name_to_id = dict(zip(config['class_id_dict'], range(len(config['class_id_dict']))))

        self.class_id_to_name = {v: k for k, v in self.class_name_to_id.iteritems()}
        self.num_augmented_images = config['num_augmented_images']
        self.max_masks_per_image = config['max_masks_per_image']
        AugmentedImage.elevation_bucket_size = self.elevation_bucket_size = float(config['elevation_bucket_size'])
        AugmentedImage.max_attempts = config['max_attempts']

        self.load_pickled_patches = bool(config['load_pickled_patches'])
        assert(isinstance(self.load_pickled_patches, bool))

        self.augmented_images_path = path.expanduser(config['augmented_images_path'])
        self.background_images_path = path.expanduser(config['background_images_path'])
        self.background_images_choice = config['background_images_choice']
        self.source_images_path = path.expanduser(config['source_images_path'])
        self.pickled_patches_filename = path.join(self.source_images_path, "pickled_patches.pkl")

        # Make sure the output folders exists
        utils.mkdir(self.augmented_images_path)

    def load_background_images(self):
        print("Loading background images...")

        for bi in utils.get_list(self.background_images_path, '*_pickle.pkl'):
            background_image = BackgroundImage(bi)
            if background_image.has_support_surface():
                self.background_images_list.append(background_image)
                self.background_images_dict[path.basename(background_image.filename)] = background_image

        print("Loaded %i background images." % len(self.background_images_list))

    def load_patches(self):
        """
        Load patches from pickle if available,
        otherwise load source images and extract patches
        :return:
        """
        if self.load_pickled_patches and path.isfile(self.pickled_patches_filename):
            print("Loading patches from", self.pickled_patches_filename, "...")

            with open(self.pickled_patches_filename, 'rb') as f:
                self.patches_list = pickle.load(f)

            for patch in self.patches_list:
                elevation_bucket = int(patch.get_elevation() / self.elevation_bucket_size)
                self.patches_dict[patch.get_class_id()][elevation_bucket].append(patch)

            print("Loaded %i patches." % len(self.patches_list))

        else:

            print("Loading patches from source images...")

            for image_path, label_path, info_path in utils.get_list(self.source_images_path, '.png', '__labels.json', '.yaml'):
                # load image
                print("\tLoading source image:", image_path)
                source_image = SourceImage(image_path, label_path, info_path, self.class_name_to_id, self.class_id_to_name)

                # extract patches
                for patch in source_image.get_patches():
                    self.patches_dict[patch.get_class_id()][int(patch.get_elevation() / self.elevation_bucket_size)].append(patch)
                    print("\t\tAdding patch:", self.class_id_to_name[patch.get_class_id()])
                    self.patches_list.append(patch)

                # unload image
                # del source_image # try this if too much memory is used

            if len(self.patches_list) > 0:
                print("Saving", self.pickled_patches_filename, "...")

                with open(self.pickled_patches_filename, 'wb') as f:
                    pickle.dump(self.patches_list, f, pickle.HIGHEST_PROTOCOL)

            else:
                print("No patches loaded, not saving the pickled patches.")

            print("Done.")

    def print_patches_statistics(self):
        # some ugly code to print the table with class name, number of elevation buckets, total number of patches
        print("Patches statistics:")
        class_names = self.class_id_to_name.values()
        for class_name in class_names:
            class_id = self.class_name_to_id[class_name]
            num_elevations = len(self.patches_dict[class_id].keys())
            num_patches = sum(map(len, self.patches_dict[class_id].values()))
            print("-\t%s (id:%3i): %4i elevations, %4i patches." % (class_name, class_id, num_elevations, num_patches))

            if num_patches == 0:
                continue

            patches_per_bucket = defaultdict(lambda: 0)
            for elevation_bucket in self.patches_dict[class_id].keys():
                patches_per_bucket[elevation_bucket] += len(self.patches_dict[class_id][elevation_bucket])

            print("\tBuckets distribution:")
            print("\tbucket:         " + ', '.join(map(lambda b: "%5s" % str(b), patches_per_bucket.keys())))
            print("\televation:      " + ', '.join(map(lambda b: "%5s" % str(b * self.elevation_bucket_size), patches_per_bucket.keys())))
            print("\tloaded patches: " + ', '.join(map(lambda p: "%5s" % str(p), patches_per_bucket.values())))

    def print_usage_statistics(self):

        print("Patches usage statistics:")

        for class_id in sorted(self.class_id_to_name):
            class_name = self.class_id_to_name[class_id]
            usages = 0
            elevation_usages = defaultdict(lambda: 0)

            for patch in filter(lambda p: p.class_id == class_id, self.patches_list):
                usages += patch.get_usages()

            print("-\t%s, used %i %s" % (class_name, usages, "time" if usages == 1 else "times"))

            if usages == 0:
                continue

            for elevation_bucket in self.patches_dict[class_id].keys():
                for patch in self.patches_dict[class_id][elevation_bucket]:
                    elevation_usages[elevation_bucket] += patch.get_usages()

            print("\tElevation buckets usages:")
            print("\tbucket:       " + ', '.join(map(lambda b: "%5s" % str(b), elevation_usages.keys())))
            print("\televation:    " + ', '.join(map(lambda b: "%5s" % str(b * self.elevation_bucket_size), elevation_usages.keys())))
            print("\tused patches: " + ', '.join(map(lambda p: "%5s" % str(p), elevation_usages.values())))

    def generate_dataset(self):
        i = 0
        augmented_image_filename = lambda n: path.join(self.augmented_images_path, '%07i_augmented_image.png' % i)

        # load background images
        self.load_background_images()

        # load patches
        self.load_patches()

        if len(self.patches_list) == 0:
            print("No patches found. Nothing to do.")
            sys.exit()

        self.print_patches_statistics()

        for t in range(self.num_augmented_images):

            if self.background_images_choice == 'random':
                # select random background image
                background_image = choice(self.background_images_list)
            elif self.background_images_choice == 'sequential':
                # select background image sequentially
                ni = self.num_augmented_images
                nb = len(self.background_images_list)
                background_image = self.background_images_list[int(nb*float(t)/ni)]
            else:
                # only use one specific background image
                if self.background_images_choice not in self.background_images_dict:
                    print("Background image not found:", self.background_images_choice, "  Available background images:")
                    print('\n'.join(map(lambda bi: bi.filename, self.background_images_list)))
                    sys.exit()

                background_image = self.background_images_dict[self.background_images_choice]

            augmented_image = AugmentedImage(background_image)

            # select random object classes
            num_patches = randint(1, self.max_masks_per_image)

            # for each object
            for _ in range(num_patches):
                # select patch given class, elevation
                class_id = choice(self.class_name_to_id.values())
                patches_by_elevation = self.patches_dict[class_id]

                # apply patch and add info to image
                augmented_image.apply_patch(patches_by_elevation)

            # export image and labels
            while path.exists(augmented_image_filename(i)):
                i += 1
            augmented_image.export(augmented_image_filename(i))

            num_objects = len(augmented_image.class_ids)
            print("Saved augmented image %s with %i %s (%s)" % (
                path.basename(augmented_image_filename(i)),
                num_objects,
                "object " if num_objects == 1 else "objects",
                ', '.join(map(lambda ci: self.class_id_to_name[ci],
                              sorted(augmented_image.class_ids)))))

        self.print_usage_statistics()

    def generate_yolo_files(self):
        config_filename = path.join(self.base_path, "config/yolo.yaml")
        if path.exists(config_filename):
            print("Yolo configuration file found. Generating yolo files from templates...")
        else:
            print("Yolo configuration file not found.")
            return

        with open(config_filename, 'r') as f:
            config = yaml.load(f)

        dataset_name = config['dataset_name']

        class_names_list = map(lambda i: self.class_id_to_name[i] if i in self.class_id_to_name else 'unused_%i' % i,
                               range(max(self.class_id_to_name.keys()) + 1))
        num_classes = len(class_names_list)
        num_filters = (num_classes + 5) * 3
        ttr = utils.template_tokens_replacer(num_classes, num_filters, dataset_name)

        yolo_files_path = path.expanduser(config['yolo_files_path'])

        # Filenames of yolo template files
        data_template_filename = path.expanduser(config['data_template'])
        cfg_testing_template_filename = path.expanduser(config['cfg_testing_template'])
        cfg_training_template_filename = path.expanduser(config['cfg_training_template'])

        # Filenames for yolo output files
        utils.mkdir(yolo_files_path)
        names_filename = path.join(yolo_files_path, '%s.names' % dataset_name)
        txt_names_filename = path.join(yolo_files_path, '%s_names.txt' % dataset_name)
        data_filename = path.join(yolo_files_path, path.basename(ttr(data_template_filename)))
        cfg_testing_filename = path.join(yolo_files_path, path.basename(ttr(cfg_testing_template_filename)))
        cfg_training_filename = path.join(yolo_files_path, path.basename(ttr(cfg_training_template_filename)))

        with open(names_filename, 'w') as f:
            f.write('\n'.join(class_names_list))

        with open(txt_names_filename, 'w') as f:
            f.write('\n'.join(class_names_list))

        utils.file_tokens_replacer(ttr, data_template_filename, output_filename=data_filename)
        utils.file_tokens_replacer(ttr, cfg_testing_template_filename, output_filename=cfg_testing_filename)
        utils.file_tokens_replacer(ttr, cfg_training_template_filename, output_filename=cfg_training_filename)

        print("Done.")


if __name__ == "__main__":
    DataAugmentation().generate_dataset()
    DataAugmentation().generate_yolo_files()
