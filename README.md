# Data Augmentation


## Example images

<p align="right">
  <img src="/doc/img/augmented-image-0000167-erl_mk.png?raw=true" width="350" title="augmented image with ERL-MK objects">
  <img src="/doc/img/augmented-image-0004143-erl_lis.png?raw=true" width="350" alt="augmented image with ERL-Lisbon objects">
</p>
<p align="right">
 Augmented images
</p>

<p align="right">
  <img src="/doc/img/polygonal-label-blue.png?raw=true" width="350" title="image with label">
</p>
<p align="right">
 Image with label
</p>

<p align="right">
  <img src="/doc/img/support-surface-rgb.png?raw=true" width="350" title="support surface rgb">
  <img src="/doc/img/support-surface-depth.png?raw=true" width="350" title="support surface depthmap">
</p>
<p align="right">
 Support surface images obtained by the support surface polygons (left: RGB, right: depth map)
</p>

## Augmentation pipeline

<p align="right">
  <img src="/doc/img/graph-data-augmentation.png?raw=true" width="350" title="augmentation pipeline">
</p>
<p align="right">
 Augmentation pipeline
</p>


These data augmentation tools allow to automatically generate a dataset of objects by cut-and-past.
The objects (patches) are "cut" from source images and pasted on background images, in specific positions.


In order to create a dataset it is necessary to:
- Define the support surface polygons, that describe the 3D position of the tables, counters, etc, where the objects could be found at detect-time.
- Acquire the source images from which the objects patches are extracted, including information about their position relative to the camera.
- Label the source images.
- Configure and run the data_augmentation tool to extract the patches and generate the dataset. The dataset includes the label information needed to train the detector.

Afterwards it is possible to generate the DarkNet YOLO training configuration files (needed to define the dataset and the network architecture).
Mask-RCNN-type labels and configuration files are not supported yet.



## Data collection

Two kinds of data need to be collected: the support surface, also called background images, and the source images.
For both it is only required to define some 3D information and run the respective tools.
It is highly suggested to record a ROS bag while collecting the data, so that the output data can be extracted again.


### Acquiring the support surface images

The support surfaces are defined by 3D polygons in map frame.
These polygons are manually defined for the specific environment for the moment (they could be estimated automatically in the future).
The polygons are defined in ``` `rospack find mbot_world_model`/maps/$ROBOT_ENV/support_surface_polygons.yaml```.
Practically it is more useful to define polygons in different files (support_surface_polygons_livingroom.yaml, support_surface_polygons_bedroom.yaml, etc), so that only the polygons in the same room are in the same file.
The points of each polygon should have the same height (z coordinate), since the support surfaces are supposed to be parallel to the ground floor, that is the z=0 plane.

Once defined the polygons, set the desired polygon config file in ``` `rospack find support_surface_saver`/ros/launch/support_surface_saver.launch``` and check the support_surface_saver with the command
```bash
roslaunch support_surface_saver support_surface_saver.launch save_images:=false set_use_sim_time:=false
```
support_surface_saver requires the python Pillow, that can be installed with `pip install --target=$HOME/.local/lib/python2.7/site-packages/ --upgrade Pillow`.

In rviz it is possible to check the support surfaces by visualising the topic `/support_surface/depth/image` with the DepthCloud plugin (showing them as point cloud).

<!-- TODO image: DepthCloud plugin -->

Make sure the robot is localised and collect the support surface data by running the command
```bash
roslaunch mbot_benchmarking record_dataset.launch dataset_name:=test_support_surfaces
```
These bags allow to extract the data again in case of problems.
While recording, monitor the available disk space with the command `watch_df`.
These bags contain all sensors (e.g., images) of the robot, so they rapidly grow in size.

It is important to separate the support surface polygons in different files, because the support_surface_saver tool computes the support surfaces regardless of occlusions from the environment.
So for example, support surfaces from another room would show up through a wall, and this is not desirable.

Once recorded the bag, play them and run the support surface saver specifying `save_images:=true` and `set_use_sim_time:=true` to save the support surface data.
```bash
rosbag play basic_sensors_2018-12-13-21-16-51.bag cameras_2018-12-13-21-16-51.bag --pause  --clock -r 0.5 # to play the bag 
roslaunch support_surface_saver support_surface_saver.launch save_images:=true set_use_sim_time:=true
```

The output of `support_surface_saver` are rgb image, depth images and pickled objects.
The image files are only saved to visualise and visually select which files to keep (e.g., discard images with only a small area occupied by support surfaces), but are not used by other data augmentation tools.
The pickled objects contain both the rgb images, the depth images, and additional information.
These are the files read from the data_augmentation tool.


### Acquiring the objects source images


The object patches, that are pasted on background images, are extracted from the source images.
The source images are images of the objects, with associated labels and 3D information about the position of the camera with respect to the objects.
The image_saver tool saves both images and 3D information given a list of reference points in map frame, corresponding to each object's pose (defined as the point where the object sits, i.e., the lowest point on z that is part of the object).

In order to collect images of many objects at the same time, it is very useful to record a bag with images, localisation and every other required topic, from which the source images and information can be extracted for every captured object at once.

To record such bags, make sure the robot is well localised and run the command
```bash
roslaunch mbot_benchmarking record_dataset.launch dataset_name:=test_1
```
Monitor the available disk space with the command `watch_df`.

For each dataset, save a yaml file named as the ROS bag for the dataset, with the reference points for each object.
```yaml
class_name_1:
  point:
    x: 1.0
    y: 2.0
    z: 3.0
class_name_2:
  point:
    x: 1.0
    y: 2.0
    z: 3.0
...
```
This is easily done by using the camera plugin in rviz and publishing a point stamped on the image where the depth map is available.

As an example, let's say the bags recorded with `mbot_bencmarking` are renamed to `basic_sensors_test_1.bag, cameras_test_1.bag`, where `test_1` is the name of the bag dataset.
The `reference_points` file can be renamed to `objects_reference_points_test_1.yaml` with the same format so that it is recognised automatically in the `image_saver` launch file.

 <!-- TODO  image: publishing point with rviz -->

Set the path of this file in the rosparam tag and set the save_reference_points param to true in ``` `rospack find image_saver`/ros/launch/image_saver.launch```.
Also update the param `bags_path` if needed.

To save the source images launch the image saver with the following parameters:

```bash
roslaunch image_saver image_saver.launch image_folder_output:=~/source_images play_bags:=true dataset_bag_name:=test_1
```
Set `image_folder_output` to the actual folder where source images should be saved, `dataset_bag_name` and `dataset_path` to the same values used for `record_dataset.launch`.
In this example, `dataset_bag_name:=test_1`.

`image_saver` requires the imageio python library.
To install the library use the command: `sudo pip install imageio`.


## Labeling the source images

After saving the source images, each image must be labeled with mbot_labeling_tool.
To label the images for a set of specific objects, say 'water, coke', set the `dataset_path` and `class_names` variables in ``` `rospack find mbot_labeling_tool`/config.yaml```.
E.g., 
```yaml
dataset_path: ~/ds/yolo/lis_v1/source_images
class_names: [
  coke,
  water
]
```

To open the labeling tool roscd to `mbot_labeling_tool` and execute the python executable with the following command then open the address `http://127.0.0.1:5000/` in a browser.

```bash
python flask_app.py
```
The labeling tool will create a json file for each (and only) labeled source image.

 <!-- TODO  image: mbot_labeling_tool -->

## Generating augmented dataset

Once the support surface images and the labeled source images are ready, the data augmentation tool can be executed to generate the augmented images.


### Generating augmented images and labels

Before running the data augmentation tool, set the parameter class_id_dict and the path to the dataset's folders for source, background and augmented images in `./config/data_augmentation.yaml` (this configuration file is relative to the folder where the tool is executed).
The default configuration file is in ``` `rospack find data_augmentation`/config/data_augmentation.yaml```.
The augmented images folder will be made if it does not exist.
The documentation of each parameter is contained in the configuration file itself as comments.


### Generating training and testing configuration

In order to automatically generate the YOLO training and testing configuration files, the configuration for YOLO must be present in `./config/yolo.yaml` (this configuration file is relative to the folder where the tool is executed).
The default configuration file is in ``` `rospack find data_augmentation`/config/yolo.yaml```.
The documentation of each parameter is contained in the configuration file itself as comments.

The YOLO configuration files (.names, .cfg and .data) are generated by substituting the `€1, €2 and €3` tokens in each template file and filename with the following values:
```
€1 with number of classes,
€2 with number of filters,
€3 with the dataset name
```
Example templates can be found in ``` `rospack find data_augmentation`/example_yolo_templates/``` and can be used without modifying the content (except for €3.data, where `SOME_DATASET_FOLDER` should be replaced with the correct path).
The file `dataset_name.names` does not need a template to be generated, being a simple list of class names.


## Training with yolo

To train DarkNet YOLO, follow the instructions at 
https://github.com/AlexeyAB/darknet/ and use the configuration files generated by the data_augmentation tool.
