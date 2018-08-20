# Autodrive Traffic Sign Detection
## Team 2: 
- Rodrigo Gomez-Palacio
- Adolfo Portilla
- Juan Vasquez
- Muhammad Ashfaq

### MAIN: https://github.com/rgomezp/autodrive-tsdr/blob/master/stop_signs/traffic-sign-detection/src/apps/main.cpp

![Alt text](screencapture.png?raw=true "Title")

# 1st Place Winners in the Stop Sign Challenge Category at the 2018 Autodrive Challenge at GM Proving Grounds in Yuma AZ! (2nd overall)
### Competition News:
- https://today.tamu.edu/tag/autodrive/
- https://tees.tamu.edu/news/2018/04/25/texas-am-autodrive-challenge-team-shares-experience-before-showcase/
- https://tees.tamu.edu/news/2018/05/18/texas-am-engineering-autodrive-challenge-team-takes-second-overall-in-first-competition-milestone/
- https://tti.tamu.edu/news/ttis-talebpour-texas-am-student-team-place-second-in-autodrive-competition/

" The stop sign challenge required the car to maneuver on a straight three-lane road and successfully detect and stop for a series of stop signs, a challenge in which the team reached a top speed of 20 mph before stopping, the fastest in the competition. "


# Sign Detection Pipeline
main.cpp can be found in **traffic-sign-detection/src/apps/main.cpp**

## RELEVANT VARS
##### `filter_count`: 
0 = red, 1 = yellow, 2 = white
##### `input_image`: 
original image
##### `stopDetected`: 
holds boolean value (is a stop sign, is not a stop sign) to be passed to ROS
##### `stopDistance`: 
distance derived from contour width in pixels to be passed to ROS
##### `stopPercentage`: 
scale from -1 (not a stop sign) to 1 (definitely a stop sign)
##### `filter_vec`: 
the object returned from the sign_filter wrapped to be accepted by the `findContours( )` function
##### `temp_contours`: 
a double vector of points to store the detected contours after the red filter is applied
##### `contours`: 
stores the polygon point vectors after processing from the `approxPolyDP( )` function
##### `poly`: 
point vector that saves the individual points that make a single polygon
##### `bounding_boxes`: 
stores the individual `box` objects that frame the polygon at its x and y extremities 

## FUNCTIONS
##### `sign_filter(int, void*)`: 
function that applies the red color filter itself to the Mat image. The Mat objects `lower_red` and `upper_red` are used to provide two ranges used in the filter object. `red_image` is the output of the `addWeighted( )` function (the filter object). This object is then passed in to the `filter_image( )` function along with the `output_image` variable which is the Mat object where the output is ultimately stored. The entire function returns this `output_image` Mat object.

##### `findContours(filter_vec[i], temp_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`: 
OpenCV function that finds contours in the provided Mat object (`filter_vec`) and stores the contours in a double vector of type Point

##### `approxPolyDP(temp_contours[j], poly, 1, true)`: 
OpenCV function that attempts to find polygons in the passed in contours object (double vec of Points). Each of these contours are converted to polygons one by one and stored in the `poly` variable (vector of type Point) and pushed onto the `contours` object.

##### `is_symmetric(box) && !is_small(box) && !is_large(box)`:
functions that return boolean values to eliminate many false positives that are most likely NOT stop signs

## NEURAL NET IMPLEMENTATION
For every bounding box not eliminated by the functions `is_symmetric`, `!is_small` and `!is_large`, a neural net is spawned to look at the individual image. To achieve this, we first create a new Mat object that stores a `crop` of the individual bounding box. The image is then resized to a 32 x 32 pixel image. Next, the image is further processed using a histogram equalization using the OpenCV `cvtColor()` function with an imported image `base.jpg`. The resulting file is better suited to processing by the neural network. This file is saved to the __bucket__.

##### bucket folder
The bucket folder contains two crucial files:
- the image to be classified
- the text file with the classification

The image file is written to the bucket and a new thread is spawned that launches the Neural2D net. After termination, the bucket contains the classification of the file as a simple "true" or "false" string followed by the classification percentage.

##### true classification
If the image is classified as a stop sign, the truthiness, percentage, and distance data is saved to the ROS message variables to tell car to stop. These variables are then published to ROS and the next frame is processed.




