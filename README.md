# IDmyDog
This is a Python machine learning app that identifies breeds of dogs from pictures.  This is a final project for the Udacity Machine Learning Nanodegree.  I may try to improve this in the future by using a neural network instead of a random forest classifier.

## Environment setup
This project was developed and tested on Ubuntu 16.04 LTS.
From within Python, the output of:
```python
import sys
print sys.version
```
was:
2.7.12 (default, Jul  1 2016, 15:12:24)
[GCC 5.4.0 20160609]

### Install Python packages
This assumes you already have Python 2.7 and pip installed.

You will need the following Python packages installed to work through this project:
* Scrapy (for scraping akc.org for a list of breed names and pics)
* mahotas (for calculating Haralick textures)
* imutils (image operations convenience functions)
* seaborn (for making nifty plots)
* pandas (for data operations)
* scikit-learn (machine learning library)
* progressbar (for showing eta and % completion of tasks)

These can be installed with the 'install_py_modules.py' script.

### Install OpenCV 2.4.10
You will also need to install OpenCV.  If you are using an Anaconda distribution of Python, installing OpenCV is as easy as `conda install -c menpo opencv=2.4.11` in a terminal.

I have provided the install_cv2.sh file, which can be run in a Linux environent as `./install_cv2.sh` from a terminal.  This will install OpenCV 2.4.10, and has worked for me in Ubuntu 16.04 LTS.  To install on another OS, you will have to consult Google for now.

### Setting save directories
Change the values of 'pickle_dir' and 'image_dir' in the file 'config.json' to the appropriate directory on your system.

## Running the project
I ran most of my code from within a Python shell in a terminal.  I tested most of my Python scripts from a terminal, as `python2 script.py` since I have Python 2 and Python 3 both installed in my system, and `python script.py` runs Python 3.

If you don't want to do all the work of outlining the dogs, or other steps of the project, you can unzip the 'process_ims/pickles.tar.bz' file in the 'process_ims' directory, which contains all the pickled databases of bounding boxes, features, and the final classifier.

### Scraping images and breed names
#### Running Scrapy
I first used the script 'generate-start_urls.py' in the folder 'scrape_ims/scrapy' to generate the starting urls for Scrapy, so if the Scrapy spider is not working, you could check that those URLs are still valid.

To run the spider, type:
```bash
scrapy crawl akc -o output.json
```
in a terminal from the 'scrape_ims/scrapy/scrape_dogs' folder.

#### Scraping Google images
If you want to use the file 'scrape_ims/scrape_google_dogs.js', you will have to install node.js and npm, install the [google-images](https://github.com/vdemedes/google-images) package with npm, and [get a Google custom search engine (CSE)](https://github.com/vdemedes/google-images#set-up-google-custom-search-engine).  You can then set your Google CSE credentials in the 'scrape_ims/credentials.cred' file.  Warning: Google CSE only allows about 100 requests without signing up for a paid plan.  You can get a free trial to increase this number.  The google-images package won't tell you if the search is failing due to maxing out your daily quota, so if the search is failing, check your CSE quota for the day.  Also [see this issue](https://github.com/vdemedes/google-images/issues/12) I opened if you want google-images to report errors.

If you are cloning this and altering the Google custom search engine credentials, you may want to run
```bash
git update-index --assume-unchanged scrape_ims/credentials.cred
```
from the 'IDmyDog' home directory to make sure you're not uploading your api key and custom search id to GitHub.

You can then run `nodejs scrape_google_dogs.js` from the scrape_ims folder.

### Pre-processing the images
#### Overview
##### Preparing Images
* get_bounding_boxes_of_dogs.py -- enables drawing rectangles around dogs in images -- WARNING: this task takes a long time to complete
* get_bounding_boxes_of_dogs-even-out-population.py -- enables drawing rectangles around dogs in images, helps to balance out classes 
* clean_bbs.py -- removes bounding boxes from accidental clicks
* grabcut_ims.py -- uses OpenCV grabCut() to segment dog from background -- WARNING: this script takes hours to complete

##### Extracting Features
* get_haralick_n_chists.py -- extracts 13-dim Haralick texture and color histogram from fore- and background of images -- WARNING: this script takes hours to complete
* analyze_haralick_n_color_hists.py -- performs PCA on Haralick, does some analysis on PCA and color histograms
* extract_haralick_fg-multicore.py -- extracts full 13x4-dim Haralick textures from the foreground of images -- WARNING: this script takes hours to complete
* extract_chists_fg-linear.py -- extracts color histograms from the foreground of images

#### Draw rectangles around dogs
My first step was to go through images of dogs, outlining the dogs with a rectangle.  This is done by typing `python process_ims/get_bounding_boxes_of_dogs.py` in a terminal.

When I originally did this project, I noticed some breeds only had a few images after going through the images once.  I decided to try to get at least 6 images for each breed, so went back through the images with the 'process_ims/get_bounding_boxes_of_dogs-even-out-population.py' file, which does the same thing as 'process_ims/get_bounding_boxes_of_dogs.py', but also shows the bounding boxes that have been recorded for the current image.  

The script will start logging the rectangles you draw as bounding boxes for dogs' bodies.  The controls for the interface are:
-right arrow = next available pic
-left arrow = previous available pic
-'n' = random pic
-'d' = next dog breed
-'b' = log coords of drawn rectangle around dog bodies
-'f' = log coords of drawn rectangle around dog faces
-'r' = reset all bounding boxes
-'q' = quit program

The bounding boxes for each image will be saved in the file 'pDogs-bounding-boxes.pd.pk' in your pickle folder.

Finally, run the 'process_ims/clean_bbs.py' file to get rid of any tiny bounding boxes you may have accidentally created.  I accidentally created some with my laptop touchpad.

You can check the bounding boxes by running the file 'process_ims/check_bounding_boxes.py', which will draw the bounding boxes for bodies on the images, and then bounding boxes for heads, and can be advanced to the next image by pressing any key.

#### Grabcut dog foregrounds
The next step was to use the grabCut() function from OpenCV to remove the backgrounds from the images.  The script to run for this is 'process_ims/grabcut_ims.py'.  Warning: this took about 3 hours on my machine.

#### Extract Haralick textures and color histograms
Next, extract our features from the images.  I chose to extract Haralick texture and color histograms from the foreground of the images.  To do this step, run the 'process_ims/get_haralick_n_chists.py' file.  It will also display a plot comparing the foreground and background color histograms.  This script will take a long time, somewhere around 2 hours.

The Haralick features are calculated by breaking the forground up into small squares, calculating the Haralick feature on each square, and averaging them to arrive at one Haralick feature vector for the foreground.

Run 'process_ims/extract_haralick_fg-multicore.py' to extract the 52-dimension Haralick features of only the foregrounds.
Run 'process_ims/extract_chists_fg-linear.py' to extract the color histograms of only the foregrounds.
Run 'process_ims/chistPCA.py' to extract the 20-dim PCA of the color histograms of the foregrounds.

#### Analyze Haralick textures
Run 'process_ims/analyze_haralick_n_color_hists.py' to extract the 13-dimension Haralick features and color histograms of the foreground and background.

### Machine learning
#### Overview
* machine_learning.py -- goes through machine learning algos and analysis
* check_robust.py -- checks performance on unseen data
* check_peturb.py -- checks reaction of model to noise in training and test data

#### Train and test the classifiers
Run 'process_ims/machine_learning.py' to see the performance of machine learning classifiers on the training data using different kinds of features.

#### Check robustness of model
Run 'process_ims/check_robust.py' and 'process_ims/check_peturb.py' to check the model's performance on unseen data and it's sensitivity to noise.

### Other
In the 'process_ims/other' folder:
* check_bounding_boxes.py -- draws bounding boxes on the images
* random_guess_benchmark.py -- checks validitiy of the random guess benchmark
* test_rect_grid.py draws rectangle grid on foreground and background of images
* 2d_haralick_map.py -- draws 2d Haralick variation map on image foregrounds