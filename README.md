# IDmyDog
This is a Python machine learning app that identifies breeds of dogs from pictures.  This is a final project for the Udacity Machine Learning Nanodegree.  I may try to improve this in the future by using a neural network instead of a random forest classifier.

## Environment setup
This project was developed and tested on Ubuntu 16.04 LTS.

### Install Python packages
This assumes you already have Python 2.7 and pip installed.

You will need the following packages installed to work through this project:
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
I ran all my Python scripts from a terminal, as `python2 script.py` since I have Python 2 and Python 3 both installed in my system, and `python script.py` runs Python 3.

If you don't want to re-run the image scraping, you can unzip the 'scrape-ims/images.zip' file in the 'scrape-ims' directory.  If you don't want to do all the work of outlining the dogs, or other steps of the project, you can unzip the 'process_ims/pickles.zip' file in the 'process_ims' directory, which contains all the pickled databases of bounding boxes, features, and the final classifier.

### Scraping images and breed names
#### Running Scrapy
I first used the script 'generate-start_urls.py' in the folder 'scrape-ims/scrapy' to generate the starting urls for Scrapy, so if the Scrapy spider is not working, you could check that those URLs are still valid.
If you want to re-run the Scrapy spider and download images, you will need to change the following line to a real directory on your computer:
```python
IMAGES_STORE = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
```
in the file 'scrape-ims/scrapy/scrape_dogs/scrape_dogs/settings.py'.

Then you should be able to run
```bash
scrapy crawl akc -o output.json
```
in a terminal from the 'scrap-ims/scrapy/scrape_dogs' folder.

#### Scraping Google images
If you want to use the file 'scrape-ims/scrape_google_dogs.js', you will have to install node.js and npm, install the [google-images](https://github.com/vdemedes/google-images) package with npm, and [get a Google custom search engine (CSE)](https://github.com/vdemedes/google-images#set-up-google-custom-search-engine).  You can then set your Google CSE credentials in the 'scrape-ims/credentials.cred' file.  

If you are cloning this and altering the Google custom search engine credentials, you may want to run
```bash
git update-index --assume-unchanged scrape-ims/credentials.cred
```
from the 'IDmyDog' home directory to make sure you're not uploading your api key and custom search id to GitHub.

You will want to change the line
```javascript
var dirs = getDirectories('/media/nate/Windows/github/IDmyDog/scrape-ims/images')
```
to the directory where you stored your Scrapy images.

You can then run `nodejs scrape_google_dogs.js` from the scrape-ims folder.

### Pre-processing the images

#### Draw rectangles around dogs
My first step was to go through images of dogs, outlining the dogs with a rectangle.  This is done by typing `python process-ims/get_bounding_boxes_of_dogs.py'.

When I originally did this project, I noticed some breeds only had a few images after going through the images once.  I decided to try to get at least 6 images for each breed, so went back through the images with the 'process-ims/get_bounding_boxes_of_dogs-even-out-population.py' file, which does the same thing as 'process-ims/get_bounding_boxes_of_dogs.py', but also shows the bounding boxes that have been recorded for the current image.  

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

Finally, run the 'process-ims/clean_bbs.py' file to get rid of any tiny bounding boxes you may have accidentally created.  I accidentally created some with my laptop touchpad.

You can check the bounding boxes by running the file 'process-ims/check_bounding_boxes.py', which will draw the bounding boxes for bodies on the images, and then bounding boxes for heads, and can be advanced to the next image by pressing any key.

#### Grabcut dog foregrounds
The next step was to use the grabCut() function from OpenCV to remove the backgrounds from the images.  The script to run for this is 'process-ims/grabcut_ims.py'.  Warning: this took about 3 hours on my machine.

#### Extract Haralick textures and color histograms
Next, extract our features from the images.  I chose to extract Haralick texture and color histograms.  To do this step, run the 'process-ims/get_haralick_n_chists.py' file.  It will also display a plot comparing the foreground and background color histograms.

Haralick features are calculated by breaking the forground up into small squares, calculating the Haralick feature on each square, and averaging them to arrive at one Haralick feature vector for the foreground.

#### Analyze Haralick textures


### Machine learning
#### Train and test the classifiers
