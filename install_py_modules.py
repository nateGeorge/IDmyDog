# this will install most necessary packages for this project
# that you may not already have on your system

import pip

def install(package):
    pip.main(['install', package])

# Example
if __name__ == '__main__':
    # for scraping akc.org for a list of breed names and pics
    install('Scrapy')
    # for calculating Haralick textures
    install('mahotas')
    # image operations convenience functions
    install('imutils')
    # plotting package
    install('seaborn')
    # data operations
    install('pandas')
    # machine learning lib
    install('scikit-learn')
    # image processing
    install('scikit-image')
    # eta and % completion of tasks
    install('progressbar')