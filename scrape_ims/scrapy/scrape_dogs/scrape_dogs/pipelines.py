# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

# also http://stackoverflow.com/questions/18081997/scrapy-customize-image-pipeline-with-renaming-defualt-image-name
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
import scrapy
import os
import shutil
from scrapy.utils.project import get_project_settings

class ScrapeDogsPipeline(object):
    def process_item(self, item, spider):
        return item

class DogImagePipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        #print(item)
        for image_url in item['image_urls']:
            print(image_url)
            #item['files']['path'] = item['breed']
            yield scrapy.Request(image_url)
    
    # from: http://stackoverflow.com/questions/28007995/how-to-download-scrapy-images-in-a-dyanmic-folder-based-on
    def item_completed(self, results, item, info):
        for result in [x for ok, x in results if ok]:
            path = result['path']
            folder = item['breed'][0]
            filename = path.split('/')[1]

            # http://doc.scrapy.org/en/latest/topics/practices.html#run-scrapy-from-a-script
            settings = get_project_settings()
            storage = settings.get('IMAGES_STORE')

            target_path = os.path.join(storage, folder, os.path.basename(path))
            path = os.path.join(storage, path)
            
            # If path doesn't exist, it will be created
            if not os.path.exists(os.path.join(storage, folder)):
                os.makedirs(os.path.join(storage, folder))
            
            shutil.move(path, target_path)
            print('saving to:' + str(target_path))

        if self.IMAGES_RESULT_FIELD in item.fields:
            item[self.IMAGES_RESULT_FIELD] = [x for ok, x in results if ok]
        return item