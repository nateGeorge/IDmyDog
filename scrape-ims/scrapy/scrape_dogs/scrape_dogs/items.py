# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
    
class AKCItem(scrapy.Item):
    image_urls = scrapy.Field()
    images = scrapy.Field()
    breed = scrapy.Field()
    link = scrapy.Field()
    desc = scrapy.Field()
    thumb = scrapy.Field()
    
class WikiItem(scrapy.Item):
    image_urls = scrapy.Field()
    images = scrapy.Field()
    breed = scrapy.Field()
    link = scrapy.Field()
    desc = scrapy.Field()