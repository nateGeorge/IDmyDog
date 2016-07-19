import scrapy
import os
import shutil
import logging
import pickle as pk
from scrape_dogs.items import AKCItem

urls = ['https://en.wikipedia.org/wiki/List_of_dog_breeds']

class WikiSpider(scrapy.Spider):
    name = "wiki"
    allowed_domains = ["wikipedia.org"]
    start_urls = urls

    def parse(self, response):
        for sel in response.xpath('//div/div/div/table/tr/'):
            item = AKCItem()
            item['breed'] = sel.xpath('td[1]/a/text()').extract()
            item['link'] = sel.xpath('td[1]/a/@href').extract()
            url = sel.xpath('td[10]/a/@href').extract()
            if len(url) == 1:
                #self.logger.warning('url:' + str(url[0][2:]))
                item['image_urls'] = ['http://wikipedia.org/' + url[0][2:]]
                #item['files'] = item['breed']
            isempty = True
            for k, v in item.items():
                if v == []:
                    continue
                else:
                    yield item