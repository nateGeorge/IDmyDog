import scrapy
import os
import shutil
import logging
import pickle as pk
from scrape_dogs.items import AKCItem

urls = pk.load(open('/media/nate/Windows/github/IDmyDog/scrape-ims/scrapy/akc-urls.pk', 'rb'))

class AKCSpider(scrapy.Spider):
    name = "akc"
    allowed_domains = ["akc.org"]
    start_urls = urls

    def parse(self, response):
        for sel in response.xpath('//section/div/article'):
            item = AKCItem()
            item['breed'] = sel.xpath('div/h2/a/text()').extract()
            item['desc'] = sel.xpath('div/p/text()').extract()
            item['link'] = sel.xpath('div/h2/a/@href').extract()
            url = sel.xpath('div/a/img/@src').extract()
            if len(url) == 1:
                #self.logger.warning('url:' + str(url[0][2:]))
                item['image_urls'] = ['http://' + url[0][2:]]
                #item['files'] = item['breed']
            isempty = True
            for k, v in item.items():
                if v == []:
                    continue
                else:
                    yield item