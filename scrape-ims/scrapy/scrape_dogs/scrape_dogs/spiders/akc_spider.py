import scrapy
import pickle as pk
from scrape_dogs.items import AKCItem

urls = pk.load(open('/media/nate/Windows/github/IDmyDog/scrape-ims/scrapy/akc-urls.pk', 'rb'))
print(urls[0])
class AKCSpider(scrapy.Spider):
    name = "akc"
    allowed_domains = ["akc.org"]
    start_urls = [
        urls[0]
    ]

    def parse(self, response):
        for sel in response.xpath('//section/div/article'):
            item = AKCItem()
            item['breed'] = sel.xpath('div/h2/a/text()').extract()
            item['desc'] = sel.xpath('div/p/text()').extract()
            item['link'] = sel.xpath('div/h2/a/@href').extract()
            item['thumb'] = sel.xpath('div/a/img/@src').extract()
            yield item