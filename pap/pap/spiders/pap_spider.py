import scrapy
from ..items import PapItem
import time

class PapSpider(scrapy.Spider):
    name = "pap"

    def start_requests(self):
        time.sleep(10)  # Délai de 2 secondes
        url = "https://www.leboncoin.fr/recherche?category=8&locations=60200__49.430772867574944_2.8292159576903857_6520"
        yield scrapy.Request(url=url, callback=self.parse_annonces)

    def parse_annonces(self, response):
        listAnnonces = response.css('.col-1-3')
        prices = []

        for annonce in listAnnonces:
            price = annonce.css('div.sc-71b093fa-1 kFVNAd div.bigpicture-housing-content div.src__Box-sc-10d053g-0 kcxqXM div.src__Box-sc-10d053g-0 jISaOx div.src__Box-sc-10d053g-0 cInbdR p.sc-77916451-0 hkMhqr span.aditem_price span::text').get()
            prices.append(price)

        item = PapItem()
        item['price'] = prices

        yield item
