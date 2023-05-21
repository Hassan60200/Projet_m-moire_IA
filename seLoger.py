import scrapy


class SelogerSpider(scrapy.Spider):
    start_urls = [
        'https://www.seloger.com/list.htm?projects=2,5&types=2,1&natures=1,2,4&places=[{%22divisions%22:[2238]},{%22inseeCodes%22:[600159]}]&mandatorycommodities=0&enterprise=0&qsVersion=1.0&m=homepage_buy-redirection-search_results']

    def parse(self, response):
        # Exemple : extraire le titre principal de la page
        title = response.xpath('//h1[@class="c-pa-top__title"]/text()').get()
        print(title)
