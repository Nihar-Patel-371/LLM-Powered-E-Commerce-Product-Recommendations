import os
import re
import math
import json
import requests
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.options import Options


def get_catalog_urls():

    ########################################################################

    # MENS SHOES
    # Get the first page
    url = "https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page=1&sort=RELEVANCE&pageId=/men/shoes/view-all&page-size=36&categoryId=men_shoes&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    # Get total number of pages
    page_response  = response.json()
    total_number_of_pages = page_response['pagination']['totalPages']

    # Get all urls for the catalog
    all_catalog_urls = []
    for every_page in range(1,total_number_of_pages+1):
        url = f"https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page={every_page}&sort=RELEVANCE&pageId=/men/shoes/view-all&page-size=36&categoryId=men_shoes&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
        all_catalog_urls.append(url)

    ########################################################################

    # WOMEN SHOES
    # Get the first page
    url = "https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page=1&sort=RELEVANCE&pageId=/ladies/shoes/view-all&page-size=36&categoryId=ladies_shoes&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    # Get total number of pages
    page_response  = response.json()
    total_number_of_pages = page_response['pagination']['totalPages']

    # Get all urls for the catalog
    for every_page in range(1,total_number_of_pages+1):
        url = f"https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page={every_page}&sort=RELEVANCE&pageId=/ladies/shoes/view-all&page-size=36&categoryId=ladies_shoes&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
        all_catalog_urls.append(url)

    #########################################################################
    
    # MEN PRODUCTS
    # Get the first page
    url = "https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page=1&sort=RELEVANCE&pageId=/men/shop-by-product/view-all&page-size=36&categoryId=men_viewall&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    # Get total number of pages
    page_response  = response.json()
    total_number_of_pages = page_response['pagination']['totalPages']

    # Get all urls for the catalog
    for every_page in range(1,total_number_of_pages+1):
        url = f"https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page={every_page}&sort=RELEVANCE&pageId=/men/shop-by-product/view-all&page-size=36&categoryId=men_viewall&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
        all_catalog_urls.append(url)

    #########################################################################
    
    # WOMEN PRODUCTS
    # Get the first page
    url = "https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page=1&sort=RELEVANCE&pageId=/ladies/shop-by-product/view-all&page-size=36&categoryId=ladies_all&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    # Get total number of pages
    page_response  = response.json()
    total_number_of_pages = page_response['pagination']['totalPages']

    # Get all urls for the catalog
    for every_page in range(1,total_number_of_pages+1):
        url = f"https://api.hm.com/search-services/v1/en_US/listing/resultpage?pageSource=PLP&page={every_page}&sort=RELEVANCE&pageId=/ladies/shop-by-product/view-all&page-size=36&categoryId=ladies_all&filters=sale:false||oldSale:false&touchPoint=DESKTOP&skipStockCheck=false"
        all_catalog_urls.append(url)

    #########################################################################
    
    print(f'Total number of catalog pages to search: {len(all_catalog_urls)}')
    return all_catalog_urls

def get_data_from_catalogs(url_list):
    payload = {}
    headers = {}
    product_list = []
    for url in tqdm(url_list, total=len(url_list), leave=False):
        response = requests.request("GET", url, headers=headers, data=payload)
        page_respose = response.json()
        product_reponse = page_respose['plpList']['productList']
        product_list.extend(product_reponse)
    print(f'Obtained {len(product_list)} product from the catalogs.')
    return product_list

class productInfo():
    def __init__(self, product_dict):
        self.product_dict = product_dict
    
    def getInfo(self):
        try:
            productId = self.product_dict['id']
        except:
            productId = None
        try:
            productName = self.product_dict['productName']
        except:
            productName = None
        try:
            brandName = self.product_dict['brandName']
        except:
            brandName = None
        try:
            url = self.product_dict['url']
        except:
            url = None
        try:
            prices = self.product_dict['prices'][0]['price']
        except:
            prices = None
        try:
            stockState = self.product_dict['availability']['stockState']
        except:
            stockState = None
        try:
            comingSoon = self.product_dict['availability']['comingSoon']
        except:
            comingSoon = None
        try:
            colorName = self.product_dict['colorName']
        except:
            colorName = None
        try:
            isOnline = self.product_dict['isOnline']
        except:
            isOnline = None
        try:
            colors = self.product_dict['colors']
        except:
            colors = None
        try:
            colorShades = self.product_dict['colourShades']
        except:
            colorShades = None
        try:
            newArrival = self.product_dict['newArrival']
        except:
            newArrival = None
        try:
            mainCatCode = self.product_dict['mainCatCode']
        except:
            mainCatCode = None

        return str(productId), productName, brandName, url, prices, stockState, comingSoon, colorName, isOnline, colors, colorShades, newArrival, mainCatCode

def filter_extracted_data(product_list):
    all_product_info = []
    for product in product_list:
        each_product = productInfo(product)
        each_product_info = each_product.getInfo()
        all_product_info.append(each_product_info)
    print(f'\nExtracted necassary Information from {len(all_product_info)} from websites.')
    return pd.DataFrame(all_product_info, 
                        columns=['productId', 'productName', 'brandName', 'url', 'price', 'stockState', 'comingSoon', 'colorName', 'isOnline'
                                 , 'colors', 'colorShades', 'newArrival', 'mainCatCode'])

# Detials and Tech ***********************************************************************************************
def get_details_block(driver):
    # Find Button form the page
    try:
        detials_block = driver.find_element(By.ID, 'section-descriptionAccordion')
        # Make the block vissible
        driver.execute_script("arguments[0].style.display = 'block';\
                        arguments[0].style.height='auto';\
                        arguments[0].style.opacity = '1';", detials_block)
        # Again extract data from the block
        detials_text = driver.find_element(By.ID, 'section-descriptionAccordion').text
    except:
        detials_text = None
    return detials_text

def get_materials_block(driver):
    # Find Button form the page
    try:
        materials_block = driver.find_element(By.ID, 'section-materialsAndSuppliersAccordion')
        # Make the block vissible
        driver.execute_script("arguments[0].style.display = 'block';\
                        arguments[0].style.height='auto';\
                        arguments[0].style.opacity = '1';", materials_block)
        # Again extract data from the block
        materials_block = driver.find_elements(By.CSS_SELECTOR, '#section-materialsAndSuppliersAccordion div div div')
        materials_text = " , ".join([every_div.text for every_div in materials_block])
    except:
        materials_text = None
    return materials_text

def extract_info_product(url_list):
    count = 0
    completed_url_list = []
    details_text_list = []
    materials_text_list = []
    # Start driver
    driver = webdriver.Chrome()
    for url in tqdm(url_list, total=len(url_list), leave=False):
        try:
            # Load Product Page
            driver.get(url)
            # Get Details and Materials
            details_text = get_details_block(driver)
            materials_text = get_materials_block(driver)
            details_text_list.append(details_text)
            materials_text_list.append(materials_text)
            completed_url_list.append(url)
        except:
            print(f'Error due to: {url}')
            break
        count += 1
    driver.quit()
    return count, completed_url_list, details_text_list, materials_text_list
# Detials and Tech ***********************************************************************************************


if __name__=="__main__":

    # DB_RESET and EXIST *****************************************************************************************
    print("\n------------------------- DB STATE -------------------------\n")
    DB_RESET = False
    if os.path.exists('handm.pkl'):
        if DB_RESET:
            os.remove('handm.pkl')
            DB_EXIST = False
            print('DB RESET Completed')
        else:
            DB_EXIST = True
            print('DB EXIST, and will persist')
    else:
        DB_EXIST = False
        print('DB is not here')
    # DB_RESET and EXIST *****************************************************************************************

    # Current Website Catalog ************************************************************************************
    # Get necassary product info from the catalogs
    print("\n----------------------- SCRAPE DATA -----------------------\n")
    all_catalog_urls = get_catalog_urls()
    all_product_all_data = get_data_from_catalogs(all_catalog_urls)
    product_data_df = filter_extracted_data(all_product_all_data)
    # Current Website Catalog ************************************************************************************

    # Filter new search product **********************************************************************************
    # Filter out the urls of the products which we already have
    if DB_EXIST:
        existing_productsId = list(pd.read_pickle('handm.pkl')['productId']) # .read_csv('handm.csv')['productId']
        existing_productsId = list(map(int, existing_productsId))
        print(f'Existing number of Products in the DB: {len(existing_productsId)}')
        productsId_to_search = []
        for id in list(product_data_df['productId']):
            if int(id) not in existing_productsId:
                productsId_to_search.append(id)
    else:
        productsId_to_search = product_data_df['productId']
    
    print(f'\nNumber of new products to search: {len(productsId_to_search)}')
    search_product_data_df = product_data_df[product_data_df['productId'].isin(productsId_to_search)]
    # Filter new search product **********************************************************************************

    # Get Filtered new search product ****************************************************************************
    # Add detials and tech of the product
    full_url_list = 'https://www2.hm.com' + search_product_data_df['url']
    # Search through all the product for their detials and materials
    number_of_new_records, completed_url_list, detials_product_list, materials_product_list = extract_info_product(full_url_list)
    # Add into the columns
    search_product_data_df = search_product_data_df[:number_of_new_records]
    search_product_data_df['url'] = completed_url_list
    search_product_data_df['details'] = detials_product_list
    search_product_data_df['materials'] = materials_product_list
    # Get Filtered new search product ****************************************************************************

    # Update DB **************************************************************************************************
    print("\n----------------------- DB UPDATE -----------------------\n")
    if DB_EXIST:
        existing_products_df = pd.read_pickle('handm.pkl') # read_csv('handm.csv')
        new_product_df = pd.concat([existing_products_df, search_product_data_df], ignore_index=True)
        new_product_df.to_pickle('handm.pkl') # to_csv('handm.csv', index=False)
        print(f'DB Updated! DB has Products: {new_product_df.shape[0]}')
    else:
        search_product_data_df.to_pickle('handm.pkl') # to_csv('handm.csv', index=False)
        print(f'DB Created! DB has Products: {search_product_data_df.shape[0]}')
    print("\n----------------------- DB UPDATE -----------------------\n")
    # Update DB **************************************************************************************************