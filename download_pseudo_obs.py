#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2020 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import re
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

url = 'https://zenodo.org/record/3892252'
base_url = 'https://zenodo.org/'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
elements = soup.find_all('a', class_='filename')

for element in elements:
    dowload_url = base_url + element['href']
    filename = re.findall('PSEUDOobs\_.*\.tar\.gz', dowload_url)[0]
    urllib.request.urlretrieve(
        dowload_url,
        filename=f'/net/h2o/climphys/lukbrunn/Data/InputData/PseudoOBS/EUCP/{filename}')
    time.sleep(5)
