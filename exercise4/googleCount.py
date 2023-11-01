import re
import datetime

import requests
from bs4 import BeautifulSoup
import argparse

REGEX = r'About (.*) results'

def number_of_search_results(key):
    def extract_results_stat(url):
        headers = { 
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/21.0'
        }
        search_results = requests.get(url)#, headers=headers, allow_redirects=True)
        soup = BeautifulSoup(search_results.text, features='lxml')
        result_stats = soup.find('div',{id:'result-stats'})
        m = re.match(REGEX, result_stats.text)
        # print m.group(1)
        return int(m.group(1).replace(',',''))

    google_main_url = 'https://www.google.co.in/search?q=' + key
    return extract_results_stat(google_main_url)



parser = argparse.ArgumentParser(description='Get Google Count.')
parser.add_argument('word', help='word to count')
args = parser.parse_args()

url = 'https://www.google.co.in/search?q=' + args.word#"https://www.tutorialspoint.com/index.htm"
headers = { 
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/21.0'
        }
cookies = {
    'CONSENT': 'PENDING+376'
}
#req = requests.get(url, headers=headers, allow_redirects=True, cookies=cookies)
#soup = BeautifulSoup(req.text, "html.parser")
#print(soup.text)
print(number_of_search_results(url))