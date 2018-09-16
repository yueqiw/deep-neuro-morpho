import requests
import re
import os
import urllib
import urllib3
from bs4 import BeautifulSoup
import json
import time 
import logging 
import re
import argparse


class NeuroMorphoDownload():
    def __init__(self, max_id, data_dir='../data_download/test/', logging_level=logging.INFO):
        self.website = 'http://neuromorpho.org/'
        self.data_dir = data_dir 
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.morphometry = dict() 
        self.metadata = dict() 
        self.cell_ids = None
        self.max_id = max_id
        logging.basicConfig(format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                            level=logging_level, 
                            handlers=[logging.FileHandler(os.path.join(self.data_dir, 'download.log')),
                                        logging.StreamHandler()])
    
    def set_cell_ids(self, cell_ids=None):
        if cell_ids is None:
            self.cell_ids = sorted(list(self.metadata.keys())) 
        else:
            self.cell_ids = cell_ids 

    def save_json(self, data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logging.info("Saved data to file {}".format(filepath))

    def download_metadata(self, filename='metadata.json', cids=None):
        logging.info("Downloading metadata from the website.")
        if cids is None:
            cids = range(1, self.max_id + 1)

        for i in cids:
            if i % 1000 == 0:
                logging.info(i)
            url = "http://neuromorpho.org/api/neuron/id/{}.json".format(i)
            try:
                req = requests.get(url)
            except:
                try:
                    time.sleep(2)  # try again after 2 seconds
                    req = requests.get(url)
                except:
                    logging.warning('{}: failed request'.format(i))
                    continue  # skip this neuron
            if req.status_code != 200:
                logging.warning('{}: bad request status'.format(i))
                continue  # skip this neuron
            req_json = req.json()
            if 'error' in req_json:
                continue # skip this neuron
            self.metadata[i] = req_json
        logging.info("Finished downloading metadata from {}/{} cells.".format(len(self.metadata), self.max_id))
        self.set_cell_ids()
        self.save_json(self.metadata, os.path.join(self.data_dir, filename))

    def read_metadata(self, filename='metadata.json'):
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            self.metadata = json.load(f) 
            self.metadata = {int(k): v for k, v in self.metadata.items()}
        self.set_cell_ids()

    def download_png_from_metadata(self, folder='web_png', cids=None, overwrite=False):
        logging.info("Dowloading PNG images (url from metadata) from the website.")
        png_dir = os.path.join(self.data_dir, folder)
        if not os.path.exists(png_dir):
            os.mkdir(png_dir)
        if cids is None:
            cids = self.cell_ids
        download_success = dict()
        for i in cids:
            if i % 1000 == 0:
                logging.info(i)
            url = self.metadata[i]['png_url']
            img_path =  os.path.join(png_dir, "{}.png".format(i))
            download_success[i] = True
            if (not overwrite) and os.path.exists(img_path):
                continue # skip this neuron
            try:
                urllib.request.urlretrieve(url, img_path)
            except:
                try: 
                    time.sleep(2)  # try again after 2 seconds
                    urllib.request.urlretrieve(url, img_path)
                except:
                    logging.warning('{}: failed request: {}'.format(i, url))
                    download_success[i] = False
                    continue # skip this neuron
        logging.info("Finished Dowloading {}/{} PNG images (url from metadata) from the website.".format(len([i for i in download_success if download_success[i] is True]), len(download_success)))

        # some cells have the wrong png link in the metadta file. Need to scrape from html. 
        cids_bad_png_link = [i for i in download_success if download_success[i] == False]
        self.download_png_from_html(folder=folder, cids=cids_bad_png_link, overwrite=overwrite)
    
    def download_png_from_html(self, folder='web_png', cids=None, overwrite=False):
        logging.info("Dowloading PNG images (url from html) from the website.")
        if cids is None:
            cids = self.cell_ids
        png_dir = os.path.join(self.data_dir, folder)
        if not os.path.exists(png_dir):
            os.mkdir(png_dir)
        http = urllib3.PoolManager()
        download_success = dict()
        for cid in cids:
            http_address = 'http://neuromorpho.org/neuron_info.jsp?neuron_id={}'.format(cid)
            html = http.request('GET', http_address)
            soup = BeautifulSoup(html.data)
            imgfile = soup.find_all('img', src=re.compile('^./images/imageFiles'))
            if len(imgfile) != 1:
                logging.warning('image link error: {} image links found'.format(len(imgfile)))
            sublink = imgfile[0].get('src')
            sublink = sublink.lstrip('.')
            sublink = sublink.lstrip('/')
            url = website + sublink
            url = url.replace(' ', '%20')
            logging.info(url)
            download_success[cid] = True
            try:
                urllib.request.urlretrieve(url, img_path)
            except:
                try: 
                    time.sleep(2)  # try again after 2 seconds
                    urllib.request.urlretrieve(url, img_path)
                except:
                    logging.warning('{}: failed request'.format(i))
                    download_success[cid] = False
                    continue 
        logging.info("Finished Dowloading {}/{} PNG images (url from html) from the website.".format(len([i for i in download_success if download_success[i] is True]), len(download_success)))

    def download_morphometry(self, filename='morphometry.json', cids=None):
        logging.info("Downloading morphometry data from the website.")
        if cids is None:
            cids = self.cell_ids
        self.morphometry = dict()
        for i in cids:
            if i % 1000 == 0:
                logging.info(i)
            url = "http://neuromorpho.org/api/morphometry/id/{}.json".format(i)
            try:
                req = requests.get(url)
            except:
                try:
                    time.sleep(2)  # try again after 2 seconds
                    req = requests.get(url)
                except:
                    logging.warning('{}: failed request'.format(i))
                    continue  # skip this neuron
            if req.status_code != 200:
                logging.warning('{}: bad request status'.format(i))
                continue 
            req_json = req.json()
            if 'error' in req_json:
                continue # skip this neuron
            self.morphometry[i] = req_json
        logging.info("Finished downloading morphometry data from {}/{} cells".format(len(self.morphometry), len(self.cell_ids)))
        self.save_json(self.morphometry, os.path.join(self.data_dir, filename))

    def load_morphology(self, morpho_path):
        with open(morpho_path, 'r') as f:
            self.morphology = json.load(f)
            self.morphology = {int(k): v for k, v in self.morphology.items()}

    def download_swc(self, folder='swc_std', cids=None):
        logging.info("Downloading SWC files from the website.")
        swc_dir = os.path.join(self.data_dir, folder)
        if not os.path.exists(swc_dir):
            os.mkdir(swc_dir)
        if cids is None:
            cids = self.cell_ids
        http = urllib3.PoolManager()
        download_success = dict()
        for cid in cids: 
            http_address = 'http://neuromorpho.org/neuron_info.jsp?neuron_id=%s' % cid
            html = http.request('GET', http_address)
            soup = BeautifulSoup(html.data, features="lxml")
            mfile = soup.find_all('a', text='Morphology File (Standardized)')
            if len(mfile) != 1:
                logging.warning('swc link error: {} swc links found'.format(len(mfile)))
            sublink = mfile[0].get('href')
            url = self.website + sublink
            swc_path = os.path.join(swc_dir, "{}.swc".format(cid))
            download_success[cid] = True
            try:
                urllib.request.urlretrieve(url, swc_path)
            except:
                try: 
                    time.sleep(2)  # try again after 2 seconds
                    urllib.request.urlretrieve(url, img_path)
                except:
                    logging.warning('{}: failed request'.format(i))
                    download_success[cid] = False
                    continue # skip this neuron
        logging.info("Finished Dowloading {}/{} SWC files from the website.".format(len([i for i in download_success if download_success[i] is True]), len(download_success)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download data from NeuroMorpho.org')
    parser.add_argument('--datapath', default='../data_download/test_download',
                    type=str, help='download path')
    parser.add_argument('--max-cell-id', default=100, type=int,
                    help='Maximum cell id to download (e.g. from cell 1 - N)')
    args = parser.parse_args()

    downloader = NeuroMorphoDownload(max_id=args.max_cell_id, data_dir=args.datapath)
    downloader.download_metadata()
    downloader.download_swc()
    downloader.download_png_from_metadata()
    downloader.download_morphometry()
    