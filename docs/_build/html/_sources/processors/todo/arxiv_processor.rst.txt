arxiv_processor
===============

.. code-block:: python

	
	
	class ArxivVanityParser:
	    def __init__(self):
	        self.base_url = "https://www.arxiv-vanity.com/"
	
	    def _get_vanity_url(self, arxiv_id):
	        return urljoin(self.base_url, "papers/" + arxiv_id)
	
	    def _fetch_html(self, url):
	        response = requests.get(url)
	        if response.status_code == 200:
	            return response.text
	        else:
	            return None
	
	    def _extract_main_content(self, html):
	        soup = BeautifulSoup(html, "html.parser")
	        paragraphs = soup.find_all("div", {"class": "ltx_para"})
	        content = {idx: p.get_text() for idx, p in enumerate(paragraphs)}
	        return content
	
	    def parse_paper(self, arxiv_id):
	        vanity_url = self._get_vanity_url(arxiv_id)
	        html = self._fetch_html(vanity_url)
	        if html is not None:
	            return self._extract_main_content(html)
	        else:
	            return None
	

.. code-block:: python

	
	
	class ArxivAPI:
	    def __init__(self):
	        self.base_url = "http://export.arxiv.org/api/query?"
	        self.pdf_download_url = "https://arxiv.org/pdf/"
	
	    def search(self, query, max_results=10):
	        url = f"{self.base_url}search_query={query}&max_results={max_results}"
	        response = requests.get(url)
	        if response.status_code == 200:
	            return response.text
	        else:
	            return None
	
	    def download_pdf(self, paper_key, save_directory="./"):
	        pdf_url = f"{self.pdf_download_url}{paper_key}.pdf"
	        response = requests.get(pdf_url)
	        if response.status_code == 200:
	            with open(os.path.join(save_directory, f"{paper_key}.pdf"), "wb") as f:
	                f.write(response.content)
	            print(f"PDF for {paper_key} downloaded successfully.")
	        else:
	            print(f"Error downloading PDF for {paper_key}.")
	

.. code-block:: python

	
	
	class ArxivParser:
	    def __init__(self):
	        self.api = ArxivAPI()
	        self.vanity_parser = ArxivVanityParser()
	
	    def _parse_arxiv_id(self, url):
	        return url.split("/")[-1]
	
	    def parse_papers(self, query, max_results=10):
	        search_results = self.api.search(query, max_results)
	        if search_results is not None:
	            soup = BeautifulSoup(search_results, "html.parser")
	            entries = soup.find_all("entry")
	            paper_list = []
	            for entry in entries:
	                paper_dict = {}
	                arxiv_id = self._parse_arxiv_id(entry.id.string)
	                paper_dict["arxiv_id"] = arxiv_id
	                paper_dict["title"] = entry.title.string
	                paper_dict["summary"] = entry.summary.string
	                paper_dict["content"] = self.vanity_parser.parse_paper(str(arxiv_id))
	                if paper_dict["content"] == None:
	                    continue
	                paper_list.append(paper_dict)
	            return paper_list
	        else:
	            return None
	

.. automodule:: arxiv_processor
   :members:
