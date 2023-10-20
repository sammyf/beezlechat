import os.path
import string
import requests
import json

import yaml
from bs4 import BeautifulSoup
from summarizer import Summarizer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
from PyPDF2 import PdfReader

class SearXing:
    CONFIG_FILE = "searxing_config.yaml"

    ############# TRIGGER PHRASES  #############
    ## you can add anything you like here, just be careful not to trigger unwanted searches or even loops
    INTERNET_QUERY_PROMPTS = [
        "search the internet for information on",
        "search the internet for information about",
        "search for information about",
        "search for information on",
        "search for ",
        "i need more information on ",
        "search the internet for ",
        "can you provide me with more specific details on ",
        "can you provide me with details on ",
        "can you provide me with more details on ",
        "can you provide me with more specific details about ",
        "can you provide me with details about ",
        "can you provide me with more details about ",
        "what can you find out about ",
        "what information can you find out about ",
        "what can you find out on ",
        "what information can you find out on ",
        "what can you tell me about ",
        "what do you know about ",
        "ask the search engine on ",
        "ask the search engine about ",
    ]

    FILE_QUERY_PROMPTS = [
        "open the file ",
        "read the file ",
        "summarize the file ",
        "get the file ",
    ]

    DBNAME = ""
    SELFSEARX_TRIGGER = "selfsearx"
    character = "None"
    CONTENT_MARKER = ""
    config = ""

    def __init__(self):
        with open(self.CONFIG_FILE) as f:
            self.config = yaml.safe_load(f)

    def call_searx_api(self, query):
        url = f"{self.config['searx_server']}?q={query}&format=json"
        try:
            response = requests.get(url)
        except:
            return (
                "An internet search returned no results as the SEARX server did not answer."
            )
        # Load the response data into a JSON object.
        try:
            data = json.loads(response.text)
        except:
            return "An internet search returned no results as the SEARX server doesn't seem to output json."
        # Initialize variables for the extracted texts and count of results.
        texts = ""
        count = 0
        max_results = self.config["max_search_results"]
        rs = "An internet search returned these results:"
        result_max_characters = self.config["max_text_length"]
        # If there are items in the data, proceed with parsing the result.
        if "results" in data:
            # For each result, fetch the webpage content, parse it, summarize it, and append it to the string.
            for result in data["results"]:
                # Check if the number of processed results is less than or equal to the maximum number of results allowed.
                if count <= max_results:
                    # Get the URL of the result.
                    # we won't use it right now, as it would be too much for the context size we have at hand
                    link = result["url"]
                    # Fetch the webpage content of the result.
                    content = result["content"]
                    if len(content) > 0:  # ensure content is not empty
                        # Append the summary to the previously extracted texts.
                        texts = texts + " " + content + "\n"
                        # Increase the count of processed results.
                        count += 1
            # Add the first 'result_max_acters' characters of the extracted texts to the input string.
            rs += texts[:result_max_characters]
        # Return the modified string.
        return rs

    ## returns only the first URL in a prompt
    def extract_url(self, prompt):
        url = ""
        # Regular expression to match URLs
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        # Find all URLs in the text
        urls = re.findall(url_pattern, prompt.lower())
        if len(urls) > 0:
            url = urls[0]
        return url

    def trim_to_x_words(self, prompt: string, limit: int):
        rev_rs = []
        words = prompt.split(" ")
        rev_words = reversed(words)
        for w in rev_words:
            rev_rs.append(w)
            limit -= 1
            if limit <= 0:
                break
        rs = reversed(rev_rs)
        return " ".join(rs)

    def extract_query(self, prompt):
        rs = ["", ""]
        # Define your sentence-terminating symbols
        terminators = [".", "!", "?"]
        # Join the terminators into a single string, separating each with a pipe (|), which means "or" in regex
        pattern = "|".join(map(re.escape, terminators))

        search_prompt = ""
        for qry in self.INTERNET_QUERY_PROMPTS:
            if qry in prompt.lower():
                search_prompt = qry
                break
        if search_prompt != "":
            query_raw = prompt.lower().split(search_prompt)[1]
            rs[1] = query_raw[0] + "."
            # Split the text so that we only have the search query
            query = re.split(pattern, query_raw)
            q = query[0]
            q = q.replace(" this year ", datetime.now().strftime("%Y"))
            q = q.replace(" this month ", datetime.now().strftime("%B %Y"))
            q = q.replace(" today ", datetime.now().strftime("'%B,%d %Y'"))
            q = q.replace(" this month ", datetime.now().strftime("%B %Y"))
            q = q.replace(
                " yesterday ", (datetime.today() - timedelta(days=1)).strftime("'%B,%d %Y'")
            )
            q = q.replace(
                " last month ",
                (datetime.today() - relativedelta(months=1)).strftime("%B %Y"),
            )
            q = q.replace(
                " last year ", (datetime.today() - relativedelta(years=1)).strftime("%Y")
            )
            rs[0] = q
            for rest in q[1:]:
                rs[1] += rest
        return rs

    def extract_file_name(self, prompt):
        rs = ""
        query_raw = ""
        for qry in self.FILE_QUERY_PROMPTS:
            pattern = rf"{qry}(.*)"
            match = re.search(
                pattern, prompt, re.IGNORECASE
            )  # re.IGNORECASE makes the search case-insensitive
            if match:
                query_raw = match.group(1)
                break
        if query_raw != "":
            pattern = r"([\"'])(.*?)\1"
            query = re.search(pattern, query_raw)
            if query is not None:
                rs = query.group(2)
        return rs

    def get_page(self, url, prompt):
        text = f"The web page at {url} doesn't have any useable content. Sorry."
        try:
            response = requests.get(url)
        except:
            return f"The page {url} could not be loaded"
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        if len(paragraphs) > 0:
            text = "\n".join(p.get_text() for p in paragraphs)
            text = f"Content of {url} : \n{self.trim_to_x_words(text, self.config['max_text_length'])}[...]\n"
        else:
            text = f"The web page at {url} doesn't seem to have any readable content."
            metas = soup.find_all("meta")
            for m in metas:
                if "content" in m.attrs:
                    try:
                        if (
                                "name" in m
                                and m["name"] == "page-topic"
                                or m["name"] == "description"
                        ):
                            if "content" in m and m["content"] != None:
                                text += f"It's {m['name']} is '{m['content']}'"
                    except:
                        pass
        if prompt.strip() == url:
            text += f"\nSummarize the content from this url : {url}"
        return text

    def read_pdf(self, fname):
        parts = []

        def visitor_body(text, cm, tm, fontDict, fontSize):
            y = tm[5]
            if y > 50 and y < 720:
                parts.append(text)

        pdf = PdfReader(fname)
        rs = ""
        for page in pdf.pages:
            page.extract_text(visitor_text=visitor_body)
            text_body = "".join(parts)
            text_body = text_body.replace("\n", "")
            rs += text_body + "\n"
            if rs != self.trim_to_x_words(rs, self.config["max_text_length"]):
                break
        return rs

    def open_file(self, fname):
        rs = ""
        print(f"Reading {fname}")
        if fname.lower().endswith(".pdf"):
            try:
                rs = self.read_pdf(fname)
            except:
                return "The file can not be opened. Perhaps the filename is wrong?"
        else:
            try:
                with open(fname, "r") as f:
                    lines = f.readlines()
            except:
                return "The file can not be opened. Perhaps the filename is wrong?"
            rs = "\n".join(lines)
        rs = self.trim_to_x_words(rs, self.config["max_text_length"])
        return f"This is the content of the file '{fname}':\n{rs}"

    def check_for_trigger(self, prompt):
        fn = self.extract_file_name(prompt)
        url = self.extract_url(prompt)
        q = self.extract_query(prompt)

        print(f"Filename found : '{fn}'\nQuery found : {q[0]}\nUrl found : {url}\n")
        if fn != "":
            prompt = self.open_file(fn) + self.CONTENT_MARKER+ prompt
        elif url != "":
            prompt = self.get_page(url, prompt)  + self.CONTENT_MARKER+ prompt
        elif q[0] != "":
            searx_results = self.call_searx_api(q[0])
            # Pass the SEARX results back to the LLM.
            if q[1] == "":
                q[1] = "Summarize the results."
            prompt = prompt + "\n"  + self.CONTENT_MARKER+ searx_results  + self.CONTENT_MARKER+ q[1]
        else:
            prompt = ""
        return prompt

