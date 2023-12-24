import math
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
        """
        :param query: The query string to search for in the SEARX server.
        :return: A string representing the search results from the SEARX server.
        """
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
        """
        Extracts the first URL found in a given prompt.

        :param prompt: The text containing URLs.
        :type prompt: str
        :return: The first URL found in the prompt. If no URL is found, an empty string is returned.
        :rtype: str
        """
        url = ""
        # Regular expression to match URLs
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        # Find all URLs in the text
        urls = re.findall(url_pattern, prompt.lower())
        if len(urls) > 0:
            url = urls[0]
        return url

    def trim_to_x_words(self, prompt: string, limit: int):
        """
        Trims a given prompt to a specified number of words.

        :param prompt: The prompt to be trimmed.
        :param limit: The desired number of words.
        :return: The trimmed prompt.

        Example:
            >>> trim_to_x_words("Lorem ipsum dolor sit amet", 3)
            'dolor sit amet'
        """
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
        """
        :param prompt: The prompt from which the query needs to be extracted.
        :return: A list containing two elements. The first element is the extracted query, and the second element is a modified version of the query to use as a search prompt.

        """
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
        """
        :param prompt: The input prompt string containing the file query.
        :return: The extracted file name from the prompt string, or an empty string if no file name is found.

        The `extract_file_name` method takes a prompt string as input and extracts the file name from it. It searches for specific query prompts defined in the `FILE_QUERY_PROMPTS` attribute
        * of the class instance and attempts to find the file name within the query.

        If a file name is found, it is returned. If no file name is found, an empty string is returned.

        The method uses regular expressions to perform the search and extraction. It is case-insensitive, meaning it will match file queries regardless of the case of the query prompts.

        Example usage:

        ```
        prompt = "Please open the file 'example.txt'"
        file_name = instance.extract_file_name(prompt)
        print(file_name)  # Output: example.txt
        ```
        """
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
        """
        :param url: the URL of the web page to retrieve content from
        :param prompt: a prompt to include in the returned text
        :return: the content of the web page, including trimmed text and prompt information

        Retrieves the content from the specified URL and returns it as a formatted text.
        If the web page has readable content, the text will be trimmed to the specified
        maximum length and a prompt will be included. If the web page doesn't have any
        readable content, it will return a default message along with any relevant meta
        information found.
        """
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
            text += f"{prompt}\ncontent from this url : {url}\n"
        return text

    def read_pdf(self, fname):
        """
        :param fname: The file path to the PDF file.
        :return: The extracted text from the PDF file as a string.
        """
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
        """
        This method opens a file and reads its content.

        :param fname: The name of the file to be opened.
        :return: The content of the file as a string.

        :raises ValueError: If the filename does not have a valid format.
        :raises FileNotFoundError: If the file does not exist or cannot be opened.

        Example usage:
            >>> obj = MyClass()
            >>> content = obj.open_file("example.txt")
            >>> print(content)
            This is the content of the file 'example.txt':
            Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        """
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

    def check_for_trigger(self, _prompt, maxlen, count_token):
        """
        :param _prompt: The input prompt string.
        :param maxlen: The maximum length of the resulting prompt.
        :param count_token: The function used to count the number of tokens in a string.
        :return: The processed prompt string.

        This method takes an input prompt string, extracts the file name, URL, and query from the prompt, and performs different actions based on the extracted values. It then constructs a new
        * prompt string with additional content and returns the processed prompt.

        The input prompt string can include various information like file names, URLs, and queries. The method first extracts the file name using the `extract_file_name` method, the URL using
        * the `extract_url` method, and the query using the `extract_query` method.

        After extracting the necessary information, the method prints the extracted file name, query, and URL for debugging purposes.

        Next, the method checks if the file name is not empty. If it is not empty, it calls the `open_file` method to fetch the content of the file. If the URL is not empty, it calls the `get
        *_page` method to fetch the content of the web page. If the query is not empty, it calls the `call_searx_api` method to fetch the content from a search engine.

        If none of the above conditions are met, the content is set to an empty string.

        The method then prints the count of tokens in the content and the content itself for debugging purposes.

        The prompt is constructed by concatenating the content, a specific marker (defined as `self.CONTENT_MARKER`), and the original prompt string.

        If the count of tokens in the prompt exceeds the maximum length (`maxlen`), the prompt is trimmed using the `trim_to_x_words` method. The trim length is calculated based on `maxlen`
        * minus 10% of `maxlen`. The trimmed prompt is then concatenated with a placeholder ("{...}"), the content marker, and the original prompt.

        Finally, the processed prompt is returned.

        Example usage:
            # Create an instance of the class
            instance = ClassName()

            # Call the method with the required parameters
            result = instance.check_for_trigger("example prompt", 100, count_token_function)

            # Print the result
            print(result)
        """
        fn = self.extract_file_name(_prompt)
        url = self.extract_url(_prompt)
        q = self.extract_query(_prompt)

        print(f"Filename found : '{fn}'\nQuery found : {q[0]}\nUrl found : {url}\n")
        try:
            if fn != "":
                content = self.open_file(fn)
            elif url != "":
                content = self.get_page(url, _prompt)
            elif q[0] != "":
                content = self.call_searx_api(q[0])
            else:
                content = ""
        except:
            content = ""

        print(count_token(content),"content : \n",content,)
        prompt = content + self.CONTENT_MARKER + _prompt
        if count_token(prompt) > maxlen:
            content = self.trim_to_x_words( prompt, maxlen - (maxlen*0.1))
            prompt = content + " {...}" + self.CONTENT_MARKER + _prompt

        return prompt

