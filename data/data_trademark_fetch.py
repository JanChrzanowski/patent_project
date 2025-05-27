# os and dotenv for environment variable management
import os
from dotenv import load_dotenv
from pathlib import Path

# API libraries
import requests
import json
import xml.etree.ElementTree as ET

# Data manipulation libraries
import pandas as pd
import numpy as np

#Load environment variables from .env file
load_dotenv()

# Define constants
base_dir = Path(__file__).parent
api_key = os.getenv("API_KEY")
b_number = 'ZT20'
date_start = '2024-04-01'


def get_JWT_token(api_key: str) -> str:
    """
    Sends a POST request to the UPRP API to obtain a JWT token in the format 'Bearer <token>'.

    :param api_key: Your API key (without the 'Bearer' prefix).
    :return: JWT token as a string in the format: 'Bearer <jwt>'.
    :raises: Exception if an error occurs while fetching the token.
    """ 
    url = "https://api.uprp.gov.pl/cmuia/api/external/auth/token"
    headers = {
        "accept": "application/json",
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "access_token": api_key
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    token = data.get("token")

    if not token:
        raise ValueError("Could not get the repsonse from API.")

    return f"Bearer {token}"

def fetch_trademark_data(b_number: str, date_start: str) -> str:

    """Sends  HTTP request and return XML in the strign format ready to parse."""

    url = f"https://api.uprp.gov.pl/ewyszukiwarka/api/external/search-bibliographics-results?publisher=BUP&pwp_type=Znaki%20Towarowe&nr_pub={b_number}&date_start=%{date_start}"
    headers = {
        "accept": "application/xml",
        "Authorization": get_JWT_token(api_key)
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def parse_trademark_xml(xml_data: str):
    """Parsing XML from request and returns data as lists of dictionaries."""
    root = ET.fromstring(xml_data)
    data = []

    for patent in root.findall(".//TradeMark"):
        app_number = patent.findtext("ApplicationNumber")
        patent_type = patent.findtext("MarkFeature") or ""
        patent_types = [pt.strip() for pt in patent_type.split(",")]

        word_mark = None
        image_uri = None
        language_code = None

        if "Word" in patent_types:
            word_elem = patent.find('WordMarkSpecification/MarkVerbalElementText')
            if word_elem is not None:
                word_mark = word_elem.text
                language_code = word_elem.get("languageCode")

        if "Figurative" in patent_types:
            image_elem = patent.find('MarkImageDetails/MarkImage/MarkImageURI')
            if image_elem is not None:
                image_uri = image_elem.text

        class_numbers = []
        class_descriptions = []

        class_details = patent.find("GoodsServicesDetails/GoodsServices/ClassDescriptionDetails")
        if class_details is not None:
            for classes in class_details:
                class_number = classes.findtext("ClassNumber")
                class_description = classes.findtext("GoodsServicesDescription")
                if class_number:
                    class_numbers.append(class_number)
                if class_description:
                    class_descriptions.append(class_description)

        descriptions_joined = "; ".join(class_descriptions)

        data.append({
            "ApplicationNumber": app_number,
            "WordMark": word_mark,
            "LanguageCode": language_code,
            "ImageURI": image_uri,
            "ClassNumbers": class_numbers,
            "ClassDescriptions": descriptions_joined
        })

    return data


def get_trademark_dataframe(b_number: str, date_start: str) -> pd.DataFrame:
    """Reqturns as a DataFrame."""
    try:
        xml_data = fetch_trademark_data(b_number, date_start)
        parsed_data = parse_trademark_xml(xml_data)
        return pd.DataFrame(parsed_data)
    except requests.HTTPError as http_err:
        print(f"HTTP Error: {http_err}")
    except Exception as e:
        print(f"Other Error: błąd: {e}")
    return pd.DataFrame() 

df = get_trademark_dataframe(b_number, date_start)


filename = f'trademark_data{b_number}-{date_start}.pkl'
filepath = base_dir / filename
df.to_pickle(filepath)
