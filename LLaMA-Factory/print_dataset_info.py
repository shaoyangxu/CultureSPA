import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


strr_template = """  "culture_{}": {{
    "file_name": "culture_{}.json",
    "columns": {{
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }}
  }},"""
# print(strr_template)
# exit()
for country, culture in country2culture_dict.items():
    strr = strr_template.format(country, country)
    print(strr)