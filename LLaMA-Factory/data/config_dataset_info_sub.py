import json

# Define the input and output file paths
input_file_path = '/data/syxu/culture_steering/LLaMA-Factory/data/dataset_info.json'
output_file_path = '/data/syxu/culture_steering/LLaMA-Factory/data/new_dataset_info.json'

# Read the JSON file
with open(input_file_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)
# append sub data
ratio_lst = [float(f'0.{i}') for i in range(1, 10)]

country2culture_dict = {
    "USA": "American",
    "CAN": "Canadian",
    "BOL": "Bolivian",
    "BRA": "Brazilian",
    "GBR": "British",
    "NLD": "Dutch",
    "DEU": "German",
    "UKR": "Ukrainian",
    "CHN": "Chinese",
    "RUS": "Russian",
    "IND": "Indian",
    "THA": "Thai",
    "KEN": "Kenyan",
    "NGA": "Nigerian",
    "ETH": "Ethiopian",
    "ZWE": "Zimbabwean",
    "AUS": "Australian",
    "NZL": "New Zealand",
    "all": "all",
}

"""
  "culture_all": {
    "file_name": "culture_all.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
"""
for ratio in ratio_lst:
    for country in country2culture_dict:
        data[f"culture_{country}{ratio}"] = {
            "file_name": f"{ratio}/culture_{country}.json",
            "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system"
            }
        }
# Save the JSON file with pretty formatting
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, indent=2, ensure_ascii=False)

print(f"Data has been saved to {output_file_path}")