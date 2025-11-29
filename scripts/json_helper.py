import pandas as pd
import json

# Parse CSV into JSON
df = pd.read_csv("../data_samples/descriptions.csv")
objects = {}
for index, row in df.iterrows():
    osc = row["osc"]
    objects[osc] = {}
    objects[osc]["progress_description"] = row["progress_description"]
    objects[osc]["finished_description"] = row["finished_description"]

json_string = json.dumps(objects, indent=4)
with open('../data_samples/descriptions.json', 'w') as f:
    f.write(json_string)


# Load JSON into dict
with open('descriptions.json', 'r') as file:
    objects = json.load(file)
print(objects.get("rolling_dough"))