import json
from pprint import pprint

def load_synthetic_dataset():
    """synthetic dataset """
    loaded_data = []

    with open("../sunshine.json", "r") as f:
        target_data = json.load(f)

    with open("../labels.txt", "r") as f:
        labels = [bool(l.strip() == "1") for l in f.readlines()]

    with open("../examples.json", "r") as f:
        vis_specs = json.load(f)
    
    for i, e in enumerate(vis_specs):
        # extract data schema type
        field_ty_map = {}
        for x in e["draco"]:
            if x.startswith("fieldtype"):
                fieldtype = x[x.index("(")+1:x.index(")")].split(",")
                field = fieldtype[0][1:-1]
                ty = fieldtype[1]
                field_ty_map[field] = ty

        loaded_data.append({
            "vl": e["vl"],
            "data_schema": field_ty_map,
            "label": labels[i]
        })

    return loaded_data

def load_codar_dataset():
    """codar dataset"""
    loaded_data = []

    # all data share 
    data_schema = {
        "Name": "string",
        "Miles_per_Gallon": "interger",
        "Cylinders": "interger",
        "Displacement": "interger",
        "Horsepower": "interger",
        "Weight_in_lbs": "interger",
        "Acceleration": "interger",
        "Year": "datetime",
        "Origin": "string"
    }

    with open("../charts.json", "r") as f:
        data = json.load(f)
    
    for key, entry in data.items():
        loaded_data.append({
            "vl": entry["vlSpec"],
            "data_schema": data_schema,
            "label": True
        })
    return loaded_data
