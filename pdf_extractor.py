import os, json

folder = "jsons_4"
files = [f for f in os.listdir(folder) if f.endswith(".json")]

for file in files[:3]:  # show first 3
    path = os.path.join(folder, file)
    print(f"\n===== {file} =====")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Print summary of structure
    if isinstance(data, dict):
        print("Type: dict")
        print("Top-level keys:", list(data.keys())[:10])
        for k in data.keys():
            val = data[k]
            if isinstance(val, list):
                print(f"  {k} -> list[{len(val)}], first item type: {type(val[0]) if val else None}")
            elif isinstance(val, dict):
                print(f"  {k} -> dict with keys: {list(val.keys())[:5]}")
            else:
                print(f"  {k} -> {type(val)}")
    elif isinstance(data, list):
        print("Type: list")
        print("First element keys:", list(data[0].keys())[:10] if isinstance(data[0], dict) else None)
    else:
        print("Type:", type(data))
