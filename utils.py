import json

with open('output.json', encoding='utf-8') as f:
    data = json.load(f)
    tmp: int = 1999
    for key, _ in data.items():
        a = int(key)
        print(f'{a}')