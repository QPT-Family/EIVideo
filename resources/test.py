import requests
paths = {"video_path": "123",
         "save_path": "123"
         }
import json

paths_json = json.dumps(paths)
r = requests.post("http://127.0.0.1:5000/infer", data=paths_json)

