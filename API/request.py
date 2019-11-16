import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Depth':2, 'WS':9, 'RH (%)':6,'Temp (Deg)':34})

print(r.json())