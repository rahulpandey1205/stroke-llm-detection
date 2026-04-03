import requests

url = "http://127.0.0.1:8000/upload_report"
# Creating a dummy text file to simulate an upload if no PDF is handy, 
# but the goal is to trigger the logic in app.py
files = {'file': ('test.txt', 'This is a test file content', 'text/plain')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Response Content:", response.text[:500])
except Exception as e:
    print("Error:", e)
