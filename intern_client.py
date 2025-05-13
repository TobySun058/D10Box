import requests

# URL of the API server
api_url = "http://127.0.0.1:5000/predict"

# Path to the image file
image_path = 'image1.jpg'  # Provide the correct path to your image

# Open the image file and send it in a POST request to the API
with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(api_url, files=files)

# Print the response from the server
if response.status_code == 200:
    print("Model Response:", response.json()['response'])
else:
    print(f"Error: {response.status_code} - {response.text}")
