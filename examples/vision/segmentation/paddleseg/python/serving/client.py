import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')


if __name__ == '__main__':
    url = "http://127.0.0.1:8000/fd/mobileseg"
    headers = {"Content-Type": "application/json"}

    im = cv2.imread("cityscapes_demo.png")
    data = {
        "data": {
            "image": cv2_to_base64(im)
        },
        "parameters": {}
    }

    resp = requests.post(url=url, headers=headers, data=json.dumps(data))
    if resp.status_code == 200:
        r_json = json.loads(resp.json()["result"])
        print(r_json)
    else:
        print("Error code:", resp.status_code)
        print(resp.text)
