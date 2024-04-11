import base64
import json
import time

import cv2
import flask
from flask import Flask
from detector import Detector
import threading
import queue

app = Flask("mask detection")
Q1 = queue.Queue(maxsize=10)
Q2 = queue.Queue(maxsize=10)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/event")
def update():
    print("event-stream 连接成功")
    return flask.Response(event_proc(), mimetype="text/event-stream")


def cv2_to_base64(image):
    image1 = cv2.imencode('.jpg', image)[1]
    image_code = str(base64.b64encode(image1))[2:-1]
    return "data:image/jpeg;base64," + image_code


def event_proc():
    while True:
        alarm, image = Q2.get(block=True)
        if image is not None:
            code = cv2_to_base64(image)
            yield 'data: {}\n\n'.format(json.dumps(({'image_base64': code, 'alarm': alarm})))


def worker_read_video():
    cap = cv2.VideoCapture("./mask_test.mp4")
    while True:
        ret, frame = cap.read()
        if ret:
            if Q1.full():
                print("que is full, clear it..")
                Q1.get()
            Q1.put_nowait(frame)
        time.sleep(0.5)


def worker_detect():
    d = Detector("best.pt")
    while True:
        frame = Q1.get(block=True)
        if frame is not None:
            alarm, image = d.detect(frame)
            if image is not None:
                if Q2.full():
                    Q2.get()
                Q2.put_nowait((alarm, image))


def worker_flask():
    app.run()


def worker_imshow():
    while True:
        image = Q2.get(block=True)
        if image is not None:
            cv2.imshow("play", image)
            cv2.waitKey(10)


if __name__ == '__main__':
    ths = [threading.Thread(target=worker_read_video),
           threading.Thread(target=worker_detect),
           threading.Thread(target=worker_flask)]
           # threading.Thread(target=worker_imshow)]

    for t in ths:
        t.start()
    for t in ths:
        t.join()
