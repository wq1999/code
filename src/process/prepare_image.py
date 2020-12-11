import os
import cv2


def get_frames(filename):
    frames = []
    v_cap = cv2.VideoCapture(filename)

    success, frame = v_cap.read()
    while success:
        success, frame = v_cap.read()
        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    v_cap.release()
    return frames


def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        print(path2img)
        cv2.imwrite(path2img, frame)


# extract image from video for mediaeval dataset
path2data = r'D:\Study\MediaEval'

extension = ".mp4"
for root, dirs, files in os.walk(path2data, topdown=False):
    for name in files:
        if extension not in name:
            continue
        path2vid = os.path.join(root, name)
        frames = get_frames(path2vid)
        path2store = path2vid.replace('D:\Study', r'D:\Study\mediaeval_image')
        path2store = path2store.replace(extension, "")
        print(path2store)
        if os.path.exists(path2store):
            continue
        os.makedirs(path2store, exist_ok=True)
        store_frames(frames, path2store)
