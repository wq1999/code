import os


def get_audio(file, path2store):
    mp3_path = os.path.join(path2store, 'audio.wav')
    EXTRACT_VIDEO_COMMAND = ('ffmpeg -i "{from_video_path}" '
                             '-f {audio_ext} -ab 192000 '
                             '-vn "{to_audio_path}"')
    command = EXTRACT_VIDEO_COMMAND.format(
        from_video_path=file, audio_ext='wav', to_audio_path=mp3_path,
    )
    os.system(command)


# extract audio from video
path2data = r'D:\Study\MediaEval'

extension = ".mp4"
for root, dirs, files in os.walk(path2data, topdown=False):
    for name in files:
        if extension not in name:
            continue
        path2vid = os.path.join(root, name)
        path2store = path2vid.replace('D:\Study\MediaEval', r'D:\Study\mediaeval_audio')
        path2store = path2store.replace(extension, "")
        print(path2store)
        if os.path.exists(path2store):
            continue
        os.makedirs(path2store, exist_ok=True)
        get_audio(path2vid, path2store)
