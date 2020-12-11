import os

path2data = r'D:\Study\data\VideoEmotion'
sub_folders = os.listdir(path2data)

# split video
for sub_folder in sub_folders:
    print('sub_folder:', sub_folder)
    path2aCatgs = os.path.join(path2data, sub_folder)

    listOfCategories = os.listdir(path2aCatgs)
    print(listOfCategories, len(listOfCategories))

    extension = ".mp4"
    for root, dirs, files in os.walk(path2aCatgs, topdown=False):
        for name in files:
            if extension not in name:
                continue
            path2vid = os.path.join(root, name)
            path2store = path2vid.replace('VideoEmotion', 'VideoEmotion-Scene')
            path2store = path2store.replace(extension, "")
            print(path2store)
            cmds = 'scenedetect -i ' + path2vid + ' -o ' + path2store + ' detect-content ' + '-t 27 split-video'
            os.system(cmds)
            # if os.path.exists(path2store):
            #     continue
            # os.makedirs(path2store, exist_ok=True)
