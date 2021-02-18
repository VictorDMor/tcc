import cv2
import glob
import io
import numpy as np
import os
import pyautogui
import time
from .classifier import check_similarity, check_threshold, load_model, identify_event
from datetime import datetime
from PIL import Image

EVENT_TRANSLATIONS = {
    'falta': 'freekick',
    'escanteio': 'corner',
    'bola rolando': 'none',
    'pênalti': 'penalty',
    'comemoração': 'celebration',
    'close': 'close'
}

def split_train_valid(path, folder):
    complete_path = path + '\\' + folder
    files = os.listdir(complete_path)
    if len(files) > 0:
        for i in range(len(files)):
            if '.png' in files[i]:
                if i % 5 == 0: # 80% train, 20% valid
                    os.rename(complete_path + '\\' + files[i], complete_path + '\\valid\\' + files[i])
                else:
                    os.rename(complete_path + '\\' + files[i], complete_path + '\\train\\' + files[i])

def generate_report(video_path, video_name, model):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    events = []
    scout = {}
    event_frames = []
    highlights_frames = []
    all_events = []
    i = 0
    while video.isOpened():
        _, frame = video.read()
        try:
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pass_threshold = check_threshold(frame)
            i += 1

            if len(event_frames) >= 1 and check_similarity(event_frames[0], frame) > 0.75:
                pass
            elif pass_threshold:
                event = 'close'
            else:
                event = identify_event(model, image=rgb_frame)

            print('Frame {}'.format(i))

            event_frames.append(frame)
            all_events.append(event)
            last_event = all_events[-2] if len(all_events) > 1 else ''

            if len(event_frames) <= 1:
                timestamp_start = timestamp
            else:
                if len(all_events) > 1 and event != last_event:
                    if len(event_frames) < fps*7:
                        pass
                    else:
                        if last_event in scout:
                            scout[last_event] += 1
                        else:
                            scout[last_event] = 1
                        frame_name = '{}_{}'.format(last_event, scout[last_event])
                        print('Timestamp: {:.2f}'.format(int(timestamp_start)))
                        print('Event: {}'.format(event))
                        video_tmp_path = 'scout_generator/static/scout_generator/video/tmp/' + frame_name + '.mp4'
                        event_video = cv2.VideoWriter(video_tmp_path, 
                                    cv2.VideoWriter_fourcc(*'avc1'), 
                                    fps, # Frames per second
                                    (frame.shape[1], frame.shape[0]))
                            
                        for event_frame in event_frames[:-1]:
                            event_video.write(event_frame)
                            highlights_frames.append(event_frame)

                        events.append({
                            'timestamp': timestamp_start,
                            'video_src': frame_name + '.mp4',
                            'video_width': frame.shape[1],
                            'video_height': frame.shape[0],
                            'event_name': last_event
                        })
                        event_frames = [event_frames[-1]]
                        timestamp_start = timestamp
        except cv2.error:
            break
        except KeyboardInterrupt:
            files = glob.glob('scout_generator/static/scout_generator/video/tmp/*.mp4', recursive=True)
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
    highlights_name = 'highlights.mp4'
    highlights_video = cv2.VideoWriter('scout_generator/static/scout_generator/video/tmp/' + highlights_name, 
                                    cv2.VideoWriter_fourcc(*'avc1'), 
                                    fps, # Frames per second
                                    (frame.shape[1], frame.shape[0]))
    for highlight_frame in highlights_frames:
        highlights_video.write(highlight_frame)
    video.release()
    return events, scout, highlights_name

def handle_uploaded_file(f, name, model):
    image = Image.open(io.BytesIO(bytearray(f.read())))
    image_path = 'scout_generator/static/scout_generator/img/{}'.format(name)
    image.save(image_path)
    event = identify_event(model, path=image_path)
    return {
        'image_path': image_path,
        'image_name': name,
        'event_name': event
    }

def handle_uploaded_video(video):
    final_path = 'scout_generator/static/scout_generator/video/' + video.name
    with open(final_path, 'wb+') as destination:
        for chunk in video.chunks():
            destination.write(chunk)
    return final_path

def get_training_image(raw_path):
    image_folder = os.getcwd() + '\\scout_generator\\static\\scout_generator\\img\\training'
    training_image = os.listdir(raw_path)[0]
    if '.png' in training_image:
        # suggested_event = identify_event(raw_path + '\\' + training_image)
        os.rename(raw_path + '\\' + training_image, image_folder + '\\' + training_image)
        return {
            'image_path': training_image,
            # 'suggested_event': suggested_event,
            # 'suggested_event_translation': EVENT_TRANSLATIONS[suggested_event]
        }
    else:
        return False

def classify_image(event_name, raw_path):
    image_folder = os.getcwd() + '\\scout_generator\\static\\scout_generator\\img\\training'
    training_image = os.listdir(image_folder)[0]
    if event_name == 'delete':
        os.remove(image_folder + '\\' + training_image)
    elif event_name == 'organize':
        split_train_valid(raw_path, 'aerial')
        split_train_valid(raw_path, 'bench')
        split_train_valid(raw_path, 'celebration')
        split_train_valid(raw_path, 'close')
        split_train_valid(raw_path, 'corner')
        split_train_valid(raw_path, 'deception')
        split_train_valid(raw_path, 'fans')
        split_train_valid(raw_path, 'freekick')
        split_train_valid(raw_path, 'goal')
        split_train_valid(raw_path, 'kickoff')
        split_train_valid(raw_path, 'manager')
        split_train_valid(raw_path, 'none')
        split_train_valid(raw_path, 'penalty')
        split_train_valid(raw_path, 'redcard')
        split_train_valid(raw_path, 'referee')
        split_train_valid(raw_path, 'yellowcard')
    else:
        os.rename(image_folder + '\\' + training_image, raw_path + '\\' + event_name + '\\' + training_image)

def screenshotter(model):
    # MacBook Pro screen: 1680x1050
    # Start: X=390, Y=385
    # Resolution: 2100x1187
    # LG monitor screen: 2560x1080
    # Full HD screen: 1920x1080
    time.sleep(3)
    while True:
        image = np.array(pyautogui.screenshot(region=(390, 385, 2100, 1187)).convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if np.allclose(image[0, 0], [255, 255, 255]):
            return True
        event = identify_event(model, image=image)
        cv2.imwrite('images/{}/{}.png'.format(EVENT_TRANSLATIONS[event], datetime.now().strftime('%Y%m%d-%H%M%S')), image)
        time.sleep(1)