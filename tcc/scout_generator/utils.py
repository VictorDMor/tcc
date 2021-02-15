import cv2
import io
import os
from .classifier import check_threshold, load_model, identify_event
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
    events = []
    scout = {}
    event_track = {
        'current': '',
        'consecutive': 0
    }
    event_frames = []
    all_events = []
    while video.isOpened():
        ret, frame = video.read()
        try:
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pass_threshold = check_threshold(frame)

            if pass_threshold:
                event = 'close'
            else:
                event = identify_event(model, image=rgb_frame)

            all_events.append({
                'timestamp': int(timestamp),
                'event': event
            })
            
            if event == event_track['current']:
                event_track['consecutive'] += 1
                event_frames.append(frame)
            else:
                event_track['consecutive'] = 1
                event_frames = [frame]
                first_frame_event = event
                timestamp_start = timestamp
                
            event_track['current'] = event

            if event_track['consecutive'] >= 60:
                if EVENT_TRANSLATIONS[event] in scout:
                    scout[EVENT_TRANSLATIONS[event]] += 1
                else:
                    scout[EVENT_TRANSLATIONS[event]] = 1
                frame_name = '{}_{}'.format(event, scout[EVENT_TRANSLATIONS[event]])
                print('Timestamp: {:.2f}'.format(timestamp_start))
                print('Frame: {}'.format(frame_name))
                print('Event: {}'.format(first_frame_event))

                video_tmp_path = 'scout_generator/static/scout_generator/video/tmp/' + frame_name + '.mp4'
                event_video = cv2.VideoWriter(video_tmp_path, 
                            cv2.VideoWriter_fourcc(*'avc1'), 
                            30, # Frames per second
                            (frame.shape[1], frame.shape[0]))
                
                for event_frame in event_frames:
                    event_video.write(event_frame)

                events.append({
                    'timestamp': timestamp_start,
                    'frame_src': frame_name + '.png',
                    'video_src': frame_name + '.mp4',
                    'video_width': frame.shape[1],
                    'video_height': frame.shape[0],
                    'event_name': first_frame_event
                })
                event_frames = []
                event_track['consecutive'] = 0
        except cv2.error:
            break
    video.release()
    print(all_events)
    return events, scout

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