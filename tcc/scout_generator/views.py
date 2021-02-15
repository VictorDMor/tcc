from django.shortcuts import render
from django.http import HttpResponse
from .forms import TrainImageForm, UploadImageForm, UploadVideoForm
from .utils import classify_image, generate_report, get_training_image, handle_uploaded_file, handle_uploaded_video, load_model

model = load_model('model/model.json')

# Create your views here.
def index(request):
    return render(request, 'scout_generator/index.html')

def predict_event(request):
    form = UploadImageForm(request.POST, request.FILES)
    if form.is_valid():
        event = handle_uploaded_file(request.FILES['image'], request.FILES['image'].name, model)
        return render(request, 'scout_generator/result.html', {'event': event})
    else:
        return HttpResponse('Fail: {}'.format(form.errors))

def train_cnn(request):
    raw_path = 'D:\\TCC\\images\\raw'
    image = get_training_image(raw_path)
    if image is not False:
        return render(request, 
            'scout_generator/training.html', 
            {
                'image': image['image_path'], 
                # 'event': image['suggested_event'],
                # 'translation': image['suggested_event_translation'],
                'path': raw_path
            })
    else:
        return HttpResponse('Não há mais imagens a serem treinadas!')


def classifier(request):
    form = TrainImageForm(request.POST)
    if form.is_valid():
        classify_image(request.POST['event'], request.POST['path'])
        next_image = get_training_image(request.POST['path'])
        return render(request, 
        'scout_generator/training.html', 
        {
            'image': next_image['image_path'], 
            # 'event': next_image['suggested_event'],
            # 'translation': next_image['suggested_event_translation'],
            'path': request.POST['path']
        })
    else:
        return HttpResponse('Fail: {}'.format(form.errors))

def generate_scout(request):
    form = UploadVideoForm(request.POST, request.FILES)
    if form.is_valid():
        video = request.FILES['video']
        video_name = video.name
        downloaded_path = handle_uploaded_video(video)
        report, scout = generate_report(downloaded_path, video_name, model)
        return render(request, 'scout_generator/scout.html', {'video_name': video_name, 'events': report, 'scout': scout})
    else:
        return HttpResponse('Fail: {}'.format(form.errors))