from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadImageForm
from .utils import handle_uploaded_file

# Create your views here.
def index(request):
    return render(request, 'scout_generator/index.html')

def predict_event(request):
    form = UploadImageForm(request.POST, request.FILES)
    if form.is_valid():
        event = handle_uploaded_file(request.FILES['image'], request.FILES['image'].name)
        return render(request, 'scout_generator/result.html', {'event': event})
    else:
        return HttpResponse('Fail: {}'.format(form.errors))