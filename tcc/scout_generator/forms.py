from django import forms

class UploadImageForm(forms.Form):
    # title = forms.CharField(max_length=50)
    image = forms.FileField()

class UploadVideoForm(forms.Form):
    # title = forms.CharField(max_length=50)
    video = forms.FileField()

class TrainImageForm(forms.Form):
    event = forms.CharField(max_length=30)
    path = forms.CharField(initial='D:\\TCC\\images\\raw')