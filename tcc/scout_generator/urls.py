from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_event, name='predict_event'),
    path('scout/', views.generate_scout, name='generate_scout'),
    path('train/', views.train_cnn, name='train_cnn'),
    path('train/classifier/', views.classifier, name='classifier'),
    path('get-images/', views.get_images, name='get_images')
]