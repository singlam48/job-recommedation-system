from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('upload', views.upload, name="upload"),
    path('data', views.resumeScanner, name="resumeScanner"),
]

