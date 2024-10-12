from django.contrib import admin
from django.urls import path
from CaptuR.views import *  # Adjust 'my_app' to the correct app name

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name = "landing page"),
    path('audio/', audio,name = 'audio'),
    path('words/', words, name='words'),
    path('pics/', pics, name = 'pics'),
    
]
