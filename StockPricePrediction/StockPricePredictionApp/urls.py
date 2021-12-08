from django.urls import path
from . import views

urlpatterns=[
    path('',views.home,name='home'),
    path('compare/',views.compare,name='compare'),
    path('download/<id>',views.download,name='download'),
    path('predict/',views.predict,name='predict'),
]