from django.urls import path
from . import views

urlpatterns=[
    path('',views.home,name='home'),
    path('compare/',views.compare,name='compare'),
    path('download/<id>',views.download,name='download'),
    path('predict/',views.predict,name='predict'),
    path('all_stocks/',views.all_stocks,name='all_stocks'),
    path('details/<id>',views.details,name='details'),
]