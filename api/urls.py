from django.urls import path
from .views import analyze_cv

urlpatterns = [
    path('analyze-cv/', analyze_cv, name='analyze_cv'),
]