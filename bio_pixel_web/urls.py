from django.contrib import admin
from django.urls import path
from detector.views import detect_eye
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', detect_eye),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
