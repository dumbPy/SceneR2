from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from .forms import UploadCANForm, UploadVideoForm
import os
import pkg_resources
from .process import process_can_and_video
from django.templatetags.static import static

# def index(request):
#     return render(request, 'index.html')

def index(request):
    if request.method == 'POST':
        video_form = UploadVideoForm(request.POST, request.FILES)
        can_form   = UploadCANForm(  request.POST, request.FILES)
        if video_form.is_valid() and can_form.is_valid():
            for key in request.FILES:
                file = request.FILES[key]
                os.makedirs('tmp', exist_ok=True)
                path = static(f'data/{file.name}')
                ext = file.name.split('.')[-1]
                if ext in ['mp4', 'avi']: vid_filename = file.name
                if ext == 'csv': can_filename = file.name
                with open(path, 'wb+') as f:
                    for chunk in file.chunks():
                        f.write(chunk)
            try: pred = process_can_and_video(static('data/'), 
                        can_filename, vid_filename)
            except:
                # Wrong file types uploaded
                messages.error(request, 'Invalid File types. upload CAN csv and corresponding video (mp4 or avi)')
                form = UploadFileForm()
                return render(request, 'index.html', {'form':form})
            return render(request, 'output.html', {'form':form})
    else:
        form = UploadFileForm()
    return render(request, 'index.html', {'form':form})