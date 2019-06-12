from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from .forms import UploadVideoAndCANForm
import os
import pkg_resources
from .process import process_can_and_video
from django.templatetags.static import static
from PIL import Image

# def index(request):
#     return render(request, 'index.html')

def index(request):
    if request.method == 'POST':
        form = UploadVideoAndCANForm(request.POST, request.FILES)

        if form.is_valid():
            form = form.save()
            out_folder = os.path.join(settings.MEDIA_ROOT, 'processed')
            message, params = process_can_and_video(out_folder,
                        form.can.path, form.video.path)
            vid_filename = form.video.name.split('/')[-1]
            form = UploadVideoAndCANForm()
            messages.info(request, message)
            context = {'form':form,
                    'video_path':f'/media/processed/{vid_filename}',
                    'can_slider_path': f'/media/processed/can_slider.png',
                    'can_image_full': f'/media/processed/can_image_full.png',
                    'can_few_cols': f'/media/processed/can_few_cols.png'}
            context.update(params)
            return  render(request, 'output.html', context)
        else:
            form = UploadVideoAndCANForm()
            messages.error(request, 'Invalid Uploads extensions, video should be either mp4 or avi and CAN should be a csv files')
            return render(request, 'index.html', {'form':form})
    else:
        form = UploadVideoAndCANForm()
    out_folder = os.path.join(settings.MEDIA_ROOT, 'processed')
    slider_size = Image.open(os.path.join(out_folder, 'can_slider.png')).size
    w,h = slider_size
    slider_height = 0.9*h
    return render(request, 'output.html', {'form':form,
                    'video_path':'/media/processed/20170211_043609_Video_Q7rohFa.mp4',
                    'can_slider_path': "/media/processed/can_slider_default.png",
                    'can_image_full': f'/media/processed/can_image_full.png',
                    'can_few_cols': f'/media/processed/can_few_cols.png','len_can':996,
                    'slider_height': f'{slider_height}px',
                    'len_video':498})
