from django import forms
from .models import Video_and_CAN

class UploadVideoAndCANForm(forms.ModelForm):
    class Meta:
        model = Video_and_CAN
        fields = ("video", "can", 'yolo')
    # video = forms.FileField(label='Video File')
    # can   = forms.FileField(label='CAN file as .csv')