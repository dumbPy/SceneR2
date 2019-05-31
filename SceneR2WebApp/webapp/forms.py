from django import forms
from .models import Video, CAN

class UploadVideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ("video",)
    # video = forms.FileField(label='Video File')
    # can   = forms.FileField(label='CAN file as .csv')

class UploadCANForm(forms.ModelForm):
    class Meta:
        model = CAN
        fields = ('csv',)