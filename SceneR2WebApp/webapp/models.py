from django.db import models
from django.core.validators import FileExtensionValidator
# Create your models here.
from django import forms

class Video_and_CAN(models.Model):
    video = models.FileField(upload_to='uploads',
            validators=[FileExtensionValidator(
                        allowed_extensions=['mp4','avi'], 
                        message='invalid video extension')])

    can = models.FileField(upload_to='uploads', 
          validators=[FileExtensionValidator(allowed_extensions=['csv'], message='invalid CAN extension')])
