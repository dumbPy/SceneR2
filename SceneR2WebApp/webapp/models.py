from django.db import models
from django.core.validators import FileExtensionValidator
# Create your models here.
from django import forms

class Video(models.Model):
    video = models.FileField(upload_to='uploads',
            validators=[FileExtensionValidator(
                        allowed_extensions=['mp4','avi'])])

class CAN(models.Model):
    csv = models.FileField(upload_to='uploads', 
          validators=[FileExtensionValidator(allowed_extensions=['csv'])])
