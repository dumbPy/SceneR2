# SceneR2

A scene recognition, pedestrian detection and vehicle action detection model, with an extensive dataset encapsulation API. The API is written for proprietary data by *Daimler*.

# SceneR2WebApp

A web application in Django that takes a video and corresponding CAN data from Daimler's dataset, and processes them, shows the object detection processed video with [synced CAN](SceneR2WebApp/webapp/media/processed/can_slider.png) as a slider
![Webapp Demo](SceneR2WebApp/webapp/media/SceneR2WebApp.gif)
 
NOTE: This code is useless for anyone except the sponsors of the project who have the coresponding data.

### To Run from docker

```
$ sudo docker run -it --rm\
       --name scener2\
       -p 8000:8000\
       dumbpy/scener2:firefox
```
access from firefox at `localhost:8000`. The sliders don't work on chrome and will be fixed in future updates.

### Credits

Under the Supervison of **Prof. Manjesh Hanawal**<sup>1</sup>,  **Prof. P Balamurugan**<sup>1</sup> and **Dr. Tilak Singh**<sup>2</sup>\
<sup>**1**</sup>Indian Institute of Technology Bombay &nbsp;   &nbsp;  <sup>**2**</sup>Mitsubishi Fuso Truck and Bus Corp. Japan

