var slider = document.getElementById('myRange');
var video = document.getElementById('video1');
var fps_slider = document.getElementById('fps_slider');
var print_fps = document.getElementById('print_fps');

function setSliderValue(video){
  if (video.paused || video.ended) return false;
  let newVal = Math.ceil(video.currentTime*(slider.max-slider.min)/(video.duration));
  slider.value = newVal;
  // console.log(newVal)
  setTimeout(setSliderValue, 20, video)
}

video.addEventListener('play', function(){
  setSliderValue(this);
}, false);

function showFrame(newVal) {
  var v = document.getElementById('video1')
  v.currentTime = newVal*v.duration/500
}

// for Chrome, while firefox works fine with ininput attribute from html
slider.addEventListener('input', showFrame(this.value))

function togglePlayPause(video){
  if (video.paused || video.ended) video.play();
  else video.pause();
}

function set_fps(newVal){
  video.playbackRate = newVal/25;
  print_fps.innerHTML = 'FPS: '+newVal;
}