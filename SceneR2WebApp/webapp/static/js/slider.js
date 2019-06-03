function setSliderValue(video){
  if (video.paused || video.ended) return false;
  
  let newVal = Math.ceil(video.currentTime*(slider.max-slider.min)/(video.duration));
  slider.value = newVal;
  // console.log(newVal)
  setTimeout(setSliderValue, 20, video)
}
var slider = document.getElementById('myRange');
var video = document.getElementById('video1');
video.addEventListener('play', function(){
  setSliderValue(this);
}, false);

function showFrame(newVal) {
  var v = document.getElementById('video1')
  v.currentTime = newVal*v.duration/500
}

function togglePlayPause(video){
  if (video.paused || video.ended) video.play();
  else video.pause();
}