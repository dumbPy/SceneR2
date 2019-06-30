
function center_output(){
    var outputs = document.getElementsByClassName('output');
    for (let i = 0; i < outputs.length; i++) {
        let output = outputs[i];
        if(!(output.classList.contains('text-center'))) {
            output.classList.add('text-center');
            console.log(output.classList);
        }
    }
    setTimeout(center_output, 1000);
}
center_output()