//padding from https://stackoverflow.com/questions/2998784/how-to-output-numbers-with-leading-zeros-in-javascript
const zeroPad = (num, places) => String(num).padStart(places, '0')

let timeLeft = 10 * 60; //number of minutes, in seconds
let timer = setInterval(function(){
    if (timeLeft == 0){
        clearInterval(timer);
    }
    else{
        timeLeft -= 1;
        let minutes = Math.floor(timeLeft/60);
        let seconds = timeLeft % 60;
        document.getElementById("timer").innerHTML = "Time Left: "+ zeroPad(minutes,2) + ":" + zeroPad(seconds,2);
    }
},1000);

