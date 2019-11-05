/* This file is referenced in the HTML files and called in order for the canvas to be drawn on and display
   Adapted from https://www.jitsejan.com/python-and-javascript-in-flask.html */

$( document ).ready(function() {
  /* Initializing the canvas so it may be passed to the initialization function */
  var container = document.getElementById('canvas');
  init(container, 200, 200, '#ddd');

  /* Main drawing function, this is ran initially */
  function init(container, width, height, fillColor) {
    var canvas = createCanvas(container, width, height);
    var ctx = canvas.getContext('2d');
    /* Variables used for mouse/position getters */
    var mouse = {x: 0, y: 0};
    var last_mouse = {x: 0, y: 0};




    /* For capturing the position of the mouse */
    canvas.addEventListener('mousemove', function(e) {
      last_mouse.x = mouse.x;
      last_mouse.y = mouse.y;
              
      /* Modified this to avoid problems with
         scrolling the page */
      if (e.offsetX) {
        mouse.x = e.offsetX;
        mouse.y = e.offsetY;
      }
    });    

    
    /* While the mouse is pressed down and moving, draw, otherwise return */
    canvas.onmousemove = function(e) {
      /* If nobody is drawing */
      if (!canvas.isDrawing) {
        return;
      }

      /* Co-ordinate values for drawing */
      var x = e.pageX - this.offsetLeft;
      var y = e.pageY - this.offsetTop;
    };

    /* The two methods below are controllers, knowing when to draw and when to stop */
    canvas.addEventListener('mousedown', function(e) {
      canvas.addEventListener('mousemove', onPaint, false);
    }, false);
    canvas.addEventListener('mouseup', function() {
      canvas.removeEventListener('mousemove', onPaint, false);
    });

    /* If someone presses their mouse down on the canvas, invoke this method */
    var onPaint = function() {
      /* For plotting lines between mouse movements */
      ctx.beginPath();
      ctx.moveTo(last_mouse.x, last_mouse.y);
      ctx.lineTo(mouse.x, mouse.y);
      ctx.closePath();
      ctx.stroke();

      /* Draw on the Canvas */
      ctx.lineWidth = 15 ;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'black';
    };
  } 
  
  /* Canvas creation method, creates canvas object */
  function createCanvas(parent, width, height) {
    /* Ref to canvasBorder within styles.css */
    var canvas = document.getElementById("canvas");
    canvas.context = canvas.getContext('2d');
    return canvas;
  }

  /* Clear the canvas, given the paramaters of the drawable area */
  function clearCanvas() {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /* Invoked when the drawing is published, sends the image data to the backend */
  function publishCanvas() {
    /* Get the canvas */
    var canvas = document.getElementById("canvas");
    var dataURL=canvas.toDataURL();
  
    /* Asynchronous JS and XML post for the backend */
    $.ajax({
        type:"POST",
        url:"/upload",
        data:{ imageBase64:dataURL }
        }).done(function(predictedNumber){
            document.getElementById('mytext').value = predictedNumber;
            console.log(predictedNumber);
        });
  }
  /* Binding for the ClearButton in index.html */
  $( "#clearButton" ).click(function(){
    clearCanvas();
  });

  /* Binding for the PublishButton in index.html */
  $( "#publishButton" ).click(function(){
    publishCanvas();
  });
});