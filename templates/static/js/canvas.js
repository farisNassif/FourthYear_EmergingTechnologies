$( document ).ready(function() {

    // Canvas creation method, creates canvas object
    function createCanvas(parent, width, height) {
     // Ref to canvasBorder within styles.css
     var canvas = document.getElementById("canvasBorder");
     canvas.context = canvas.getContext('2d');
     return canvas;
   }
   // Initializing the canvas
   var container = document.getElementById('canvas');
 });