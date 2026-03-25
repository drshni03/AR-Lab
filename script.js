let canvas=document.getElementById("canvas");

let ctx=canvas.getContext("2d");

let status=document.getElementById("status");

let currentImage=null;



function previewImage(event){

let file=event.target.files[0];

if(!file){

return;

}

let img=new Image();

img.src=URL.createObjectURL(file);

img.onload=function(){

canvas.width=img.width;

canvas.height=img.height;

ctx.drawImage(img,0,0);

currentImage=img;

status.innerText="RGB Image loaded successfully";

}

}



function generate(){

if(typeof cv==='undefined'){

alert("OpenCV not ready");

return;

}

if(currentImage==null){

alert("Upload image first");

return;

}

status.innerText="Converting RGB to Grayscale...";

let src=cv.imread(canvas);

let gray=new cv.Mat();

cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);

cv.imshow('canvas',gray);



setTimeout(()=>{

status.innerText="Extracting features using ORB detector...";

let orb=new cv.ORB();

let keypoints=new cv.KeyPointVector();



orb.detect(gray,keypoints);



cv.drawKeypoints(

gray,

keypoints,

src,

[255,0,0,255]

);



cv.imshow('canvas',src);



status.innerText=

"Feature extraction complete. Total features: "

+keypoints.size();



orb.delete();

keypoints.delete();

gray.delete();

src.delete();



},700);



}



function save(){

if(currentImage==null){

alert("Nothing to save");

return;

}



let link=document.createElement('a');

link.download="feature-marker.png";

link.href=canvas.toDataURL();

link.click();



status.innerText="Marker saved successfully";

}