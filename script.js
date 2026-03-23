function generate(){

let file = document.getElementById("upload").files[0];

if(!file){

alert("Upload image first");

return;

}

let img = new Image();

img.src = URL.createObjectURL(file);

img.onload=function(){

let canvas=document.getElementById("canvas");

let ctx=canvas.getContext("2d");

canvas.width=img.width;

canvas.height=img.height;

ctx.drawImage(img,0,0);

let src=cv.imread(canvas);

let gray=new cv.Mat();

cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);

let keypoints=new cv.KeyPointVector();

let orb=new cv.ORB();

orb.detect(gray,keypoints);

cv.drawKeypoints(
gray,
keypoints,
src,
[255,0,0,255]
);

cv.imshow('canvas',src);

src.delete();

gray.delete();

keypoints.delete();

}

}

function save(){

let canvas=document.getElementById("canvas");

let link=document.createElement('a');

link.download="features.png";

link.href=canvas.toDataURL();

link.click();

}