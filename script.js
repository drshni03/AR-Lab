let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let statusDiv = document.getElementById("status");
let currentImage = null;
let originalImageData = null;
let grayscaleImageData = null;
let grayscaleArray = null;
let featurePoints = [];
let orbDescriptors = [];
let arLaunchBtn = document.getElementById("arLaunchBtn");

// Grayscale
function convertToGrayscaleArray(imageData, width, height) {
    let grayscale = new Array(height);
    
    for (let y = 0; y < height; y++) {
        grayscale[y] = new Array(width);
        for (let x = 0; x < width; x++) {
            let idx = (y * width + x) * 4;
            let r = imageData[idx];
            let g = imageData[idx + 1];
            let b = imageData[idx + 2];
            // Standard luminance formula for grayscale
            grayscale[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
    
    return grayscale;
}

function grayscaleToImage(grayscaleArray, width, height) {
    let imageData = ctx.createImageData(width, height);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let idx = (y * width + x) * 4;
            let val = Math.min(255, Math.max(0, Math.floor(grayscaleArray[y][x])));
            imageData.data[idx] = val;     // R
            imageData.data[idx + 1] = val; // G
            imageData.data[idx + 2] = val; // B
            imageData.data[idx + 3] = 255; // A
        }
    }
    
    return imageData;
}

// ORB
class FeatureDetector {
    constructor(nFeatures = 500, scaleFactor = 1.2, nLevels = 8) {
        this.nFeatures = nFeatures;
        this.scaleFactor = scaleFactor;
        this.nLevels = nLevels;
        this.fastThreshold = 20;
    }

    // FAST corner detection
    fastCornerDetector(gray, width, height, threshold) {
        let corners = [];
        
        for (let y = 3; y < height - 3; y++) {
            for (let x = 3; x < width - 3; x++) {
                let center = gray[y][x];
 
                let circlePoints = [
                    gray[y-3][x],
                    gray[y-2][x+1], gray[y-1][x+2], gray[y][x+3],
                    gray[y+1][x+2], gray[y+2][x+1], gray[y+3][x],
                    gray[y+2][x-1], gray[y+1][x-2], gray[y][x-3],
                    gray[y-1][x-2], gray[y-2][x-1]
                ];

                let brighter = 0;
                let darker = 0;
                
                for (let i = 0; i < circlePoints.length; i++) {
                    if (circlePoints[i] > center + threshold) brighter++;
                    if (circlePoints[i] < center - threshold) darker++;
                }

                if (brighter >= 12 || darker >= 12) {
                    corners.push({ x, y, response: Math.abs(brighter - darker) });
                }
            }
        }
        
        return corners;
    }

    computeOrientation(gray, x, y, radius = 15) {
        let m00 = 0, m10 = 0, m01 = 0;
        
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                let px = x + dx;
                let py = y + dy;
                if (px >= 0 && px < gray[0].length && py >= 0 && py < gray.length) {
                    let intensity = gray[py][px];
                    m00 += intensity;
                    m10 += dx * intensity;
                    m01 += dy * intensity;
                }
            }
        }
        
        if (m00 === 0) return 0;
        let orientation = Math.atan2(m01, m10);
        return orientation;
    }

    // BRIEF descriptor
    computeBRIEFDescriptor(gray, x, y, orientation, patchSize = 31) {
        let descriptor = [];
        let halfPatch = Math.floor(patchSize / 2);

        let cosTheta = Math.cos(orientation);
        let sinTheta = Math.sin(orientation);
    
        let pointPairs = this.generatePointPairs(256, patchSize);
        
        for (let i = 0; i < pointPairs.length; i++) {
            let p1 = pointPairs[i].p1;
            let p2 = pointPairs[i].p2;

            let x1_rot = p1.x * cosTheta - p1.y * sinTheta;
            let y1_rot = p1.x * sinTheta + p1.y * cosTheta;
            let x2_rot = p2.x * cosTheta - p2.y * sinTheta;
            let y2_rot = p2.x * sinTheta + p2.y * cosTheta;

            let px1 = Math.min(Math.max(x + Math.round(x1_rot), 0), gray[0].length - 1);
            let py1 = Math.min(Math.max(y + Math.round(y1_rot), 0), gray.length - 1);
            let px2 = Math.min(Math.max(x + Math.round(x2_rot), 0), gray[0].length - 1);
            let py2 = Math.min(Math.max(y + Math.round(y2_rot), 0), gray.length - 1);
            
            let val1 = gray[py1][px1];
            let val2 = gray[py2][px2];

            descriptor.push(val1 < val2 ? 1 : 0);
        }
        
        return descriptor;
    }

    generatePointPairs(nPairs, patchSize) {
        let pairs = [];
        let halfPatch = Math.floor(patchSize / 2);
        
        for (let i = 0; i < nPairs; i++) {

            let p1 = {
                x: Math.floor(Math.random() * patchSize) - halfPatch,
                y: Math.floor(Math.random() * patchSize) - halfPatch
            };
            let p2 = {
                x: Math.floor(Math.random() * patchSize) - halfPatch,
                y: Math.floor(Math.random() * patchSize) - halfPatch
            };
            pairs.push({ p1, p2 });
        }
        
        return pairs;
    }

    nonMaxSuppression(corners, minDistance = 10) {
        corners.sort((a, b) => b.response - a.response);
        let filtered = [];
        
        for (let corner of corners) {
            let tooClose = false;
            for (let kept of filtered) {
                let dx = corner.x - kept.x;
                let dy = corner.y - kept.y;
                if (dx * dx + dy * dy < minDistance * minDistance) {
                    tooClose = true;
                    break;
                }
            }
            if (!tooClose) {
                filtered.push(corner);
            }
        }
        
        return filtered.slice(0, this.nFeatures);
    }

    // Build pyramid to detect scale invariance
    buildPyramid(gray, width, height) {
        let pyramid = [gray];
        let scales = [1.0];
        
        for (let level = 1; level < this.nLevels; level++) {
            let prevLevel = pyramid[level - 1];
            let newWidth = Math.floor(prevLevel[0].length / this.scaleFactor);
            let newHeight = Math.floor(prevLevel.length / this.scaleFactor);
            let scaled = new Array(newHeight);
            
            // Simple scaling by averaging
            for (let y = 0; y < newHeight; y++) {
                scaled[y] = new Array(newWidth);
                for (let x = 0; x < newWidth; x++) {
                    let origX = Math.floor(x * this.scaleFactor);
                    let origY = Math.floor(y * this.scaleFactor);
                    scaled[y][x] = prevLevel[origY][origX];
                }
            }
            
            pyramid.push(scaled);
            scales.push(scales[level - 1] * this.scaleFactor);
        }
        
        return { pyramid, scales };
    }

    detectAndCompute(grayArray, width, height) {
        console.log("Starting feature detection...");
        let { pyramid, scales } = this.buildPyramid(grayArray, width, height);
        
        let allKeypoints = [];
        let allDescriptors = [];
        
        for (let level = 0; level < pyramid.length; level++) {
            let levelImage = pyramid[level];
            let levelHeight = levelImage.length;
            let levelWidth = levelImage[0].length;
            let scale = scales[level];

            let corners = this.fastCornerDetector(levelImage, levelWidth, levelHeight, this.fastThreshold);

            let keypoints = this.nonMaxSuppression(corners, 10);
 
            for (let kp of keypoints) {
                let orientation = this.computeOrientation(levelImage, kp.x, kp.y);
                let descriptor = this.computeBRIEFDescriptor(levelImage, kp.x, kp.y, orientation);

                allKeypoints.push({
                    x: Math.round(kp.x * scale),
                    y: Math.round(kp.y * scale),
                    response: kp.response,
                    orientation: orientation,
                    scale: scale,
                    level: level
                });
                
                allDescriptors.push(descriptor);
            }
        }

        let finalKeypoints = this.nonMaxSuppression(allKeypoints, 8);

        if (finalKeypoints.length > this.nFeatures) {
            finalKeypoints = finalKeypoints.slice(0, this.nFeatures);
        }

        return { keypoints: finalKeypoints, descriptors: allDescriptors.slice(0, finalKeypoints.length) };
    }
}



// Draw features
function drawFeatures(keypoints) {
    ctx.save();

    for (let i = 0; i < keypoints.length; i++) {
        let kp = keypoints[i];
    
        let colors = ['#FF3366', '#33FF66', '#FF33FF', '#33FFFF', '#FFFF33'];
        let color = colors[kp.level % colors.length];

        let radius = 5 + (kp.scale * 2);
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();

        let orientationX = kp.x + Math.cos(kp.orientation) * radius * 1.2;
        let orientationY = kp.y + Math.sin(kp.orientation) * radius * 1.2;
        ctx.beginPath();
        ctx.strokeStyle = '#FFFF00';
        ctx.lineWidth = 2;
        ctx.moveTo(kp.x, kp.y);
        ctx.lineTo(orientationX, orientationY);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(orientationX, orientationY, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFF00';
        ctx.fill();
    }    
    ctx.restore();
}

function previewImage(event) {
    let file = event.target.files[0];
    if (!file) return;
    
    let img = new Image();
    img.src = URL.createObjectURL(file);
    
    img.onload = function() {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        currentImage = img;

        originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        arLaunchBtn.disabled = true;
    };

    statusDiv.innerHTML = '<p style="color: green;">Image Loaded!</p>';

}

function generateFeatures() {
    if (!currentImage) {
        statusDiv.innerHTML = '<p style="color: red;">✗ Please upload an image first</p>';
        return;
    }
    // RGB Image Data
    let rgbImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Grayscale
    grayscaleArray = convertToGrayscaleArray(rgbImageData.data, canvas.width, canvas.height);
    let grayscaleImageDataObj = grayscaleToImage(grayscaleArray, canvas.width, canvas.height);
    ctx.putImageData(grayscaleImageDataObj, 0, 0);
    
    // Small delay to show the grayscale image
    setTimeout(() => {        
        // Detect feature
        let orb = new FeatureDetector(500, 1.2, 8);
        let { keypoints, descriptors } = orb.detectAndCompute(grayscaleArray, canvas.width, canvas.height);
        featurePoints = keypoints;
        orbDescriptors = descriptors;
        
        // Draw feature
        drawFeatures(keypoints);

        statusDiv.innerHTML = '<p style="color: green;">Features detected!</p>';

        arLaunchBtn.disabled = false;
    }, 100);
}

function saveMarker() {
    if (!currentImage) {
        statusDiv.innerHTML = '<p style="color: red;">✗ Nothing to save. Please upload and process an image first.</p>';
        return;
    }
    
    let link = document.createElement('a');
    link.download = "ar-marker.png";
    link.href = canvas.toDataURL();
    link.click();
}

function resetImage() {
    if (originalImageData) {
        ctx.putImageData(originalImageData, 0, 0);
        statusDiv.innerHTML = '<div style="background: #fff3e0; padding: 10px; border-radius: 8px;">' +
                              '<p>Image reset to original RGB</p>' +
                              '<p>Click "Detect Features" to detect features</p>' +
                              '</div>';
        featurePoints = [];
        orbDescriptors = [];
        grayscaleArray = null;
        arLaunchBtn.disabled = true;
    } else if (currentImage) {
        ctx.drawImage(currentImage, 0, 0);
        statusDiv.innerHTML = '<p>Image reset. Click "Detect Features" to detect features.</p>';
        featurePoints = [];
        orbDescriptors = [];
        grayscaleArray = null;
        arLaunchBtn.disabled = true;
    }
}

function prepareAndLaunchAR() {
    if (currentImage) {
        let markerData = canvas.toDataURL();
        localStorage.setItem('arMarkerImage', markerData);
        localStorage.setItem('featurePoints', JSON.stringify(featurePoints));
        localStorage.setItem('orbDescriptors', JSON.stringify(orbDescriptors.map(d => Array.from(d))));
        localStorage.setItem('markerDimensions', JSON.stringify({
            width: canvas.width,
            height: canvas.height,
            featureCount: featurePoints.length,
            isGrayscale: true,
            detectorType: 'ORB'
        }));
    }

    window.location.href = 'ar.html';
}

