# facecrop
Face recognition-based video cropper. Will crop landscape video into portrait video based on face tracking.

## Demo on YouTube
[<img src="https://img.youtube.com/vi/L86kAF-R9Tg/0.jpg" width=200>](https://www.youtube.com/watch?v=L86kAF-R9Tg)

## Setup
Basic setup should just be `npm i -g mogery/facecrop`.

To use GPU acceleration, you should do the following:
 * Download and install the [CUDA 10.0 repo package](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork). **Don't install CUDA yet, just the repository containing it.**
 * Install the following packages: `cuda-10-0 libcudnn7 libcudnn7-dev`

## Usage

```
facecrop

  Face recognition-based video cropper. 

Synopsis

  $ facecrop [--interval 6] [--output output.mp4] input.mp4                     
  $ facecrop [-f 6] [-o output.mp4] -i input.mp4                                

Options

  -i, --input file        Input video to process.                                                       
  -f, --interval number   Every nth frame to analyze with AI. Defaults to half of FPS.                  
  -o, --output file       Output video. Defaults to [filename]_facecrop[.ext]                           
  -l, --disable-lerp      Disables interpolation between frames. May speed up final video rendering,    
                          but will cause footage's crops to feel choppy. This is useful if interval is  
                          1, since lerp is unnecessary in that case.                                    
  -c, --codec string      FFMPEG output video codec. copy uses the same codec as input file. Defaults   
                          to ffmpeg's default. 
```

## Model
The model used and contained in `models/` is SSD Mobilenet V1 from [face-api.js](https://github.com/justadudewhohacks/face-api.js#ssd-mobilenet-v1) and [yeephycho](https://github.com/yeephycho/tensorflow-face-detection).