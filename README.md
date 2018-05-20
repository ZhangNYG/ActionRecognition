# ActionRecognition
### Graduation Project for Bachelor Degree. Action Recognition based on pose estimation.  
By Kenessary Koishybay, Nauryzbek Razakhbergenov, Anara Sandygulova. Nazarbayev University  


## Introduction
Pose estimation algortihm is based on [tensorflow implementation](https://github.com/ildoonet/tf-pose-estimation) of
[Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)   

<p align="left">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/asd1.gif", width="400">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/asd2.gif", width="400">
</p>
<p align="left">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/demo.gif", width="400">
</p>


Requirements:
- Python 2.7  
- OpenCV3  
- sklearn  
- scipy  
- imutils  
- xgboost    
	
	
## Running
To run my code you need to type:  
&nbsp;&nbsp;&nbsp;&nbsp;python -B Main.py &lt;input_video&gt; &lt;output_video&gt;  
Here, arguments <input_video> and <output video> are optional, 
and default values can be seen in 
  
## How it works
### Pose Estimation
<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/pose.png">
</p>
Pose estimation is the process of locating body key points.<br/>
Pose estimation problem is usually solved by training Deep Learning architectures with annotated datasets such as 
<a href="http://human-pose.mpi-inf.mpg.de">MPII</a> or <a href="http://cocodataset.org/">COCO</a><br/>
We didn't have computational power to train on these datasets. Thus, we used pre-trained model mentioned at the beginning.
</br>
Architecture:
<p align="center">
<img src="https://github.com/I3orn2FLY/Git_add-ons/blob/master/ActivityRecognition/arch.jpg">
</p>
However after looking that even prediction on that architecture takes too much time we use here Mobile Net.

## Training


### Data Extraction



### Model Selection and Training


## TODO


