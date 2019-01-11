# light-face-person-object-detection
Powered by Tensorflow Object Detection API &amp; Opencv, detecting traffic lights, faces and person in real time by SSD_mobilenet.

Using tf.Dataset to get video frames in non-real time mode (faster~).

Get frames one by one in real-time mode.

20-30 FPS on average, limited by drawing detection boxes on CPU :(

Rewriting the codes of drawing for GPU may be helpful......
