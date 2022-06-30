# Pushup-recognition
MediaPipe Pose is an ML solution for tracking high-quality body poses, deriving 33 3D landmarks and background segmentation masks on the entire body from RGB video frames using the Blazepose study. This application classifies poses and counts repetitions.
To build it, you need:
1) Collect sample images of target exercises and use them to predict the pose;
2) Convert the obtained pose landmarks into a representation suitable for the kNN classifier and form a training set using Colab;
3) Perform the classification itself, followed by counting repetitions (using Colab).