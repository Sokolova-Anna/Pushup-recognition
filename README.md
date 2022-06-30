# Pushup-recognition

MediaPipe Pose is an ML solution for tracking high-quality body poses, deriving 33 3D landmarks and background segmentation masks on the entire body from RGB video frames using the Blazepose study. This application recognizes push-ups and counts repetitions on the video. Repository contains training set (csv files), image samples with and without pose landmarks.

The k-NN algorithm used for pose classification requires a feature vector representation of each sample and a metric to compute the distance between two such vectors to find the nearest pose samples to a target one.

To count the repetitions, the algorithm monitors the probability of a target pose class.
