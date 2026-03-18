Currently I have trained a model from the hockey AI, the issue is that pwhl is different in how it looks and the size of the images 

So i have retrained the model and created best.pt, now I will annotate pwhl photos and add those in. 

17th 
Built new model with annotations, uploaded to git 

18th 
talked with james ?? 
He recmomended using a Kalman filter to track when players go behind each other. 
https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
https://medium.com/@siromermer/predicting-objects-motion-with-kalman-filter-and-fast-algorithm-2278c551670b
https://github.com/marwankefah/Kalman_Tracking_Single_Camera
https://pdfs.semanticscholar.org/8544/46e26b6ec08a8f012cf7ba4750d6c83e42a0.pdf
https://ieeexplore.ieee.org/document/9355351

Ok so just looking at puck tracking with kalman, 
1 predict location
then if we have the location draw circle over ball
2 update kalman 

---- 
Some ideas from talking to mr gpt 
dynamically cull spectators, this could be done through detecting white regions and culling above it. This should happen after detecting players so we dont cull them. 

bytetrack can work well for tracking player ids, I want to swap to squares around the players rather than eclipess at the feet. 
This could help focus the paper into more of a improvements on hockeyai model.... maybe 

"""
Introduction
  Problem: tracking in hockey is hard (occlusion, speed)
  
  Gap: detection quality impact underexplored
  
  Contribution: improved detector → better MOT

Related Work
Method
Experiments/results
Analysis
Conclusion/closing remarks 

"""



