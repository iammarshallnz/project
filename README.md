
# Detecting game elements within ice hockey

install python version 3.11 
and verify with
```bash
py -0
```

create a venv 
```bash
py -3.11 -m venv venv
```
activate
```bash
.\venv\Scripts\Activate
```
then install all dependancys 
```bash
pip install opencv-python numpy ultralytics filterpy
```
Then check and install what torch you need
https://pytorch.org/get-started/locally/

run py hockey.py

## Settings
Within hockey.py there are some variables you may want to change 
These exist near the top of the file
```python
video = "./highlight.mp4"

process_noise = 0.5

similarity_thresh = 0.7
```

## Other files
There exists other files that can be used for various reasons 

~compare_models.py~ is used to run the validation of a model on the same data stored within labeled


~get_frames.py~ is used to take frames out of a mp4 every 30 seconds and save it to a folder. I used the frames in new_frames

~train.py~ is what I used to train the fine-tuned model from labeled data within labeled. 

## Labeled

Data was annotated using label-studio, as the original model uses more than what we wanted the data.yaml must have 

```bash
nc: 7
names: ['centerIce','faceoff','goal','goaltender','player','puck','referee']
```
