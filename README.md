# CTP_Phase2
![architecture](https://github.com/baejustin/CTP_Phase2/assets/16787987/9f5d2377-4458-48f0-8fc4-b444a179a935)

This code is an implementation of our most recent CTP paper. 


# Usage
1. Clone the repo



2. Put all the data in `DATA/*`. The sub directories are already there for your reference.



3. Train the model
```
python train.py
```
Data in `DATA/labeled` and `DATA/unlabeled` will be used for training. <br>


You can change the training configurations via argument options. Check inside the code `train.py` to explore different options. e.g. `python train.py --flip_prob 0.2 --save_ckpt_freq 5000`.
Training intermediates (loss log, visuals, checkpoint, etc) will be saved inside the newly created experiment folder `CTP_[YYMMDD_HHMMSS]`.
Given that our project involves quite a lot of losses to keep track of (parameter maps, segmentation, etc), we use tensorboard for easier tracking and visualization.



4. Test the model
```
python test.py --experiment_name CTP_[YYMMDD_HHMMSS] 
```
Data in `DATA/labeled_val` and `DATA/unlabeled_val` will be used for testing. <br>


Inference results will be saved `CTP_[YYMMDD_HHMMSS]/test_result`.
