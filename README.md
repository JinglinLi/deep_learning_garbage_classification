# deep_learning_garbage_classification
deep learning project : garbage classification - MobileNet transfer learning

### Goal : 
- classify webcam caputured image into one of 12 garbage classes 
### Methods :
- Data used for training :
    - https://www.kaggle.com/mostafaabla/garbage-classification
    - 15,150 images in total 
    - 12 different classes of household garbage
    - paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, and trash 

- Model : 
    - transfer learning of pretrained MobileNet deep learning model 
- Data used for test :
    - webcam image (tool for capturing webcam image : (c) 2021 Malte Bonart)
### Results : 
- Model training results: train accuracy: 0.9386, val_accuracy: 0.9440)
- Usage :
  - go to folder my_imageclassifier
  - run python capture.py
  - press space to make photo
  - terminal returns prediction of object class prediction (original MobileNet) and garbage class (transfer learning model)
  - press q to terminate



