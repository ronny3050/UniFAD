# UniFAD: Universal Face Attack Detection
By Debayan Deb, Xiaoming Liu, and Anil K. Jain

<a href="https://arxiv.org/abs/1908.05008"><img src="https://raw.githubusercontent.com/ronny3050/UniFAD/master/assets/cover.png" width="80%"></a>

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" align="right" width="100"/>

A tensorflow implementation of [UniFAD](https://arxiv.org/pdf/2104.02156), a state-of-the-art face attack detector that can detect adversarial, digital, and physical face attacks. UniFAD learns joint representations for coherent attacks, while uncorrelated attack types are learned separately.

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" width="25"/> Tensorflow release
Currently this repo is compatible with Tensorflow 1.15.5.

## <img src="https://cdn-icons-png.flaticon.com/512/2103/2103832.png" width="25"/> Training

### JointCNN: A Single Binary CNN Detector
    
The configuration files for training are saved under ```config/``` folder, where you can define the dataset prefix, training list, model file, attack setting and other hyper-parameters. Use the following command to run the default training configuration:
    ``` Shell
    python train_binary_detector.py config/joint_cnn.py
    ```

The command will create an folder under ```log/JointCNN/``` which saves all the checkpoints, test samples and summaries. The model directory is named as the time you start training.

*Note: training JointCNN requires one-hot labels (0 for real and 1 for all attacks). Please refer to `config/train_joint.txt` for reference.*  

### ChimneyCNN: Early Layers + Branching
The configuration files for training are saved under ```config/``` folder, where you can define the dataset prefix, training list, model file, attack setting and other hyper-parameters. Use the following command to run the default training configuration:
    
    ``` Shell
    python train_chimney_detector.py config/chimney.py
    ```

The command will create an folder under ```log/Chimney/``` which saves all the checkpoints, test samples and summaries. The model directory is named as the time you start training.

*Note: training ChimneyCNN requires class-wise labels (0 for real and 1,2,3,... for different attack types). Please refer to `config/train_chimney.txt` for reference.*  

## <img src="https://www.marc-jekel.de/media/icon_hu175a232152e93f3c6bae4698ffed542c_28254_512x512_fill_lanczos_center_2.png" width="25"/> Testing
* Run the test code in the following format:
    ```Shell
    python test_binary_detector.py <PATH_TO_SAVED_MODEL>
    python test_chimney_detector.py <PATH_TO_SAVED_MODEL>
    ```
* For example, if you want to use the pre-trained model, create a folder called `models`, download the model and unzip it into `models/JointCNN` or `models/Chimney` folder. Then, run 
    ```Shell
    python test_binary_detector.py models/JointCNN
    ``` 

## <img src="https://image.flaticon.com/icons/png/512/816/816167.png" width="25"/> Pre-trained Models
##### JOINT-CNN MODEL: 
[Dropbox](https://www.dropbox.com/s/5hgzxhtftlf6fu8/JointCNN.zip?dl=0)
##### CHIMNEY-CNN MODEL: 
[Dropbox](https://www.dropbox.com/s/h43mw561ilgi8or/Chimney.zip?dl=0)
