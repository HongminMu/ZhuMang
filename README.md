# **Wearable Visuomotor Navigation and Mobility Assistance Device for the Partially Sighted and Visually Impaired in Unfamiliar Indoor Scenes** 👓 

## **🌟 Project Overview**  
This project, formally titled **"Wearable Visuomotor Navigation and Mobility Assistance Device for the Partially Sighted and Visually Impaired in Unfamiliar Indoor Scenes"**, enhances **mobility and independence** for visually impaired individuals by enabling **navigation in unfamiliar indoor environments without relying on dense pre-built maps**.  

Overview of our device and navigation system :
![12334](https://github.com/user-attachments/assets/a1a99db3-299d-4ad8-83e5-a10197fc02fd)

🚀 **Key Features:**  
✅ **SLAM-free navigation** – Uses **Topological Semantic Mapping (TSM)** instead of conventional dense mapping.  
✅ **Text-based Localization** – Matches detected text from environmental signage with TSM nodes for precise positioning.  
✅ **Binocular Stereo Vision** – Enhances depth estimation to assist in navigation toward key landmarks.  
✅ **AI-Driven Obstacle Detection** – Uses deep learning for real-time scene awareness.  
✅ **Multi-Device Wearable Interaction** – Smart glasses process visual data, while wristbands provide haptic feedback.  

---

## **🏆 Awards & Achievements**  
This project was highly recognized in two prestigious robotics competitions:

- 🥇 **First Prize** - **World Robot Contest - Tri-Co Robots Challenge** (August 2024)  
  - **Ranked Top 3 out of 19 teams**  
  - **Project Name**: *OpenAEye: an Assistive Glasses for the Visually Impaired*  

![image](https://github.com/user-attachments/assets/05ec16c9-2c00-4bf0-80ff-181da953cfa5)

- 🥈 **National Runner-up** - **China Graduate Robotics Innovation Design Contest** (September 2023)  
  - **Ranked 2nd out of 1,177 teams**  
  - **Project Name**: *Wearable Assistive Device for the Visually Impaired to Navigate in Unfamiliar Indoor Environments*
 
![333](https://github.com/user-attachments/assets/1ca68f01-71f6-4268-87a5-5b560f90ba76)

## **🎥 Video Demonstration**  
You can download the demo videos *(within 25MB each)* here!

[Volunteer 1](https://github.com/HongminMu/ZhuMang/blob/main/volunteer1.mp4)  
[Volunteer 2](https://github.com/HongminMu/ZhuMang/blob/main/volunteer2.mp4)

Volunteer 1
![image](https://github.com/HongminMu/ZhuMang/assets/57067148/7820972f-91ab-4a45-aa9f-684060dc663b)

Volunteer 2
![image](https://github.com/user-attachments/assets/9a6beec2-f583-49e0-ba18-19e6ffacf9b7)

## **🚀 Code Release**  
✅Code for Obstacle Avoidance.

This repository provides a modified version of [YOLACT++](https://github.com/dbolya/yolact) for instance segmentation of passable areas.

### Step 1: Install YOLACT++
Clone the original YOLACT++ repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/dbolya/yolact.git
cd yolact

# Install dependencies
pip install -r requirements.txt

# Compile and install DCNv2
cd external/DCNv2
python setup.py build develop
```

### Step 2: Integrate Demo Code 
Copy `zhendong.py` to the YOLACT++ root directory.

```bash
# Place `zhendong.py` in the root directory
cp path_to_zhendong.py yolact/
```

### Step 3: Download Pretrained Weights
Download our custom-trained weights from [this link](https://drive.google.com/file/d/10LK7Iq2vLNiBRs2EWcySygQpdUQFpC_7/view?usp=drive_link) and place them in the `weights` folder.
Replace the `data/config.py` file with our custom configuration.

```bash
# Move the weights to the correct directory
mkdir -p weights
cp path_to_weights.pth weights/

# Replace configuration file
cp path_to_custom_config.py yolact/data/config.py
```

### Step 4: Running demo
Run the modified `zhendong.py` script with the following command:

```bash
python zhendong.py \
    --trained_model=weights/yolact_coco_custom_91_16000.pth \
    --config=yolact_coco_custom_config \
    --score_threshold=0.35 \
    --top_k=6 \
    --video=1 \
    --display
```

### Visualization without Vibration Modules
If you want to visualize the obstacle avoidance process without using vibration modules, use `zhendong_visualization.py`:

```bash
# Place `zhendong_visualization.py` in the root directory
cp path_to_zhendong_visualization.py yolact/

# Run with the same parameters
python zhendong_visualization.py \
    --trained_model=weights/yolact_coco_custom_91_16000.pth \
    --config=yolact_coco_custom_config \
    --score_threshold=0.35 \
    --top_k=6 \
    --video=1 \
    --display
```

## Notes
- To enable obstacle avoidance with vibration modules, purchase two Vibration Modules (LILYGO® T-WATCH 2021) and configure the serial communication according to the comments in `zhendong.py`.
- Make sure to correctly set up the serial communication if using the vibration modules.
- The visualization script allows you to see how the obstacle avoidance algorithm works without requiring hardware.

For more details, please check the code comments or contact us for further support.

Our full code will be made publicly available soon! Stay tuned for updates. 🌍💡  

## **📩 Contact**  
For inquiries, feel free to reach out via email at **hongmin_@163.com**, specifying the purpose of your request.  
We can provide hardware specifications and software code. Contributions, discussions, and collaborations are highly welcome!  

## **📚 Related Work**  
This work builds upon our previous research on obstacle avoidance systems for visually impaired users. Specifically, we extend our prior work on **dynamic obstacle avoidance** using **instance segmentation** to improve the **navigation experience in indoor environments**.  

You can find our previous work referenced below:  

> **Mu, Hongmin and others**, *Dynamic Obstacle Avoidance System Based on Rapid Instance Segmentation Network*, **IEEE Transactions on Intelligent Transportation Systems**, **2024**, **Vol. 25**, **No. 5**, **Pages 4578-4592**, [DOI: 10.1109/TITS.2023.3323210](https://doi.org/10.1109/TITS.2023.3323210).

We thank the authors of YOLACT++ for providing a fast and accurate instance segmentation algorithm, which enables **real-time segmentation** of **passable areas** in our project:
> **Bolya, Daniel and others**, *YOLACT++ Better Real-Time Instance Segmentation*, **IEEE Transactions on Pattern Analysis and Machine Intelligence**, **2022**, **Vol. 44**, **No. 2**, **Pages 1108-1121**, [DOI: 10.1109/TPAMI.2020.3014297](https://doi.org/10.1109/TPAMI.2020.3014297).


```bibtex
@ARTICLE{Mu2024Dynamic,
  author={Mu, Hongmin and Zhang, Gang and Ma, Zhe and Zhou, Mengchu and Cao, Zhengcai},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Dynamic Obstacle Avoidance System Based on Rapid Instance Segmentation Network}, 
  year={2024},
  volume={25},
  number={5},
  pages={4578-4592},
  keywords={Feature extraction;Task analysis;Collision avoidance;Real-time systems;Distance measurement;Cameras;Semantics;Obstacle avoidance;instance segmentation;mobility assistance;indoor navigation},
  doi={10.1109/TITS.2023.3323210}
}

@ARTICLE{Bolya2022,
  author={Bolya, Daniel and Zhou, Chong and Xiao, Fanyi and Lee, Yong Jae},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={YOLACT++ Better Real-Time Instance Segmentation}, 
  year={2022},
  volume={44},
  number={2},
  pages={1108-1121},
  keywords={Prototypes; Real-time systems; Image segmentation; Object detection; Detectors; Task analysis; Shape; Instance segmentation; real time},
  doi={10.1109/TPAMI.2020.3014297}
}

