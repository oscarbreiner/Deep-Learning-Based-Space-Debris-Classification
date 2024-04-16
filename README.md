# Deep Learning Based Space Debris Classification in Low Earth Orbit Using Space-Borne Radar Simulations

Technical University Munich Bachelor Thesis in Collaberation with Satellite Startup Actlabs

Author: Oscar Breiner

## Brief Simulation Overview

### Simulation Setup implemented using Matlab

This thesis addresses the challenge of classifying centimeter-sized space debris in Low Earth Orbit (LEO) using radar technology and deep learning techniques, given the increasing threat they pose to active space infrastructures. We propose deploying space-borne radar systems to overcome the limitations of conventional space debris observation methods in detecting small debris particles. Therefore, we developed a simulation pipeline designed to mimic a monostatic radar system. This system emits high-frequency radar radiation at 94 GHz and captures the reflections from rotating debris. Since this operation is deployed within the optical scattering region, it enables us to extract spatial characteristics of the target object from the captured signal. The target objects are composed of fully conducting materials and are categorized into four distinct classes: cones, cylinders, spheres, and circular plates, each containing instances in varying shapes and sizes.

<p align="center">
  <img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/93127cee-1213-4a92-89c3-1d0ecb6f16f4" width="50%">
</p>

### Example Radar Echo: Raw Amplitude Signal vs Processed Decibel Signal

Data samples include radar echo, additive gaussian noise, and fluctuations on radar level to simulate micro vibrations of each target object.

<p align="left">
  <img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/6937856d-3e0f-4888-90cd-32c487d8c22e" width="49%">
  <img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/4f74f183-0bec-47be-93fb-a97959d44e83" width="49%">
</p>

### Example Radar-Cross-Section (RCS) Signatures as 3D Plot in **dBsm**

<table>
  <tr>
    <td>Tall Cylinder</td>
    <td>Wide Cone</td>
    <td>Sphere (r = 0.05m)</td>
    <td>Plate (r = 0.05m)</td>
  </tr>
  <tr>
    <td><img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/a65f32ee-7a94-4378-bb41-eb76dbb42635" alt="3d_tall_cyl"></td>
    <td><img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/fd096b8f-4353-45d5-9d65-756129087e14" alt="3d_wide"></td>
    <td><img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/9cbf2cc0-b27b-45e3-a15c-9c963fe41a58" alt="3d_tall_sphere"></td>
    <td><img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/c2854c4e-0a90-4d8d-84d6-090b852876da" alt="3d_large_plate"></td>
  </tr>
</table>


## Brief Overview of Deep Learning Experiments

The resulting dataset of captured signals is used to evaluate the application of deep learning based space debris classification. The examined deep learning architectures include Multilayer Perceptron (**MLP**), Long Short-Term Memory (**LSTM**), Residual Network (**ResNet**), and **Transformer** models equipped with either Sparse-Attention or Full-Attention mechanisms. Experimentation highlighted the impact of signal processing. Decibel transformation leads in many cases to improved accuracy, better generalization, and faster convergence. However, bidirectional LSTM and Transformers are exceptions to this trend, demonstrating the capability to effectively process either raw signal samples or decibel signals.

<p align="center">
  <img src="https://github.com/oscarb-TUM/Deep-Learning-Based-Space-Debris-Classification/assets/82709788/6d09744b-f9b3-480e-93f8-92fecfd5ec7a" width="60%">
</p>

Through extensive experimentation and hyperparameter tuning, we achieved accuracy results in classification close to 95%. This threshold is linked to similar radar cross section (RCS) signatures from certain observational angles of tall cones and cylinders (read thesis paper to find out more about misclassification problem).

<div align="center">

| Model     | Accuracy | AvgPrec | Signal type | LR     | Optim. | Scheduler | Epoch |
|-----------|----------|---------|-------------|--------|--------|-----------|-------|
| **LSTM**         | 94.96%   | 98.57%  | decibel     | 0.01   | Adam   | step      | 10    |
| **Bi-LSTM**       | 94.83%   | 98.17%  | raw         | 0.001  | Adam   | step      | 30    |
| **ResNet**        | 94.76%   | 98.39%  | decibel     | 0.001  | Adam   | step      | 10    |
| **Bi-LSTM**       | 94.72%   | 98.46%  | decibel     | 0.01   | Adam   | step      | 10    |
| **FullTRAN**      | 93.93%   | 98.18%  | raw         | 0.0001 | Adam   | plateau   | 50    |
| **SparseTRAN**    | 93.33%   | 97.46%  | decibel     | 0.0001 | Adam   | step      | 17    |
| **FullTRAN**      | 89.84%   | 95.42%  | decibel     | 0.0001 | Adam   | plateau   | 50    |
| **MLP**           | 80.3%    | 86.87%  | decibel     | 0.001  | Adam   | step      | 10    |

</div>

### Robustness Evaluation

The simulation data also factors in real-world challenges in the form of **additive Gaussian noise** and **RCS signature fluctuations** due to object vibrations. For further robustness evaluations, we tested for radar-specific noise on pretrained models by distorting the test set with various types of **signal occlusion, clutter, and sensor saturation**. The results underscore the effectiveness of applying decibel transformations before feature extraction. Models trained on decibel signals appear to benefit from the noise reduction capabilities of the logarithmic scale, improving clarity in radar distortions like clutter and saturation. Yet again, Full-Attention Transformer trained on raw signals defies this trend of better decibel efficacy, showcasing unmatched resistance to occlusion and various frequency-based clutter scenarios, outperforming every other examined model.

**Occlusion:** Implemented through Random Point Dropouts and Random Window Dropout techniques.

**Sensor Saturation:** Assessed using Percentile Saturation methods.

**Clutter Noise:** Evaluated by introducing Random Peaks (Anomalies) and Sinusoidal Clutter signals.

## Impact

Overall, this research thesis highlights the potential of deep learning for classifying radar targets, improving the surveillance of space debris and improving the safety of space op- erations. It showcases the application of space-borne radar systems and emphasizes the capabilities of deep learning models in processing complex radar signals.




