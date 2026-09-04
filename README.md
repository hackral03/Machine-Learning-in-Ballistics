# Machine-Learning-in-Ballistics
## Introduction
Ballistic limit is a key metric to determine ballistic performance of a material system for application as body armour. It is defined as the velocity with a particular probability of perforation. Ballistic limit velocities are defined based on application and research interest. A researcher studying Behind Armour Blunt Trauma (BABT) might be interested in a different ballistic limit velocity than a researcher who is studying material failure. 

V50 is the most commonly used ballistic limit velocity and will be the main focus in this report. It is defined as the velocity where probability of perforation of armour panel is 50%. Other ballistic velocity limits like V10 and V99, have also been of interest in some research. To determine these ballistic limit velocities, ballistics tests are conducted on a sample according to well defined testing standards and ballistic limit velocity is calculated using statistical methods like Probit analysis.

Performing ballistics tests require a lot of resources, high costs and long timelines. Researchers around the world are trying to develop non-destructive testing methods to predict ballistic limit velocities and other metrics. Machine Learning methods have emerged as the major tools to develop such methods. This repository will investigate some machine learning models and compare their performance for application in ballistics.
Machine learning methods are becoming a tool of interest for research in ballistics and composites because of their abilities in pattern recognition, reinforcement learning and prediction applications. 

Deep learning is sub-branch of machine learning, inspired by neural networks inside human brain, to learn and generate complex patterns and datasets. They have huge applications in Generative AI and Natural Language Processing (NLP). For ballistics tests, primarily these neural network models are being researched: Generative Adversarial Networks (GANs), Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs). The objectives for using these are:

- Develop non-destructive methods to find ballistic limit.
- Predict ballistic performance at high velocities.
- Reduce cost of material characterization and testing.
- Speed up the process of design prototyping.
- Predict damage patterns for composites.

## Probit Regression
The phenomenon of perforation of armour can be modelled as a stochastic event defined by a probability density function which gives probability of perforation as a function of impact velocity. For application of probit method, it is assumed that the probability density function follows the normal or Gaussian law.

Hypothesis: The probability density function for perforation follows the normal or Gaussian law.

$$
\text{pdf} = \frac{1}{sd\sqrt{2\pi}} \, e^{-\frac{(v - V50)^2}{2\,sd^2}}
$$

## Cunniff Model
In 1999, Phillip M. Cunniff [15] used dimensional analysis to relate system and projectile characteristics to the design parametrs of an armour, namely V50 and areal density of the system. A system was developed which related system characteristics – tensile strength(σ), tensile strain(ε), density(ρ) and Elastic modulus (E); projectile characteristics – projectile surface area (Ap) and projectile mass (mp); and armour design parameters – V50 and areal density (Ad).

$$
\frac{V_{50}}{(U^*)^{1/3}} = K_1 \left(\frac{A_d A_p}{m_p}\right)^n
$$

where 

$$
U^* = \frac{\sigma \varepsilon}{2 \rho} \sqrt{\frac{E}{\rho}}
$$
