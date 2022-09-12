# Adroit Manipulation Platform

Adroit manipulation platform is reconfigurable, tendon-driven, pneumatically-actuated platform designed and developed by [Vikash Kumar](https://vikashplus.github.io/) during this Ph.D. ([Thesis: Manipulators and Manipulation in high dimensional spaces](https://digital.lib.washington.edu/researchworks/handle/1773/38104)) to study dynamic dexterous manipulation. Adroit is comprised of the [Shadow Hand](https://www.shadowrobot.com/products/dexterous-hand/) skeleton (developed by [Shadow Robot company](https://www.shadowrobot.com/)) and a custom arm, and is powered by a custom actuation system. This custom actuation system allows Adroit to move the ShadowHand skeleton faster than a human hand (70 msec limit-to-limit movement, 30 msec overall reflex latency), generate sufficient forces (40 N at each finger tendon, 125N at each wrist tendon), and achieve high compliance on the mechanism level (6 grams of external force at the fingertip displaces the finger when the system is powered.) This combination of speed, force, and compliance is a prerequisite for dexterous manipulation, yet it has never before been achieved with a tendon-driven system, let alone a system with 24 degrees of freedom and 40 tendons.

## Mujoco Model
Adroit is a 28 degree of freedom system which consists of a 24 degrees of freedom **ShadowHand** and a 4 degree of freedom arm. This repository contains the Mujoco Models of the system developed with extreme care and great attention to the details.


## In Projects 
Adroit has been used in a wide variety of project. A small list is appended below. Details of these projects can be found [here](https://vikashplus.github.io/). 
[![projects](https://github.com/vikashplus/Adroit/blob/master/gallery/projects.JPG)](https://vikashplus.github.io/)
## In News and Media
Adroit has found quite some attention in the world media. Details can be found [here](https://vikashplus.github.io/news.html)

[![News](https://github.com/vikashplus/Adroit/blob/master/gallery/news.JPG)](https://vikashplus.github.io/news.html)


## Citation
If the contents of this repo helped you, please consider citing 

``` 
@phdthesis{Kumar2016thesis,
    title    = {Manipulators and Manipulation in high dimensional spaces},
    school   = {University of Washington, Seattle},
    author   = {Kumar, Vikash},
    year     = {2016},
    url      = {https://digital.lib.washington.edu/researchworks/handle/1773/38104}
}
```
