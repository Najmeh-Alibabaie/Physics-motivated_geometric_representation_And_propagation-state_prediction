# Physics-Motivated Geometric Representation for Propagation-State Prediction in 3D UAV Communication Environments
## **Author(s):** 
Najmeh Alibabaie , Antonello Calabrò, Eda Marchetti, 
**Journal**: IEEE Open Journal of the Communications Society, 
**DOI**: 10.1109/OJCOMS.2026.3755141

## Abstract.
Maintaining reliable communication in air-to-ground networks requires accurate modeling of the physical environment and radio propagation, as rapid spatial variations in channel conditions significantly affect communication quality. Existing approaches either rely on computationally intensive propagation models or employ generic environment representations. A key challenge is to develop environment representations that are computationally efficient while preserving the physical characteristics relevant to radio propagation. 
This paper introduces a physics-motivated geometric representation that links environment geometry to radio propagation for characterizing line-of-sight (LOS), non-line-of-sight (NLOS), and blocked propagation states.
Unlike conventional geometry descriptors based on generic environmental statistics, the proposed representation  captures structural features associated with blockage, multipath interaction, and spatial confinement.
The effectiveness of the proposed representation is evaluated through a geometry-to-propagation mapping framework across diverse urban Unmanned Aerial Vehicle (UAV) communication scenarios. Experimental results achieve an average accuracy of 94.02\% while maintaining computational efficiency. Ablation experiments and interpretability analysis demonstrate that the proposed representation preserves complementary physical propagation mechanisms and contributes to improved state characterization.

## Description:
This repository contains the implementation accompanying our work on
**Physics-Motivated Geometric Representation for Propagation-State Prediction in 3D UAV Communication Environments**.

The proposed approach extracts a compact set of geometric descriptors directly
from 3D CAD environments and uses them to classify each wireless link into
three propagation states:

- **Blocked**
- **LOS (Line-of-Sight)**
- **NLOS (Non-Line-of-Sight)**

Instead of processing complete 3D building models or relying directly on
high-dimensional spatial representations, the method constructs a compact
representation that captures the geometric factors most relevant to
propagation-state transitions.

---

## Method Overview

The overall pipeline is:

```text
3D CAD Environment
        │
        ▼
LOS-Oriented Local Coordinate System
        │
        ▼
3D Surface Extraction
        │
        ▼
Relevant Surface Selection
        │
        ▼
Physics-Motivated Descriptor Construction
        │
        ▼
Compact Geometric Representation Φ
        │
        ▼
XGBoost Propagation-State Classifier
        │
        ▼
Blocked / LOS / NLOS

