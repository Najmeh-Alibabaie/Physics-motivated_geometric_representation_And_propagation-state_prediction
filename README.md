# 
# 

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

