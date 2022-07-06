# Saliency Detector for Biochemical Education Media

Visual representations of biochemical processes are important tools for biochemistry educators to build mental representations of various phenomena. These representations are made up of several interacting units called features. Representations seek to show how these distinct features interact and change to create the overall phenomena

One key metric to assess the effectiveness of such representations in building understanding is the saliency of the different features present in the representation. Saliency refers to how noticeable a feature is in comparison to the rest of the representation.

This projects aims to create a tool that use prevalent computer vision and machine learning techniques to create a computational model that extracts the features in a representation and identifies the most salient one


## Setup
To install dependencies execute the following command in a terminal

```
pip install -r requirements.txt
```

## Usage
Once setup execute the following command to run saliency detection script

python sal_detect_2.py [path_to_input_file]


