# Building Footprint Detection 

Building footprint detection from satellite/aerial images with Mask R-CNN and image matting (KNN matting, closed-form matting and Grabcut).

![overview](.\overview.png)

## Usage

Run `python detect.py` for detection, `python evaluate.py` for evaluation, and `python regularize.py` for regularizing the footprint. You may have to change the hardcoded paths in these scripts.

## Dataset

You can download the [cropped aerial image tiles and raster labels](https://study.rsgis.whu.edu.cn/pages/download/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip) from [WHU Building Dataset](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html).

## File Structure

```
│  detect.py
│  evaluate.py
│  mrcnn_demo.ipynb
│  
├─matting
│      closed_form_matting.py
│      grabcut.py
│      knn_matting.py
│      solve_foreground_background.py
│      
├─models
│      models_here.txt
│      
├─mrcnn
│      config.py
│      model.py
│      parallel_model.py
│      utils.py
│      visualize.py
│      __init__.py
│      
└─polygon
        regularize.py
```

