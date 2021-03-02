# Asynchronous Convolutional Networks for Object Detection in Neuromorphic Cameras

Code for the CVPR2019 Event-based Vision Workshop paper "A Differentiable Recurrent Surface for Asynchronous Event-Based Data"<br>
Authors: Marco Cannici, Marco Ciccone, Andrea Romanoni, Matteo Matteucci

### Citing:
If you use this work for research, please cite our accompanying CVPR2019 Event-based Vision Workshop paper:
```
@inproceedings{cannici2019asynchronous,
  title={Asynchronous Convolutional Networks for Object Detection in Neuromorphic Cameras},
  author={Cannici, Marco and Ciccone, Marco and Romanoni, Andrea and Matteucci, Matteo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```

### Requirements
- TensorFlow 1.4.0
- Cython extensions: `build_setup.sh` 

You can create a conda environment to run the code as it follows:
```
conda create -n aync-ev-cnn python=3.6`
conda activate aync-ev-cnn
conda env update -f=requirements.yml
python cython_setup.py build_ext --inplace
```

### Run scripts

- To check event layers equivalence (no dataset or checkpoint required):<br>
`python src/scripts/test_correctness.py`

- To run network predictions on a dataset (select the proper .yml file):<br>
   - Unzip under `data/` the [N-Caltech101 data and checkpoint](https://polimi365-my.sharepoint.com/:f:/g/personal/10425666_polimi_it/Evq7q4F5KG9Fq-faVuTLqucBbo7zp8_ZadVsD7fKRboJgQ?e=mg6N3v)
   - `python src/scripts/run_networks.py -c configs/efcn_event.yml`
