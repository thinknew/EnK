# EnK: Encoding time-information in convolution
[Singh, A.K. and Lin, C.T., 2020. EnK: Encoding time-information in convolution. arXiv preprint arXiv:2006.04198](https://arxiv.org/pdf/2006.04198.pdf).

# Requirements

- Python == 3.7 or 3.8
- tensorflow == 2.X (both for CPU and GPU)
- PyRiemann >= 0.2.5
- scikit-learn >= 0.20.1
- matplotlib >= 2.2.3

# How to run

- Input Data Format: Number of EEG Channels x Number of Samples X Number of Trials for EEG data and Labels as a vector. See [testData.mat](https://github.com/thinknew/BCINet/tree/main/testdata) for references with sampling rate of 400 Hz.
- Provide input data related information in 'op.py' such as path, sampling rate, number of classes, etc.
- Execute the following line of code

```
python main.py
```

# Models implemented/used
- EEGNet [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013) 
- DeepConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[3]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)

EEGNet, DeepConvNet, and ShallowConvNet implementation are based on EEGNet repo [[4]](https://github.com/vlawhern/arl-eegmodels)
