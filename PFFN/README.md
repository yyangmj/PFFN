# Battery RUL Early Prediction

This project is for paper "  PFFN: A Parallel Feature Fusion  Network for Remaining Useful Life Early Prediction  of Lithium-ion Battery  "

## Getting Started

The original data is available at https://data.matr.io/1/, download the data and put them into folder /Data.

### Prerequisites

1.After downloading the data, run  BuildPkl_Batch1.py, BuildPkl_Batch2.py and BuildPkl_Batch3.py to extract the data for training and test.

```python
python BuildPkl_Batch1.py
python BuildPkl_Batch2.py
python BuildPkl_Batch3.py
```

2.Run Load_Data.py to delete bad battery data.

```python
python Load_Data.py
```

Note: BuildPkl_Batch*.py and Load_Data.py are provided by author, small changes are made. 
Original code: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

### Feature Extraction

feature_selection.py provide the implementation of feature extraction, clustering and selection.

```python
python feature_selection.py
```

### Data processed for model

data_process_for_model.py integrates two parts of features to provide input data for the model.

```python
python data_process_for_model.py
```

### Proposed PFFN model with Bayesian Optimization

PFFN.py includes PFFN model implementation with Bayesian Optimization, run it to obtain the best hyperparameter for the model.  Then, comment out the code for Bayesian optimization and rerun PFFN.py to get batter prediction result.

```python
python PFFN.py
```
