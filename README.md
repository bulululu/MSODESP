# MSODESP: An Explanatory Multiscale Neural Differential Equation for Handling Missing Data Multi center Sepsis Prediction


## train the model

###  run model by using internal center ***mimic3cv mimiciv*** and external center ***eicu***
```shell
python main.py --merge_time_window 8 --predict_time_window 8 --adpot_time_window 24 --threshold_missing 0.5
```

---

## Data details

| Data center   | eicu  | xjtu | mimic3cv | mimiciv |
|---------------|-------|------|----------|---------|
| Sample number | 23454 | 4917 | 9128     | 33495   |


## Data division

|            | TrainCenter      | External Center 1 | External Center 2 |
|------------|------------------|-------------------|-------------------|
| DataCenter | mimic3cv, mimic4 | eicu              | xjtu              |

## Baseline 

| model | dataset | trainset | validset | testset | eicu   | xjtu   |
|-------|---------|----------|----------|---------|--------|--------|
|       | rate    | 0.7      | 0.1      | 0.2     | 1.0    | 1.0    |
| lstm  | auc     | 0.9401   | 0.9298   | 0.9212  | 0.8970 | 0.8566 |

---

- [ ] dwa
- [x] daw
- 