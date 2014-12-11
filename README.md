matlab-cv
=========
**matLearn_CV.m** performs cross-validation for both classification and regression problems.

**Collaboration**: Matt and Issam combined CV for regression and classification.

**Value-added**: 
- k-Fold cross-validation
- leave-one-out cross-validation
- option to randomly shuffle dataset before performing CV
- simple early-stopping option: stops grid search when the error starts increasing after it has decreased at least once.
- loss functions: squared-error, absolute error, and zero-one loss.


## Matt's demo files for CV regression
- **html_demo_CV_KNN/demo_CV_KNN.html**: Demo file in readable format
- **data/mineral-assay.mat**: Demo dataset. Sensor readings taken of various rock, including weight, XRF, and electromagnetic sensors (68 features total). The target is to predict the amount of aluminum in the rock.
- **demo_CV_KNN.m**: Demo code of Cross-Validation for regression models, using KNN as an example.
- **matLearn_regression_KNN_Dirks.m**: My version of KNN, made for the purpose of demoing Cross-Validation.