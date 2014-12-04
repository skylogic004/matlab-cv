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


## Matt's files (for testing only - will not be submitted in the end)
- **matLearn_regression_CV.m**: Cross-validation for regression
- **matLearn_regression_CV_testModel_intParams.m**: Test regression model (uses fake int parameter)
- **matLearn_regression_CV_testModel_stringParams.m**: Test regression model (uses fake string parameter)
- **demo_CV_regression_intParams.m**: CV Regression demo using fake int parameter
- **demo_CV_regression_stringParams.m**: CV Regression demo using fake string parameter

