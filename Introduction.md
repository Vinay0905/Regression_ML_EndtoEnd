# Complete Project Theory: Housing Price Prediction (End-to-End Regression ML)

This document provides a comprehensive theoretical background and architectural overview of the **Housing Regression MLE** project. It justifies the technical decisions made throughout the lifecycle of the project, from problem formulation and model selection to deployment infrastructure. It also includes an extensive list of 60 possible interview questions you could face.

---

## 1. Project Formulation: Why Regression?

The goal of this project is to predict housing prices based on various features like location (city, zipcode), dates, and property characteristics. 

### Why Regression instead of Classification?
- **Continuous vs. Discrete:** Housing prices are continuous numerical values (e.g., $450,000, $1,250,500). **Regression** models are specifically designed to predict continuous outcomes. 
- **Classification** algorithms (like Logistic Regression, SVMs, or Random Forest Classifiers) predict discrete class labels (e.g., "Spam" or "Not Spam", "Expensive" or "Cheap"). While we could theoretically group prices into buckets (e.g., Low, Medium, High) and use classification, doing so throws away valuable granularity and precision, making the predictions much less useful.
- **Why not Logistic Regression?** Despite having "Regression" in its name, Logistic Regression is practically a classification algorithm used to predict probabilities of mutually exclusive categories (using the Sigmoid or Softmax function). It cannot output a continuous price prediction spanning from zero to infinity.

---

## 2. Model Selection: Why XGBoost?

We chose **XGBoost (Extreme Gradient Boosting)** as our core predictive model. 

### Why XGBoost over Linear Regression?
Linear Regression assumes a strictly linear relationship between features and the target variable. Housing prices often involve complex, non-linear relationships and intricate feature interactions (e.g., the value of a house heavily depends on the *combination* of zip code and square footage, not just them added together). XGBoost natively captures these non-linearities and interactions.

### Why XGBoost over Deep Learning?
- **Tabular Data:** XGBoost and tree-based ensemble methods routinely outperform deep neural networks on structured, tabular data (like CSVs with columns for zip code, house area, etc.).
- **Data Size & Computation:** Deep learning requires massive amounts of data and compute (GPUs) to train effectively, whereas XGBoost trains very fast on CPUs and handles moderate-sized datasets exceptionally well.
- **Interpretability:** Tree-based models offer direct measures of feature importance, making it much easier to explain *why* a house was priced a certain way compared to a black-box neural network.

### Why XGBoost over Random Forest?
- **Gradient Boosting vs Bagging:** Random Forest builds independent trees in parallel (bagging) and averages them. XGBoost builds trees sequentially, where each new tree specifically tries to correct the residual errors of the previous trees (boosting). This typically leads to lower bias and higher accuracy.
- **Under-the-hood Optimizations:** XGBoost includes advanced regularization (L1/L2) which prevents overfitting, and it features cache-awareness and out-of-core computing, making it exceptionally fast and efficient.

---

## 3. Hyperparameter Tuning: Why Optuna?

For achieving the best performance from XGBoost, hyperparameters (like `learning_rate`, `max_depth`, `n_estimators`) need tuning.

### Grid Search & Random Search vs. Optuna
- **Grid Search** tries every single combination of parameters. It is exhaustive but extremely slow and computationally expensive.
- **Random Search** randomly samples configurations. It's faster than Grid Search but doesn't learn from previous iterations.
- **Optuna** uses **Bayesian Optimization** (specifically the Tree-structured Parzen Estimator, or TPE algorithm). Unlike random search, Optuna looks at the results of past trials to smartly guess which hyperparameters are likely to perform best in the next trial. This leads to finding optimal or near-optimal parameters much faster and with fewer total runs.

---

## 4. Experiment Tracking & MLOps: Why MLflow?

Building ML models is an iterative process. Without a tracking system, data scientists often lose track of which code, data, and parameters produced which model.

### Why MLflow?
- **Experiment Tracking:** It automatically logs hyperparameters, metrics (RMSE, MAE), and code versions for every run of the model. 
- **Model Registry:** It saves the actual model artifacts (the `.pkl` files), allowing us to version models and always be able to revert to our "Best Model."
- **Reproducibility:** If an interviewer asks, "Can you reproduce the exact model you deployed three months ago?", MLflow makes the answer a simple "Yes", because the exact environment, code commit, and parameters are bundled together.

---

## 5. Architectural Pipeline & Deployment Stack

### Data Preprocessing & Target/Frequency Encoding
We use Time-based splitting (Train <2020, Eval 2020-21, Holdout 2022+). This is vital to prevent **Data Leakage** (using future information to predict past events). We use Frequency Encoding for zipcodes and Target Encoding for cities. Encoders are strictly fitted on the training split and applied to the eval/holdout sets.

### Deployment Stack: FastAPI + Streamlit + Cloud Run + Supabase
- **FastAPI:** Used to wrap the inference engine in a REST API. It is incredibly fast (built on Starlette) and provides automatic swagger documentation and validations.
- **Streamlit:** A rapid prototyping UI framework that allows us to build an interactive frontend for the end-user to input house features and get predictions from the API.
- **Supabase:** Acts as our cloud storage (replacing AWS S3) for models and datasets, and a Postgres DB for logging predictions.
- **Google Cloud Run:** A serverless execution environment. Both the FastAPI backend and Streamlit UI are containerized using Docker and deployed here. It scales automatically to zero (saving costs when no one is using it) and instantly scales up during traffic hits.

---

# 60 Mock Interview Questions on this Project

### Section 1: Problem Formulation & Data Handling (1-10)
1. **Walk me through your end-to-end machine learning pipeline.**
*Answer:* My pipeline follows a modular structure: "Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Serve." First, data is loaded and split chronologically (pre-2020 for training, 2020-21 for evaluation, 2022+ for holdout). Preprocessing involves removing deduplications and outliers. Feature engineering includes target encoding for cities and frequency encoding for zipcodes. The model is an XGBoost Regressor tuned via Optuna and tracked using MLflow. Finally, the model is served using a FastAPI backend and consumed via a Streamlit UI, both containerized and deployed on Google Cloud Run with Supabase handling storage and DB.

2. **Why did you frame this as a regression problem and not classification?**
*Answer:* Housing prices are continuous numerical values ($400,000, $1.2M, etc.). Regression is built specifically to output unbounded continuous values. If we used classification, we would have to bucket prices into arbitrary categories (e.g., "cheap", "medium", "expensive"), destroying a wealth of precision and meaning in the target variable.

3. **If you were forced to use classification for this problem, how would you restructure the data?**
*Answer:* I would discretize the target variable (price) into distinct bins or quantiles. For example, using quartiles to classify a house into Class 1 (0-25th percentile), Class 2 (25-50th), and so on. I'd then train a multi-class classifier.

4. **Why wouldn't Logistic Regression work for predicting housing prices?**
*Answer:* Despite its name, Logistic Regression is a classification algorithm. It uses a logit function to squeeze outputs between 0 and 1 representing probabilities of mutually exclusive classes. It mathematically cannot predict a continuous raw price like $500,000.

5. **How did you split your data, and why did you choose a time-based split over a random train/test split?**
*Answer:* I used a strict chronological split (Train: <2020, Eval: 2020-21, Holdout: 2022+). Random splits lead to severe data leakage in time-series or temporal-based data because the model would be training on future macroeconomic trends to predict the past, which is unrealistic in a production environment where you strictly predict the future using the past.

6. **Explain the concept of Data Leakage. How did you ensure your pipeline doesn't suffer from it?**
*Answer:* Data leakage occurs when the model is inadvertently exposed to data during training that it wouldn't have access to in production, leading to vastly inflated performance metrics. I prevented this by 1) Using chronological splitting, 2) Only fitting encoders (like Target Encoding) on the training set and using those fitted parameters to transform the eval/holdout sets, and 3) removing inherently leaky columns before training.

7. **What is the difference between a validation set and a holdout/test set in your pipeline?**
*Answer:* The validation set (eval) is used extensively during the model building phase for hyperparameter tuning (Optuna) and early stopping. The model *sees* validation performance. The holdout set acts as a pseudo-production environment—it is kept completely sealed until the very end to estimate the model's true real-world generalization.

8. **How do you handle missing values in your dataset?**
*Answer:* In real estate data, I identify if data is "Missing Completely At Random" or "Missing Not At Random". For minor missing categorical features, I either impute with the mode or a placeholder "Unknown." However, because I use XGBoost, I rely on its native sparse-aware algorithm to handle NaN values automatically during split-finding, avoiding heavy imputation strategies that might introduce bias.

9. **Housing prices are often heavily right-skewed. Did you apply any transformations (like log-transform) to the target variable? Why or why not?**
*Answer:* Yes, housing prices are often right-skewed. In this project, applying a log transformation `np.log1p(y)` normalizes the target distribution. This prevents the model from heavily over-indexing on extremely expensive, rare outlier mansions during the gradient descent process. Predictions are then exponentiated `np.expm1(y_pred)` to convert them back to dollar values.

10. **Explain the evaluation metrics you chose (e.g., MAE vs RMSE). When would you prefer one over the other?**
*Answer:* RMSE (Root Mean Squared Error) squares the errors before averaging, penalizing large errors very heavily. MAE (Mean Absolute Error) is the linear average of errors and is much more robust to outliers. I use both, but focus on MAE for a more interpretable business metric (e.g., "we are off by $20k on average"). I would prefer RMSE if large pricing mistakes were financially disastrous to the company (acting as a "worst-case" penalty).

### Section 2: Feature Engineering (11-20)
11. **Why did you use Target Encoding for city names? What is the risk associated with it?**
*Answer:* City names have high cardinality. One-Hot Encoding would result in thousands of sparse, useless columns. Target Encoding replaces the categorical city name with the average target (housing price) for that city. The severe risk is data leakage and overfitting; if a city has only 1 row, its target encoding equals its exact label. 

12. **How do you prevent data leakage specifically when applying Target Encoding?**
*Answer:* First, I strict-fit the `TargetEncoder` ONLY on the Training set. Validation and holdout sets are only `transformed`. Second, many implementations of Target Encoding (like `category_encoders`) apply internal K-fold cross-validation or smoothing/regularization based on frequency to prevent a 1-to-1 label mapping for rare categories.

13. **What is Frequency Encoding, and why did you apply it to zip codes instead of One-Hot Encoding?**
*Answer:* Frequency encoding replaces a category with the number of times (or proportion) it appears in the dataset. I applied it to zip codes because zipcodes represent density/popularity in my set. One-Hot encoding would result in thousands of dimensions, causing the curse of dimensionality, slowing down tree splitting, and creating extreme sparsity.

14. **What are the downfalls of using One-Hot Encoding on high-cardinality categorical variables?**
*Answer:* 1) Curse of dimensionality: tree algorithms must search through thousands of sparse features for the best split, drastically increasing compute time. 2) Sparse data leads to shallow trees unable to map the true signal. 3) Memory limits are easily hit.

15. **How did you handle date/time features? Did you extract month/year, and if so, how does that help the model?**
*Answer:* I extracted purely numerical temporal features like `sale_year`, `sale_month`, and `day_of_week`. This helps the model pick up on seasonality (homes sell for more in the summer) and macroeconomic trends over intervals of years.

16. **Is feature scaling (Standardization/Normalization) necessary for XGBoost? Why or why not?**
*Answer:* No, feature scaling is structurally irrelevant for tree-based models like XGBoost. Trees branch based on inequalities (e.g., `sqft > 1500`). The splitting points are order-preserving; it doesn't matter if you scale `1500` down to `0.85` because the relative ordering remains exactly the same. 

17. **How did you handle outliers in square footage or price during your preprocessing step?**
*Answer:* In the preprocessing script, I analyzed the Interquartile Range (IQR) and absolute thresholds. Extremely unrealistic values (e.g., 50 sqft homes or $500 homes) were dropped because they represent data entry errors. For valid but extreme outliers, XGBoost is inherently robust to them, especially if using MAE as the evaluation function, so I let the algorithm handle them.

18. **If your model goes into production and sees a new city or zip code it has never seen before, how does your pipeline handle it?**
*Answer:* Safe encoding libraries (like `category_encoders`) allow setting a `handle_unknown='value'` parameter. If a new city arrives, the `TargetEncoder` will impute the global mean (or median) of the target variable from the training data, ensuring the API doesn't crash while giving a baseline average guess.

19. **Why is it important to save your fitted encoders (as pickle files) alongside your model?**
*Answer:* If I deploy the model without the encoders, the raw API input (e.g., `{"city": "Denver"}`) cannot be converted to the numerical representation the model expects. Fitting a *new* encoder in production would be disastrous, as it has no historical target data to aggregate. The inference must utilize the exact statistical mappings from the original training run.

20. **Can you explain how you ensured the inference pipeline applies exactly the same transformations as the training pipeline?**
*Answer:* The inference pipeline explicitly loads the saved `.pkl` encoder artifacts generated during `feature_engineering.py`. It calls the exact same `transform` functions, ensuring that parameters strictly replicate the training state without fitting anything fresh.

### Section 3: Model Selection - XGBoost (21-30)
21. **Why did you choose XGBoost? What are its primary advantages for tabular data?**
*Answer:* XGBoost is the state-of-the-art for tabular datasets. It implicitly handles non-linear relationships and intricate feature interactions, seamlessly deals with missing data natively, incorporates built-in L1/L2 regularization to combat overfitting, and trains extremely fast due to parallelized tree building and cache-awareness.

22. **Why didn't you use a Deep Learning framework like PyTorch or TensorFlow for this task?**
*Answer:* Deep Learning requires massive amounts of data to beat tree-based methods and requires heavy feature scaling, imputation, and embedding layers for categorical data. It is severely difficult to interpret. For moderate-sized tabular datasets without unstructured data (like images or text), XGBoost provides higher accuracy much faster with significantly less hardware.

23. **Can you explain the main difference between Bagging (Random Forest) and Boosting (XGBoost)?**
*Answer:* Bagging (Random Forest) builds many deep trees independently and in parallel on bootstrapped subsets of data, then averages their guesses to lower variance. Boosting (XGBoost) builds shallow trees *sequentially*, where each subsequent tree targets the *residual errors* left over by the previous trees. Boosting actively tries to lower bias.

24. **How exactly does a Gradient Boosting algorithm build its trees?**
*Answer:* An initial tree makes a baseline prediction (e.g., the mean of the target). We calculate the pseudo-residuals (the gradient of the loss function). The next tree is then trained *not* on the original target, but to predict the *residuals* of the previous step. The new prediction is the old prediction plus the new tree's scaled prediction.

25. **What is the `learning_rate` (eta) in XGBoost, and how does it interact with the number of trees (`n_estimators`)?**
*Answer:* `learning_rate` shrinks the weight of each tree. A lower learning rate makes the model more robust to overfitting but requires more trees (`n_estimators`) to reach convergence. They are inversely proportional: if you lower `eta`, you must increase `n_estimators`.

26. **How does XGBoost natively handle missing data internally if you choose not to impute it?**
*Answer:* XGBoost's "Sparsity-aware Split Finding" algorithm automatically learns a default direction. During training, it tries sending missing values down the left branch, then down the right branch, and permanently assigns the default direction that results in the greatest reduction in loss. 

27. **What are the key hyperparameters you tuned to control overfitting in XGBoost?**
*Answer:* 
- `max_depth`: Limits max tree size. Lower = less overfit.
- `min_child_weight`: Requires a minimum sum of instance weight in a node. Higher = less overfit.
- `subsample`: Randomly samples a percentage of rows before growing a tree.
- `colsample_bytree`: Randomly samples a percentage of columns per tree.
- `alpha/lambda`: L1 and L2 regularization terms.

28. **How does altering `max_depth` affect the bias-variance tradeoff?**
*Answer:* Increasing `max_depth` allows the model to learn highly specific, complex patterns (low bias, high variance), leading to massive overfitting. Lowering `max_depth` forces the model to learn broader patterns (higher bias, low variance), improving generalization on unseen data.

29. **What regularization techniques does XGBoost employ compared to standard Gradient Boosting?**
*Answer:* XGBoost mathematically incorporates L1 (Lasso, alpha) and L2 (Ridge, lambda) regularization penalties directly inside its objective function. Regular Gradient Boosting (e.g., sklearn's old implementation) does not penalize leaf weights heavily in the objective function.

30. **How do you extract and interpret Feature Importance from an XGBoost model?**
*Answer:* XGBoost tracks "Gain" (the average improvement in accuracy brought by a feature to the branches it is on), "Weight" (number of times a feature is used to split), and "Cover". Gain is the most reliable metric for predictive power. I extract this using `.feature_importances_` or `plot_importance`.

### Section 4: Hyperparameter Tuning - Optuna (31-40)
31. **Why did you choose Optuna over Python's built-in `GridSearchCV`?**
*Answer:* `GridSearchCV` tests every rigid combination exhaustively, wasting massive amounts of compute time on mathematically terrible regions of the search space. Optuna actively learns from past trials, converging on the optimal parameter region dramatically faster.

32. **Explain the difference between Random Search and the Bayesian Optimization (TPE) used by Optuna.**
*Answer:* Random Search blindly picks parameters without memory. Optuna's TPE (Tree-structured Parzen Estimator) builds probabilistic models of how parameters affect the objective score. It calculates expected improvement, inherently balancing exploration (trying new areas) vs. exploitation (drilling down on known good areas).

33. **How does the Tree-structured Parzen Estimator (TPE) algorithm actually work at a high level?**
*Answer:* TPE splits past observations into two groups: "Good" trials and "Bad" trials based on the target metric. It then models the probability distribution of parameters for both groups. To pick the next hyperparameters, it draws values that have a high probability of belonging to the "Good" group and a low probability of belonging to the "Bad" group.

34. **How do you define a search space in Optuna?**
*Answer:* You define boundaries using the `trial.suggest_*` API inside the objective function. For example, `trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)` samples on a logarithmic scale, while `trial.suggest_int("max_depth", 3, 10)` sets distinct integers.

35. **Did you implement Early Stopping in your training phase? How does it save time during Optuna hyperparameter tuning?**
*Answer:* Yes. Early stopping halts a specific model training run if the validation loss hasn't improved over $N$ rounds. This prevents Optuna from wasting compute time trying to build 1,000 trees if the model already converged (or started overfitting) at tree 150.

36. **What metric were you optimizing for during your Optuna trials, and why?**
*Answer:* I optimized for minimizing Validation MAE. MAE gives a linear, easily interpretable representation of cost errors ($x dollars off on average), ensuring Optuna didn't overly sacrifice general accuracy just to fix a few rare massive outliers (which minimizing RMSE might do).

37. **How do you handle the risk of "overfitting to the validation set" when running hundreds of Optuna trials?**
*Answer:* By repeatedly querying the validation set 500 times, you run the risk of finding parameters that randomly memorize the validation set. To counteract this, I maintain a completely hidden, untouchable Holdout test set and use robust K-fold cross-validation inside the Optuna objective if computing power allows.

38. **Can Optuna be parallelized across multiple workers?**
*Answer:* Yes. Using an RDB backend (like SQLite or Postgres), multiple processes or machines can run the optimization script simultaneously. Optuna workers read the shared database history and calculate new Bayesian proposals in parallel.

39. **What happens if Optuna selects an optimal learning rate that is incredibly small—what other parameter must compensate?**
*Answer:* A very small learning rate essentially takes incredibly tiny steps towards the gradient minima. Therefore, the algorithm might stop learning before reaching the minimum. The `n_estimators` (total number of trees built) must be increased drastically to give the model enough "time/steps" to converge.

40. **How does pruning work in Optuna to discard non-promising trials early?**
*Answer:* Optuna uses pruning algorithms (like Hyperband or Median Pruner). It periodically checks the intermediate learning curves (e.g., loss at epoch 10). If the current trial is performing worse than the median performance of all previous trials at that exact epoch, Optuna halts the trial instantly to save time.

### Section 5: MLOps & MLflow (41-50)
41. **What is MLflow and what specific problem does it solve in your project?**
*Answer:* MLflow is an open-source platform for the machine learning lifecycle. It solves the massive problem of "experiment chaos." Without it, determining which hyperparameter combination, dataset split, or codebase version led to `model_v5.pkl` is nearly impossible.

42. **What is the difference between MLflow Tracking and the MLflow Model Registry?**
*Answer:* Tracking is a logging UI/database that records parameters, metrics, run IDs, and environment configs during execution. The Model Registry is a centralized version control system designed to transition a logged model into lifecycle stages (e.g., Staging -> Production -> Archived).

43. **What exact artifacts, parameters, and metrics are you logging in MLflow?**
*Answer:* I log hyperparams (`max_depth`, `eta`), metrics (Train_MAE, Val_MAE, RMSE), tags (timestamp, developer name), and importantly, Model Artifacts (the `pipeline.pkl` or XGBoost format, encoder `.pkl` files, and feature importance graphs).

44. **How does MLflow ensure that the model you trained locally can be reliably reproduced in production?**
*Answer:* By automatically generating isolated environment configuration files (`conda.yaml` or `requirements.txt`). MLflow inherently tracks exactly which libraries (like `xgboost==3.0.4` and `pandas`) were present when the model was trained, removing environment mismatch issues in production.

45. **How did you integrate MLflow with your Optuna tuning script?**
*Answer:* By wrapping the Optuna objective function with `mlflow.start_run(nested=True)`. The parent run tracks the overall Optuna study, while nested child runs log every single trial Optuna executes, giving me a giant, searchable database of hyperparameters vs. metrics.

46. **What is an MLflow "Run" versus an MLflow "Experiment"?**
*Answer:* An Experiment is a high-level grouping (e.g., "Housing_Project_v1"). A Run is a single execution or training pass (e.g., "Trial 45 max_depth=6") that belongs to that specific Experiment.

47. **If the business asks you to revert to the model version you were using two weeks ago, how does MLflow facilitate that?**
*Answer:* Because the Model Registry is version-controlled, I use MLflow's interface or API to simply tag `model version 2` as "Production" and `model version 3` as "Archived". My production API then pulls the currently tagged "Production" artifact instantly.

48. **In a team environment, how does a centralized MLflow tracking server improve collaboration?**
*Answer:* Instead of metrics saving locally to `mlruns/` on my laptop, the centralized server allows engineering, data science, and management to all look at the exact same dashboard, compare each other's runs, and approve model staging deployments via the registry.

49. **How do you load the "best" model from MLflow to be packaged for your FastAPI server?**
*Answer:* Inside the FastAPI code, I can use the `mlflow.xgboost.load_model('models:/Housing_Model/Production')` script, which directly fetches the authorized model binary and its dependencies directly into memory. 

50. **Besides MLflow, did you consider other platforms like Weights & Biases or Neptune? Why MLflow?**
*Answer:* W&B is excellent for deep learning and cloud visualization. I chose MLflow due to its unparalleled open-source adoption, ease of local hosting, native integration with Databricks (an industry standard), and robust programmatic tracking API that scales nicely into GCP/AWS.

### Section 6: Deployment, API & Infrastructure (51-60)
51. **Why did you use FastAPI instead of Flask or Django to serve your model inference?**
*Answer:* FastAPI is inherently asynchronous, built on Starlette (offering vastly superior performance/throughput under concurrent load than standard Flask). It utilizes Pydantic for rigid request validation, directly preventing inference errors, and auto-generates Swagger (OpenAPI) docs.

52. **What role does Docker play in your deployment architecture?**
*Answer:* Docker guarantees absolute consistency. It isolates my operating system, `uv` managed python version, dependencies, model binaries, and API code into an immutable, portable block. "It works on my machine" means it physically must work everywhere Docker runs.

53. **Explain the concept of a "Multi-stage Docker build" and why it's beneficial for lightweight production containers.**
*Answer:* A multi-stage build creates a massive 'builder' image to compile wheels, install heavy compilers, and build dependencies. Only the finalized lightweight artifacts and python code are copied over to a slim, clean 'runner' image. This drastically lowers image sizes, saving storage costs and speeding up Cloud Run boot times.

54. **Why did you migrate your storage and database from AWS to Supabase? What are the benefits?**
*Answer:* Supabase bundles Postgres DB and Cloud Storage directly with unified APIs, offering a more intuitive developer experience. Integrating a simple dashboard database alongside storage without navigating complex IAM permissions (AWS S3 + RDS) heavily streamlined my deployment lifecycle.

55. **You used Google Cloud Run. What does it mean that Cloud Run is a "serverless" technology?**
*Answer:* I do not provision, patch, or maintain underlying servers/VMs. GCP manages all infra. "Serverless" means I merely provide GCP my Docker Container; GCP executes it on demand, automatically scales it up via HTTP traffic from 0 to 1000 instances, and scales it back to 0. I only pay down to the exact millisecond my code runs.

56. **What is a "Cold Start" in serverless computing, and how does it affect your API response times?**
*Answer:* When a Cloud Run instance scales to exactly 0 containers to save money, the next incoming API request must wait for Google to physically allocate a server, inject my Docker Image, boot Python, and load the XGBoost `.pkl` into RAM. This induces a latency spike (Cold Start). Keeping minimum instances at `1` solves it, but increases cost.

57. **How does your Streamlit dashboard communicate with your backend model?**
*Answer:* Streamlit acts solely as the presentation layer (frontend). It captures the user's GUI inputs (sliders, drop-downs), packs them into a JSON payload, and executes an HTTP `POST` request using the `requests` library to my deployed FastAPI Cloud Run URL. It then renders the API response.

58. **In your upcoming features, you mentioned adding Pydantic validation. Why is input validation directly on the API layer critical before passing data to the model?**
*Answer:* ML models fail violently or silently when fed incorrect datatypes. If a user passes an integer for a categorical city, or $0 for sqft, Pydantic catches it instantly at the HTTP layer, returning a helpful standard `422 Unprocessable Entity` rather than throwing internal Python `KeyError` 500s that crash model threads.

59. **You mentioned setting up Evidently for drift monitoring. What is data drift versus concept drift in the context of housing prices?**
*Answer:* Data Drift is a shift in the independent variables (features), like a sudden influx of luxury home listings vs regular homes. Concept Drift is a shift in the underlying relationship; for example, due to a macroeconomic housing crash, a 1500 sqft home is suddenly worth $150k less than it mathematically "should" have been worth a year ago.

60. **If suddenly house prices drop by 20% due to a macroeconomic event, how will your model behave, and how would you fix it?**
*Answer:* The model will suffer catastrophic Concept Drift and consistently predict prices 20% too high, severely punishing MLflow metrics. To fix this, I must schedule pipelines to pull the new reality's data, validate drift via `Evidently`, re-run the Optuna Tuning pipeline to unlearn the old weights, and redeploy to FastAPI to reflect the current market.
