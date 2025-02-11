### Predictive Modeling of Diabetes Risk Factors

**Anjana Cox**

#### Executive summary

#### Rationale
Health and Social Impact
Public Health: Diabetes is a major public health concern, affecting millions of people worldwide. Early prediction and intervention can prevent or delay the onset of diabetes, reducing healthcare costs and improving quality of life.

Preventive Care: By identifying individuals at risk of developing diabetes, healthcare providers can offer targeted preventive measures, such as lifestyle changes or medical interventions, to reduce the likelihood of disease progression.

Personalized Medicine: Machine learning models can help tailor personalized healthcare plans based on an individual's unique risk factors, leading to more effective and efficient treatment strategies.


#### Research Question
"Can we accurately predict the risk of diabetes and prediabetes in individuals using machine learning models based on a set of health indicators and lifestyle factors?"

Specific Objectives:
Prediction Accuracy: How accurately can different machine learning models predict the presence of diabetes or prediabetes using features such as blood pressure, cholesterol levels, BMI, smoking status, physical activity, and dietary habits?

Model Comparison: Which machine learning model (Logistic Regression, SVM, KNN, Random Forest, Decision Tree) performs best in terms of accuracy, recall, precision, and F1-score for predicting diabetes risk?

Feature Importance: Which health indicators and lifestyle factors are most significant in predicting diabetes risk, as identified by the machine learning models?

#### Data Sources
[The dataset can be found on Kaggle.](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

The data is obtained via the  Behavioral Risk Factor Surveillance System (BRFSS) which is a health-related telephone survey that is collected annually by the CDC. Each year since 1984, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. For this project, a csv of the dataset available on Kaggle for the year 2015 was used. This original dataset contains responses from 441,455 individuals and has 330 features but the features in this dataset were narrowed down to features directly pertaining to diabetes.

There are 3 csv files that are available and the one chosen for this analysis is diabetes_binary_health_indicators_BRFSS2015.csv where the values for the target variable are 0 for no diabetes and 1 for prediabetes and diabetes. This dataset pertains to patient health indicators and includes both demographic and health-related features. Each entry is uniquely identified by a patient ID (ID). The primary target variable, Diabetes_binary, indicates the presence of diabetes or prediabetes. Various binary features capture health conditions and behaviors, such as high blood pressure (HighBP), high cholesterol (HighChol), smoking history (Smoker), history of stroke (Stroke), and coronary heart disease or myocardial infarction (HeartDiseaseorAttack). Additional features track physical activity (PhysActivity), dietary habits (Fruits and Veggies), heavy alcohol consumption (HvyAlcoholConsump), and access to healthcare (AnyHealthcare, NoDocbcCost). General health status is assessed using a scale (GenHlth), and mental and physical health issues are quantified by the number of days affected in the past month (MentHlth, PhysHlth). The dataset also includes information on difficulties with walking (DiffWalk) and the sex of the patient (Sex). All these features are essential for analyzing and predicting diabetes risk, with no missing values reported.

#### Methodology

1. Data Preprocessing
Data Cleaning: The dataset did not have any missing values but did contain some duplicates. Those were removed. It was observed that there was a huge disparity in the target column for the minority class (individuals with diabetes or prediabetes). To compensate for this, we did Data Resampling. This addressed the impalance by upsampling the minority class to match the number of samples in majority class.


2. Feature Engineering
Feature Selection: All of the features in the dataset were included in the analysis. Preprossing was done for all of the columns except the binary. 

3. Machine Learning Models
A baseline accuracy was obtained and then Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest and Decision Tree models were implemented.

4. Model Evaluation and Hyperparameter Tuning
Grid Search with Cross-Validation: Tuning hyperparameters for each model to optimize performance. Using 5-fold/ 3-fold cross-validation ensures that the model generalizes well to unseen data.

5. Evaluation Metrics
Using accuracy, precision, recall, and F1-score to assess and compare the performance of each model. While all of these metrics are very important, recall was priortitized because it is crucial to minimize false negatives since this can result in a patient not receiving necessary treatments.

6. Feature Importance
Logistic Regression Coefficients: Analyzing the magnitude and direction of coefficients to determine feature importance.

Random Forest Feature Importance: Using feature importance scores from the Random Forest model to identify significant predictors.
Visualization and Reporting

Confusion Matrix: Visualizing the performance of each model in terms of true positives, true negatives, false positives, and false negatives.

Bar Plots: Comparing the performance metrics (accuracy, precision, recall, F1-score) of all models using bar plots.

Feature Importance Plots: Visualizing the importance of different features as determined by the models.


#### Results

![comparison_metrics](https://github.com/anjana250/capstone/assets/15185723/08a97141-f585-4bf5-8113-607f2f6e8775)


Random Forest did the best on all the metrics except for Recall. KNN did the best for recall. Since Random Forest seems to have performed the best overall, the recommendations will be made using this model.

#### Observations from Feature Importance:
1. Factors that increase likelihood of diabetes:
    1. Cholestoral Check: Those that have cheked their cholestoral in the last 5 years are more likely to be diabetic.
    2. High Blood Pressure and High Cholesterol: Patients who have high blood pressure or high cholesterol are more likely to have prediabetes/diabetes.
    3. General Health: Those that have poor general health are more likely to have prediabetes/diabetes.
    
2. Factors that decrease likelihood of diabetes:
    1. Eating Fruits and Vegetables decreases the likelihood of getting diabetes!
    2. Physical activity decreases the likelihood.
    
    
#### Conclusion:
Individuals who already have other conditions such as High Blood Pressure, High Cholestoral, Mental Health issues are more likely to be prediabetic/diabetic. It is important for those individuals to get early intervention. Factors that could improve an individuals chances of not getting diabetes is eating more fruits and vegetables and being more active.


#### Next steps

1. Advanced Modeling Techniques : Ensemble Methods: Explore advanced ensemble techniques such as Gradient Boosting (e.g., XGBoost, LightGBM) and AdaBoost, which often provide superior performance by combining the strengths of multiple models.

2. Model Stacking: Combine predictions from multiple models (e.g., Logistic Regression, SVM, Random Forest) to create a meta-model that can improve overall predictive performance.

3. Model Evaluation and Validation: ROC and AUC: Use ROC (Receiver Operating Characteristic) curves and AUC (Area Under the Curve) scores to evaluate and compare model performance more comprehensively.


#### Outline of project

[Full Jupyter Notebook located here.](https://github.com/anjana250/diabetes_capstone/blob/main/Diabetes_Capstone.ipynb)

##### Contact and Further Information

[You can reach me on LinkedIn!](https://www.linkedin.com/in/anjana-cox-593b407a/)
