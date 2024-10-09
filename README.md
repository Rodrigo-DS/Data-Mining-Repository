<h1>Repository with data mining methodologies</h1>
<h4 style="text-align:justify;">In this repository, you'll find my approach to generating solutions for data mining projects. I follow the widely-used CRISP-DM methodology, as outlined in the image below. It provides an overview of the typical phases of a project, the tasks associated with each phase, and an explanation of how these tasks interconnect.</h4>
<figure class="image" data-ckbox-resource-id="sQTij5pLY_TL">
    <picture>
        <source srcset="https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/80.webp 80w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/160.webp 160w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/240.webp 240w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/320.webp 320w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/400.webp 400w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/480.webp 480w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/514.webp 514w" type="image/webp" sizes="(max-width: 514px) 100vw, 514px"><img src="https://ckbox.cloud/ce2cf76480eda7687bf6/assets/sQTij5pLY_TL/images/514.jpeg" width="514" height="514">
    </picture>
</figure>
<h5>1. Business Understanding<br>&nbsp;</h5>
<ul>
    <li>Determine the main objectives of the organization related to the project;</li>
    <li>Analyze the current scenario and identify challenges or opportunities related to treasury management;</li>
    <li>Conduct an in-depth analysis of treasury management concepts, identifying challenges related to forecasting invoice payment dates;</li>
    <li>Clearly define the objectives of data mining in the context of treasury management;</li>
    <li>Develop a detailed plan outlining the phases and tasks to be carried out during the project.</li>
</ul>
<h5>2. Data Understanding<br>&nbsp;</h5>
<ul>
    <li>Description of the data (meaning of the variables, number of variables and records, meaning of the records, among others);</li>
    <li>Data exploration (graphical and statistical analysis, identification of correlations);</li>
    <li>Data quality verification (identification of possible errors, such as missing values, outliers, or inconsistent data);&nbsp;</li>
    <li>
        <p>Analyze the prevalence and nature of the identified errors</p>
        <p>&nbsp;</p>
    </li>
</ul>
<p><strong>3. Data Preparation</strong><br>&nbsp;</p>
<ul>
    <li><strong>Data cleaning: </strong>
        <ul>
            <li>Selection of relevant data for analysis, considering its importance to the project objectives;</li>
            <li>Removal or imputation of data and selection of the best estimator, if necessary);</li>
            <li>Cleaning incorrect, incomplete, or duplicated data;</li>
        </ul>
    </li>
    <li><strong>Data transformation:</strong>
        <ul>
            <li>Feature engineering: Creating new aggregated or derived features from existing data to uncover insights;</li>
            <li>Encoding categorical data: Converting text categories to numbers for modeling;</li>
            <li>Data normalization or scaling: Standardizing data ranges to enable meaningful comparisons.</li>
            <li>Anomaly Detection</li>
            <li>Combining data from different sources</li>
            <li>Anonymizing personal information</li>
            <li>Converting data types</li>
            <li>Structuring unstructured data</li>
            <li><strong>&amp; Others</strong></li>
        </ul>
    </li>
    <li><strong>Data integration</strong> (grouping data, combining information from various columns);<ul>
            <li>Feature selection: Selecting the most relevant features to avoid over-fitting.</li>
            <li>Addressing class imbalance: Re-sampling if one target class dominates to prevent bias.</li>
            <li><strong>&amp; Others</strong></li>
        </ul>
    </li>
    <li><strong>Data formatting</strong>.</li>
    <li>Data Splitting: Divide the data into three sets — training data, validation data, and test data.</li>
</ul>
<h6>4. Data Modeling for Machine Learning</h6>
<p>Data Modeling is a process of the Crisp-dm methodology that aims to estimate the inherent structure of a dataset to reveal valuable patterns and predict unseen instances. Remember, Crisp-dm is an interactive process. This step needs to be complemented by the previous step (data preparation).</p>
<p>At this step we need to choose the correct machine learning algorithms according to our objectives. In the following diagram we can see all the possible options:&nbsp;</p>
<figure class="image" data-ckbox-resource-id="qeaEbSmAbOcG">
    <picture>
        <source srcset="https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/288.webp 288w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/576.webp 576w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/864.webp 864w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/1152.webp 1152w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/1440.webp 1440w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/1728.webp 1728w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/2016.webp 2016w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/2304.webp 2304w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/2592.webp 2592w,https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/2880.webp 2880w" sizes="(max-width: 2880px) 100vw, 2880px" type="image/webp"><img src="https://ckbox.cloud/ce2cf76480eda7687bf6/assets/qeaEbSmAbOcG/images/2880.png" width="2880" height="1304">
    </picture>
</figure>
<p><strong>Supervised Learning:</strong></p>
<p>Supervised learning can be categorized into two main types:&nbsp;</p>
<ul>
    <li>Classification: This involves predicting a discrete label, such as identifying an email as spam or not spam.</li>
    <li>
        <p>Regression: This involves predicting a continuous value, like forecasting the price of a house based on its features.</p>
        <p><strong>Note</strong>: The category depends on the target variable. The predictors can be continuous or discrete (or both) depending on the model selected.&nbsp;</p>
    </li>
</ul>
<p>&nbsp;</p>
<p><a target="_blank" rel="noopener noreferrer" href="https://www.simplilearn.com/10-algorithms-machine-learning-engineers-need-to-know-article">Popular Supervided Learning Algorithms:</a></p>
<ul>
    <li>Linear Regression: Used for predicting continuous outcomes. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.</li>
    <li>Logistic Regression: Used for binary classification tasks (e.g., predicting yes/no outcomes). It estimates probabilities using a logistic function.</li>
    <li>Decision Trees: These models predict the value of a target variable by learning simple decision rules inferred from the data features.</li>
    <li>Random Forests: An ensemble of decision trees, typically used for classification and regression, improving model accuracy and overfitting control.</li>
    <li>Support Vector Machines (SVM): Effective in high-dimensional spaces, SVM is primarily used for classification but can also be used for regression.</li>
    <li>Neural Networks: These are powerful models that can capture complex non-linear relationships. They are widely used in deep learning applications.</li>
</ul>
<p>&nbsp;</p>
<p><a target="_blank" rel="noopener noreferrer" href="https://www.restack.io/p/supervised-learning-answer-advanced-algorithms-cat-ai">Advanced Supervised Learning Algorithms</a></p>
<p><br><strong>Hyper-Parameter Optimization Techniques:</strong></p>
<ul>
    <li><strong>Grid Search</strong>: This method exhaustively searches through a specified hyperparameter space. While it guarantees finding the best model, it can be computationally expensive and time-consuming, especially with numerous hyperparameters.</li>
    <li><strong>Random Search</strong>: Unlike grid search, random search samples a fixed number of hyperparameter combinations randomly. This approach can be more efficient and often yields comparable results to grid search with less computational cost.</li>
    <li><strong>Bayesian Optimization</strong>: This advanced technique models the performance of the model as a probabilistic function and uses this model to select the most promising hyperparameters to evaluate next. It is particularly effective in reducing the number of evaluations needed to find optimal hyperparameters.</li>
</ul>
<p>&nbsp;</p>
<p><strong>Ensemble Methods in Supervised Learning</strong><br>&nbsp;</p>
<ul>
    <li><strong>Bagging</strong>: This technique involves training multiple models on different subsets of the data and aggregating their predictions. Random Forest is a well-known example of bagging, where multiple decision trees are trained and their outputs are combined to enhance accuracy.</li>
    <li><strong>Boosting</strong>: Boosting focuses on sequentially training models, where each new model attempts to correct the errors made by the previous ones. AdaBoost is a popular boosting algorithm that adjusts the weights of misclassified instances to improve the model's performance.</li>
    <li><strong>Stacking</strong>: This method involves training multiple models and then using another model to combine their predictions. It leverages the strengths of various algorithms to achieve better accuracy.</li>
</ul>
<p>&nbsp;</p>
<p><strong>Unsupervised Learning:&nbsp;</strong></p>
<p>&nbsp;</p>
<p><strong>Semi-Supervised Learning:</strong></p>
<p>&nbsp;</p>
<p><strong>Reinforcement Learning:&nbsp;</strong></p>
<p>&nbsp;</p>
<h5>5. Evaluation<br>&nbsp;</h5>
<h5>6. Deployment<br>&nbsp;</h5>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
