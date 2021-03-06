# Beau Kramer

## [Resume](https://github.com/beaukramer/beaukramer.github.io/blob/master/Beau_Kramer_Resume.pdf)

## Projects

### Finance

- [Eigen Portfolios](https://nbviewer.jupyter.org/github/beaukramer/financial_analysis/blob/95a513992ca604e0116ec9690453fe3ee0d5fa50/Eigen%20Portfolios/eigen_portfolios.ipynb)
This was an implementation of a tactical asset allocation strategy where principal component analysis helped separate and weight assets into an offense and defense portfolio.

- [Nowcasting Recessions](https://github.com/beaukramer/financial_analysis/blob/master/Nowcasting/Nowcasting_ML.ipynb)
The goal of this project is to create a machine learning algorithm to detect whether the economy is in a recesssion or not. The NBER announces the beginning and end of recessions *after* they have passed. This presents an opportunity: if we could know the economy is likely already in  a recession months *before* it is officially announced then we could take appropriate action. For example, a manufacturer could scale down production in anticipation of the coming recession. Alternatively, it could scale up production if it knew that the recession was likely already over. I have two primary goals in developing this model: for it to be 1) performant and 2) interpretable.

- [Pycerno](https://github.com/beaukramer/financial_analysis/blob/master/Pycerno/Pycerno.ipynb)
This is a Python "translation" of the book "Quantitative Investment Portfolio Analytics In R" by James Picerno. I found this to be a uesful exercise because 1) it familiarized me with different Python packages relevant to financial analysis 2) made it easy for me to port R code I find to Python and 3) gave me lots of template code to do basic financial analysis of new datasets. Some of the code is clunky but this was in an effort to be faithful to the source code.



- [CPI Visualization](https://docs.google.com/presentation/d/1GzarxmT_PxbcuKjcMhzQWHIqU5EMhqPfHCnWNRi_Ghc/edit?usp=sharing)
This was a small project around picking a dataset and performing some EDA on it. Due to my background in finance, I chose the consumer price index. The numerous sub-components made it fun to disaggregate.


### Machine Learning

- [Automated Essay Grading and Inference Using Linear and Deep Learning Models](https://github.com/pkurapati/W266-NLP-Project/blob/master/W266_Automated_Essay_Grading.pdf) 

> NLP | Deep Learning  | Feature Engineering | Data Visualization

For this NLP project, my team selected an automated essay grading challenge. We were interested in automatically generating feedback for students and for contrasting linear models with deep learning ones. We performed extensive feature engineering and trained deep learning models on an essay and sentence level. 


- [News Article Topic Classification](https://github.com/beaukramer/mids/blob/master/ML/TopicClassification/topic_classification.ipynb)

> NLP | Classification | Data Processing

I trained classifiers using a bag-of-words model to identify the topic of a news article. After some intial attempts at the problem, I applied some preprocessing to the texts which improved the ability of the model to generalize.


- [Poisnous Mushroom Clustering](https://github.com/beaukramer/mids/blob/master/ML/Mushroom%20Clustering/mushroom_clustering.ipynb)
> Clustering | Data Visualization

Using PCA to reduce dimensionality, I clustered data about mushrooms to try to classify poisonous ones. I used KMeans and Gaussian Mixture Models.


- [Forest Cover Prediction](https://github.com/beaukramer/mids/blob/master/ML/ForestCoverPrediction/Forest_Cover_Master_v4.ipynb) 
> Classification | Data Visualization | Ensembling

For this group project, my team selected the forest cover prediction challenge. We had to predict the species of tree that lived in a 30x30m cell in several Colorado forests. We summarized our lessons learned in this [presentation](https://github.com/beaukramer/mids/blob/master/ML/ForestCoverPrediction/Forest_Cover_Prediction_Type.pdf).



### Statistics
- [Crime Policy North Carolina](https://github.com/beaukramer/mids/blob/master/Stats/Crime/Kramer_Liu_crime.pdf) 
> Linear Regression | EDA

Classic linear regression was the focus of this project. We were given a dataset about crime in North Carolina in the 1980s with the goal of providing policy recommendations to politicians.

- [Challenger Explosion](https://github.com/beaukramer/mids/blob/master/Stats/Challenger/Kramer_Papandrew_Challenger.pdf) 
> Discrete Response | Logistic Regression

This discrete response modeling project addressed whether temperature and/or pressure have a relationship with the failure of the primary o-rings in the space shuttle.

- [Time Series Forecasting with SARIMA Model](https://github.com/beaukramer/mids/blob/master/Stats/TimeSeries/Kramer_Papandrew_TS.pdf)
> Time Series | ARIMA

This was a simple exercise in forecasting an e-commerce time series from the Federal Reserve Bank in St. Louis. After exploring and verifying the data's suitability for the model, we fitted a seasonal ARIMA model to the data.

- [Fatality Rates with Fixed Effects](https://github.com/beaukramer/mids/blob/master/Stats/DrunkDriving/Kramer_Papandrew_DrunkDriving.pdf)
> Panel Data | Fixed Effects

This panel dataset on driving laws proved a challenge to explore visually, but we managed to create some visuals that helped us understand the data. We then used a fixed effects model to capture the effects of different policies on fatality rates.

- [Cereal Content and Shelf Probabilities](https://github.com/beaukramer/mids/blob/master/Stats/Cereal/Kramer_Papandrew_Cereal.pdf)
> Multinomial Regression | Odds Ratios

Multinomial regression of a cereal dataset was the main task in this project. In addition to calculating odds ratios, we built up toward visuals that showed the shelf probability by nutritional content.

- [Forest Fires](https://github.com/beaukramer/mids/blob/master/Stats/ForestFire/liu_warther_kramer_hegde_fires.pdf)
> Data Visualization | EDA

The focus of this project was on the importance of exploratory data analysis in any project. We explored a dataset about a portuguese national park that experienced severe wild fires to see if we could begin predicting conditions that make an area vulnerable to extreme fires.

### Fun


