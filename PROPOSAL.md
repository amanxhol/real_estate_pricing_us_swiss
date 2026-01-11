Project Title:
How Raw Material Costs Influence the Real Estate Market
Category:
Data Science / Economics / Machine Learning
Problem Statement / Motivation:
Fluctuations in housing prices are often attributed to factors such as interest rates, economic growth, and demand. However, the impact of construction material costs—steel, wood, concrete, and other raw inputs—remains less explored, even though these materials represent a significant share of total construction expenses.
This project aims to explore and model the relationship between construction material costs and housing prices, with a focus on identifying a potential lead-lag effect between the two. In other words, can increases in material costs help predict rises in housing prices and when?
The study will focus on Switzerland and the United States, two countries with reliable, long-term economic datasets covering both real estate prices and production indices. The goal is to apply machine learning techniques to quantitatively assess and predict these relationships.
Planned Approach and Technologies:
The project will follow a complete data science workflow with a strong focus on predictive modeling.
1.	Data Collection: from public sources such as FRED (Federal Reserve Economic Data), the Swiss Federal Statistical Office (BFS/OFS) and the WorldBank.
2.	Data Preparation: using Pandas and NumPy to clean and harmonize data weekly.
3.	Machine Learning Modeling, training , testing : The models are : Ridge Regression, Random Forest, Gradient Boosting. Models will be trained on data from 2005 to 2018 and tested on data from 2019 to 2024.  For every models, different lag period will be tested to determine which predict the best. Both of the countries will be tested separately.
4.	Visualization and Interpretation: generating visualizations with Matplotlib and Seaborn to illustrate trends, lag effects, and feature importance.
Expected Challenges and Solutions:
One main challenge is distinguishing correlation from causation, as housing prices are affected by many other factors such as demand, inflation, and interest rates. The project will therefore focus on identifying predictive patterns rather than establishing definitive causal relationships.
Additionally, designing relevant lag structures (for example 1-month, 3-month, or 6-month lags) will require experimentation to determine which provides the best predictive performance. 
Success Criteria:
The project will be considered successful if it produces a clean and well-aligned dataset for both countries, implements and compares multiple machine learning models with clear evaluation metrics, generates interpretable visualizations showing the lead-lag dynamics between material costs and housing prices and delivers organized, documented, and reproducible code.
Stretch Goals (if time permits):
Include additional explanatory variables such as building permits, different commodities (gold, silver,…) and expand to additional countries.



