# TIP - Technical Innovation Project
# INTRODUCTION
**Project Overview**
Project title: Traffic Forecasting and Dashboard Visualisation

**Project topic background and motivations**
Solving the predicting problems is still an active area of research, especially with big data and machine learning models. In transport, learning the trends and predicting the future speeds or flows can help both the decision-makers and authorities having adapted policy-making and operate the transport management system in an efficient and effective way. Combining machine learning, artificial intelligence with big data such as RNNs, GRNNs, MNNs, DLBP, RBFNs, and LTSM, analyzed the real-life traffic data collecting by the setting up sensor stations along the freeways. The good results of these models have enabled the chances to enhance the usage of the general public to the services and assist policy and decision-makers to proficiently in planning and managing the transportation facilities more easily.

# DESING CONCEPT
## High Level Design
At the analysing the project requirement, the first version of high level design for our team proposal  was as following:
Figure 01 - High Level Design
![High Level Design](/images/1.png)

In this first design we have 3 tiers as following: 
**Data Modelling Tier**: The dataset will be split into training set/testing set and predict set. The predicting results will be saved into the relational database with appropriate formats that need to be visual in the web tier. The results of visualization steps are also persistent in the database for demonstrating with the dashboard. Python, R, or Matlab will be used to work with modeling the data.
<br>**Database Tier**: Mysql or mongodb are using as the data engine with some simplet database tables for storing all necessary data for the project. 
<br>**Application/Web Tier**: Django will be used to implemented both backend layer and frontend layer with will be combinning with some predefined dashboard templates (implemented in HTML/CSS/Javascript) to illustrate the results.
However, after researching in details the project, we have adjusted the high level design as below. The main changes in the new design is for transforming of the workload in data manipulation part and the data visualisation from 50%-50% (the initial design) to 75%-25% (the final one).

Figure 02: Adjusted High level design
![Adjusted High level design](/images/2.png)

**Data Modelling Tier**: The modelling is keeping as the initial design with some minor steps to extract (R code), transform (R code) then load the data to the selected tool of visualisation (Tableaus public) to develop the interactive web for the dashboards.
<br>
**Dataset / Database Tier**: Instead of using big data engine like mysql or mongodb, the datasets and middle results are mainly using the CSV and Excel formats. 
<br>
**Visualisation Web Tier with Tableaus Public**: Take the advantages of Tableaus public into account, which are supporting speedly development the dashboard for multiple testing times as well as the capacity of publishing as the web result for using by public users. The high level design was changed from using complicated coding framework like Django or D3js to using Tableaus public for visualizing and publishing the results. 

## Methodologies
### Graph Neural Network (GNN)
GNN has the capability to handle almost all graph types such as directed/undirected or acyclic/cyclic graphs. Furthermore, many studies about forecasting traffic have applied GNN in creating their predicting model. Xie et al. (2020) explained his forming GNN with road segments are the nodes and can be connected by the edges.

General view of working with GNN (source Stanford online)
![General view of working with GNN](/images/3.png)

GNN expects the traffic networks be reprenting in a graph form (Figure 05). In such cases the GNNs needs well-defined graph structures, in that structure the sensors considering as the graph nodes with assigned features in each node, and the roads (the edge between nodes) will work as graph connections. Wu et al. have proposed a GNN method explicitly designed for multivariate time series data (Wu et al. 2020).

### SST-GNN Accuary Final Result
The first try with all East-bound 26 sensors data, the MAPE value and the accuracy is just 11.37% and 88.63% respectively. After digging more details in the sensors data, we found that there were 9 sensors has many missing values so remaining 17 sensors will be chosen for the next try, it shows a little bit improvement in MAPE and Accuracy values at 10.73% and 89.27%.

All trying time and results    
![All trying time and results  ](/images/4.png)

As the consulting from tutor, the missing values were considered with multiple strategies. As the original study, he and his team has used local average for missing values – Speed(t) = (Speed(t-1) + Speed(t+1))/2. He strategy is suitable for using with small duration (like 30 days). However, in our case, we used up to 87 days for the data duration so we have chosen a simpler strategies with MEDIAN value (1st try) then MEAN value (final try). At the 1st try with median value for 17 selected sensor, we have a promising value of MAPE (8.29%) and Accuracy (91.71%). When trying with MEAN values, we have better accuracy so that we sticked with MEAN strategy for missing values for later results.

Comparison between SST-GNN and 4 layer Bi-LSTM
![Comparison between SST-GNN and 4 layer Bi-LSTM](/images/5.png)

For the predicting values of “15 min”, it shows that 4 layer Bi-LSTM having the better result in both MAPE (3.06%) and Accuracy (96.94%) than STT-GNN model – MAPE (7.79%) and Accuracy (92.21%) for East bound and MAPE (6.98%) and Accuracy (93.02%) for West boud. 
However, when running with “30 min” and “45 min”, the result of GNN is take over the place, becoming a little bit better than 4-layer Bi-LSTM, such as MAPE and Accuracy for WB. 
 
### Multilayer Perceptron Model (MLP)
Multilayer Perceptron is a deep learning and artificial neural network. It is composing of more than one perceptron (multiple). They are combination of one input layer receiving the signal, one output layer making a decision or prediction about the input, and in between those two layers, an arbitrary number of hidden layers that are the true computational engine of the MLP. MLPs with one hidden layer are capable of approximating any continuous function. MLPs are normally use for the supervised learning problems. For the time-series dataset like the current traffic date, MLPRegressor will be used for forecasting the future values based on the regression calculation of existing ones.

<br>**MLP Accuracy Final Result**
Not similar to the GNN model result (with 17 sensors for each bound), at the time we worked with MLPRegressor, we just focused on the comparison between 4 selected sensors of base studies.

Comparing the result of LSTM and MLP
![Comparing the result of LSTM and MLP](/images/6.png)

The result in the above table shows that LSTM result is better, more stable, and has less variant than the result when we run with MLPRegressor. For example, the highest value of accuracy of East Bound is 94.98% with sensor 14010EB and the lowest value of accuracy is for 14045EB at 61.62%.  The delta (variant) value between these two accuracies is high – at (94.98% - 61.62%) = 33.36%. It’s totally different from the result of LSTM, they have almost value greater than 90% and the variant is just few percents (~ 10%).

### Visualisation Methodologies
<br>**Tableaus Public**
The Tableaus Public is the fastest growing data visualisation tool in Business Intelligent Industry. It was considerd to take over the role of complex implementation for dashboards front-end because of its advantages:
Create quickly the interactive visualisation: the end users can easily create a good interactive widgets and dashboards mostly by drag and drop functions.
<br>**Comfortable implementation**: there are many options for visualisation in Tableaus and has been improved the user experience. It’s easy to understand then use comparing to learning code Python for same purpose. 
Easily handles large amount of data in different format of inputs: Tableaus now suports multiple data source connections with SQL, NoSQL, Flat files, etc. The different data source and large amount of data does  not affect to the performance of the dashboards.  

# PROJECT ACTIVITIES
## Model research, test and select platforms with MT-GNN, SST-GNN and MLP Regressor.
Coding with python and R, we have started to research then apply the GNN models like SST-GGN, MT-GNN, etc to the project. After playing around the some model implementation, the SST-GNN is becoming promising one for our dataset. We have brought the model up then modifying its code to adapt with our expectations then sharing the knowledge to others to bring them up then analyze their assigning tasks. The code was sync up over the github repositories. Furthermore, when comes to MLP model, there are other studies with MLPRegressor for Time series predicting, which has been successfully implemented by MLPRegressor. Based on these implementation, the MLPRegressor model is modified and applied for Traffic Dataset.

## ETL the data
Completed research the platform for SST-GNN and MLPRegressor, I found that the provided dataset should be extracted then transformed to different format that expected by the models. Started the R code to extract then transform firstly for GNN model which requires two different files: one for vector of speeds for all selected sensors, another for the adjacency matrix contains the relationship between sensors, in this case is distance between sensors. The second one is for transforming the data to MLPRegressor input which is consider as a matrix of time series which each row of data will be format as “Speed (t-4), Speed (t-3), Speed(t-2), Speed(t-1), Speed(t)”.
After completing the predicted value and true value of each model, the output data will be transformed again for visualisation purpose with Tableaus. Each of the East bound and West bound Worksheets included the below: 
Excel file including a separate sheets for 
-	Time stamp 
-	Sensor ID 
-	15mins True 
-	15mins Predicted 
-	30mins True 
-	30mins Predicted 
-	45mins True 
-	45mins Predicted 

All steps in ETL the data
![All steps in ETL the data](/images/ETL.png)


Initial the visualisation idea with Tableus Public
At the first try with Tableaus, the data was using with multiple files for predicted and true values for 17 sensors (over 34 files were using). With the complex in the relationship when loading data into Tableaus, it caused some troubles for developing our expected boards. Worked closely with team, we have proposed the new format so that the dashboards can integrate and show the correct output. Below is the first result of the board that we have proposed.

Tableaus Public Initial Boards
![Tableaus Public Initial Boards](/images/9.png)

Detailed views at:
https://public.tableau.com/app/profile/qui.tran/viz/TIP-Dashboards/Stories

# PROJECT OUTCOMES
As the requirement of the project, our team has provided almost aspects for the project includes:
## SST-GNN Model
Modelling the data model for GNN, training with implementation for SST-GNN and predicting the traffic speeds then visualized the result into interactive web based dashboards. 
The accuracy result is promising and comparable with original results of the based studies. 
## MLPRegressor Model
Modelling the data model for MLP, training with implementation of MLPReressor for time series and predicting the traffic speeds for each sensor. The visualisation is not developed for this case of the model.
The accuracy of MLPRegressor is not good and not stable as the accuracy of LSTM model. 

## Visualisation Dashboard
There are two dashboards for two directions East Bound and West Bound. Each dashboard contains:
-	A map show all selected sensors, and each sensor could be select for navigating the charts
-	Time slider for select the inspecting duration 
-	3 multiple line charts (1 for predicted value, 1 for true value) of three categories 15 minutes, 30 minutes and 45 minutes results.

# CONCLUSION
Our team has demonstrated the project with a good outcomes as the requirements by using a good systematic approach to explore, choose, and then apply the models to specific context of dataset of traffic data. The result has proved that GNN model type can be applied in real application for predicting the traffic speed or flow. However, its running speed is quite slow when the data keep growing with CPU so the real application should be allocated with a high GPU capability. 
# RECOMMENDATIONS
Besides some promising outcomes for two new models and the visual results, there are some limitations which should be considered in the future researchs with this topic and dataset:
-	The better strategy for replacing missing data instead of using mean or median value.
-	Responsive mobile views for user easily to view the dashboard by their mobiles
-	Create new enquiry UI so that general public user can lookup by their own purpose
-	Change the model of data engine by integrating with sensors’ output in order to show the data real-time in dashboard.

# REFERENCES
[1] Resul L. Abduljabbar, Hussein Dia, Pei-Wei Tsai and Sohani Liyanage, “Short-Team Traffic Forecasting: An LSTM Network for Spatial-Temporal Speed Prediction”, Future Transp. 2021, 1, 21-37
[2] Rusul L. Abduljabbar ,1 Hussein Dia,1 and Pei-Wei Tsai2, “Unidirectional and Bidirectional LSTM Models for Short-Term Traffic Prediction”, Journal of Advanced Transportation, Vol 2021
[3] H. Dia, G. Rose, and A. Snell, “Comparative performance of freeway automated incident detection algorithms,” in Institute of Transport and Logistics Studies Working Paper ITS-WP-96-15 The University of Sydney, Sydney, Australia, 1996, https://ses.library.usyd.edu.au/handle/2123/19426. 
[4] Abduljabbar, R.; Dia, H.; Liyanage, S.; Bagloee, S.A. Applications of artificial intelligence in transport: An overview. Sustainability 2019, 11, 189.
[5] Roy, A, Roy, KK, Ahsan Ali, A, Amin, MA & Rahman, AKMM 2021, 'SST-GNN: Simplified Spatio-Temporal Traffic Forecasting Model Using Graph Neural Network,' Springer International Publishing, 90-102.
[6] Xie, Z, Lv, W, Huang, S, Lu, Z, Du, B, Huang, R 2020, ‘Sequential Graph Neural Network for
Urban Road Traffic Speed Prediction’, IEEE access, vol. 8, pp. 63349-63358
[7] Aleksandr Pletnev , Rodrigo Rivera-Castro and Evgeny Burnaev, ‘Graph Neural Network for Model Recommendation using Time Series Data’,  2020, 19th IEEE International Conference on Machine Learning and Applications
[8] Nelles, O., 2001. Nonlinear System Identification: From Classical Approaches to Neural Networks And Fuzzy Models. Springer, ISBN: 9783540673699 Berlin Pages: 785

# TEAM MEMBERS
Kristian, Chanaka, Vu, Qui
