# Movie-Recommender-Model

## Purpose
There are nearly endless types of movies out there. Even ignoring subgenres, there are 20 “main” movie genres that add up to many more when combined together.
However, what genre is the best? More importantly, which genre do audiences prefer the most? To tackle this question, we created models that both predicts a movie’s rating based on its genre combination and also which main genres tend to factor most into higher ratings. Along the way, we also created a recommendation algorithm based on movie genres and ratings. In doing so, we hope to both create something interesting to look at and provide potential value to both movie producers and providers by informing actionable data.

## Data

MoviesLens is a project of GroupLens, which is itself a research lab based out of the University of Minnesota. They both make their data sets public while providing free movie recommendations to users. This helps achieve their purpose of continuing to develop resources for data exploration. Best of all, they are a non-commercial entity and do not even run advertisements!

## Pipeline

• Data <br>
  ○ Load movie and rating CSVs into DataFrames <br>
• ETL <br>
  ○ Split movies into separate “name” and year” columns <br>
  ○ Create an average ratings dataset with group by <br>
  ○ Merge DataFrames together <br>
• EDA <br>
  ○ analyze which genres have the most movies <br>
  ○ analyze which genres have the highest ratings <br>
• Create genre recommendation model <br>
  ○ Split dataset into train and test <br>
  ○ Extract features from genre column with CountVectorizer <br>
  ○ Compute similarity <br>
• Create regression model to predict rating based on genre <br>
  ○ Use CountVectorizer to again extract ‘genre’ as our X <br>
  ○ Rating column as our y <br>
  ○ Train, fit, evaluate <br>
  
## Feature Importance

One of the key parts of our model is the importance of different Genre features in predicting a movie’s rating, which is tricky to do manually due to many movies having several genre features. While no genre results in an easy high score, the drama, documentary, and particularly the horror genre proved to be by far the best indicators of a high user score. <br>

![alt text](https://github.com/jstnkuo/Movie-Recommender-Model/blob/main/Image_video/feature_importance.png) <br>

## Exploratory Data Analysis

To find potential inefficiencies between movies rated and movies created, we also did a count of each genre combination represented in movies. While dramas and documentaries are as highly represented as one would expect based on their feature importance, horror movies are surprisingly edged out in representation by comedies, making it easily the least represented relative to user preference. <br>

![alt text](https://github.com/jstnkuo/Movie-Recommender-Model/blob/main/Image_video/movie_count.png) <br>
<br>
While certain genre features were more predictive for higher ratings, the top genre combinations in the data set ended up surprisingly close. All of the top ten highest rated combinations finished with a mark in the 3.5 range with no outliers. The top 3 genre combinations are animation/drama, documentary/war, and animation/comedy/fantasy. <br>
<br>
![alt text](https://github.com/jstnkuo/Movie-Recommender-Model/blob/main/Image_video/average_rating.png) <br>


## Objective and Value Proposition

Leveraging the analysis of our machine learning model, we have identified specific genres of movies with significantly higher feature importance compared to others: drama, documentary, and horror. By focusing on these genres, especially "horror," movie producers and providers can optimize their products to potentially achieve higher ratings and ensuing increased profitability. <br>

## Next Steps

Our first step would be to Develop our model into a Flask app that maps to an API endpoint, which would make user interactions much simpler. Next, we would split our data into smaller and recent time frames, so we can investigate trends and see if they follow our current conclusions. From there, we would like to increase the scope of our models.  We could add extra features such as user tags to add nuance to our current model, or  even go in a completely different direction by making a recommendation algorithm for users based on watch or rating history. 



