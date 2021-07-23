import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline
import os
os.environ["PYSPARK_PYTHON"] = "python3"


# Part1: Data ETL and Data Exploration

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
movies_df = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings_df = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links_df = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags_df = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)

movies_df.show(5)
ratings_df.show(5)
links_df.show(5)
tags_df.show(5)

# some basic counts
tmp1 = ratings_df.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings_df.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

tmp1 = sum(ratings_df.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings_df.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))


ratings_df.show()
movie_ratings=ratings_df.drop('timestamp')


# Data type convert
from pyspark.sql.types import IntegerType, FloatType
movie_ratings = movie_ratings.withColumn("userId", movie_ratings["userId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("movieId", movie_ratings["movieId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("rating", movie_ratings["rating"].cast(FloatType()))

movie_ratings.show()

# ALS model and evaluation
# import package
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

#Create test and train set
(training,test)=movie_ratings.randomSplit([0.8,0.2])

#Create ALS model
als = ALS(
         userCol="userId", 
         itemCol="movieId",
         ratingCol="rating", 
         nonnegative = True, 
         implicitPrefs = False,
         coldStartStrategy="drop"
)

#Tune model using ParamGridBuilder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
 
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [40, 50, 60]) \
            .addGrid(als.maxIter, [5, 10, 15]) \
            .addGrid(als.regParam, [.13, .15, .17]) \
            .build()



# Define evaluator as RMSE
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))



# Build Cross validation 
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

#Fit ALS model to training data
model = cv.fit(training)

#Extract best model from the tuning exercise using ParamGridBuilder
best_model = model.bestModel


#Generate predictions and evaluate using RMSE
predictions=best_model.transform(test)
rmse = evaluator.evaluate(predictions)


#Print evaluation metrics and model parameters
print ("RMSE = "+str(rmse))
print ("**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())

predictions.show()

# model application and performance evaluation
alldata=best_model.transform(movie_ratings)
rmse = evaluator.evaluate(alldata)
print ("RMSE = "+str(rmse))

# recommend movies to users with id: 575 and 232
recommendations = best_model.recommendForAllUsers(5)
recommendations.registerTempTable("recommendations")
recommendations.show()

%sql
SELECT userID, recommendations
FROM recommendations
WHERE userID IN (575, 232)

# find similar movies for movie with id: 471
movieRecs = best_model.recommendForAllItems(10)
movieRecs.registerTempTable("movieRecs")
movieRecs.show()

%sql
SELECT movieID, recommendations
FROM movieRecs
WHERE movieID in (463, 471)