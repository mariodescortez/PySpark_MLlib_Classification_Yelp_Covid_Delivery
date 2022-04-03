# Databricks notebook source
# Mario Cortez | Nour Azar | Tatiane Dutra
# Big Data Tools
# Master of Science in Big Data and Analytics for Business 2021-2022

# COMMAND ----------

# DBTITLE 1,Summary
# Data Preprocessing and Exploratory Analysis of Table
### Business
### Checkin
### Tips
### Review
### User
### Covid

# Model Fitting 
### Variables Selection
### Model Fitting with Cross Validation
### Models Performance

# Analysis 
### Lift Curve
### Business Profilling

# COMMAND ----------

# install packages
!pip install missingno
!pip install nltk
!pip install xgboost
!pip install sklearn
!pip install wordcloud
!pip install spark_ml_utils
!pip install Basemap

# COMMAND ----------

import pyspark
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import *
from pyspark.sql.types import StringType, ArrayType, StructType
from pyspark.sql.functions import explode
import pandas as pd
import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
import seaborn as sns
#from mpl_toolkits.basemap import Basemap
#import nltk 
#nltk.download('stopwords')

# Import libraries
from pyspark.sql.functions import *
from pyspark.sql.functions import to_timestamp 
from pyspark.sql.types import *
import matplotlib.pyplot as plt

# COMMAND ----------

# path

# timeline date
date_timeline = to_date(lit("2018-12-31"), "yyyy-MM-dd")

# COMMAND ----------

pd.set_option('display.max_columns', None)

# COMMAND ----------

# DBTITLE 1,Business
# Business Table
business = spark.read.json("/FileStore/tables/parsed_business.json", multiLine = "true")
business.printSchema()

# Creating pandas dataframe to visualize the table
name_list= list(map(lambda x: x.replace("attributes.","" ), business.columns))
business = business.toDF(*name_list)
business_df = business.toPandas()
pd.set_option('display.max_columns', None)
business_df.head(2)

# COMMAND ----------

# Check Null values 
business_df.isnull().sum()
import missingno as msno
msno.bar(business_df)

# The atribute columns have a lot of null values, so they won't be used in the model
# columns to identify the business, won't be considered in the model as well

# COMMAND ----------

from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(14, 8), edgecolor='w')
m = Basemap(llcrnrlon=-130, llcrnrlat=25, urcrnrlon=-65.,urcrnrlat=52., lat_0 = 40., lon_0 = -80)
m.drawmapboundary(fill_color='#A6CAE0', color="black")
m.fillcontinents(color='#e6b800',lake_color='#A6CAE0')
m.drawcountries(color='grey', linewidth=1)
# Show states
m.drawstates(color='lightgrey', linewidth=1)

mloc = m(business_df['latitude'].tolist(), business_df['longitude'].tolist())
m.scatter(mloc[1],mloc[0],color ='red',lw=3,alpha=0.3,zorder=5)

# COMMAND ----------

business_1 = business_df.dropna(subset=['categories'])
business_1
print('Maximum number of category',business_1['categories'].str.split(',').str.len().max())
print('Median category of business',business_1['categories'].str.split(',').str.len().median())
corpus = ' '.join(business_1['categories'])

corpus = pd.DataFrame(corpus.split(','),columns=['categories'])
cnt = corpus['categories'].value_counts().to_frame()[:20]
plt.figure(figsize=(14,8))
sns.barplot(cnt['categories'], cnt.index, palette = 'tab20')
plt.title('Top main categories listing')
plt.subplots_adjust(wspace=0.3)
plt.show()

# COMMAND ----------

# check unique values
print("num observations " + str(business.select(col("business_id")).count()))
print("unique business_id " + str(business.select(col("business_id")).distinct().count()))
print("unique city " + str(business.select(col("city")).distinct().count()))
print("unique state " + str(business.select(col("state")).distinct().count()))
print("unique Is_open " + str(business.select(col("Is_open")).distinct().count()))

# fill null values of city
business.select(col("city")).na.fill(0)

# COMMAND ----------

# Select columns to be used in the model
business_sp = business.select(col("business_id"),col("city"),col("state"),col("stars"),col("review_count"),col("Is_open"),col("categories"))
business_sp = business_sp.withColumnRenamed("review_count","review_count_bus")

# COMMAND ----------

#Extract category
business_sp = business_sp.withColumn('First_category', split(business_sp['categories'], ',').getItem(0))
cat = business_sp.groupby('First_category').count()
cat = cat.where(col("count")>=30).select("First_category",'count')
business_sp = business_sp.withColumn('category', when(col('First_category').isin('Restaurants', 'Food', 'Pizza',
'Fast Food', 'Sandwiches', 'Mexican', 'American', 'Italian', 'Breakfast & Brunch', 'Chinese', 'Burgers', 'Bakeries', 'American (New)', 'Specialty Food', 'Grocery', 'Sushi Bars', 'Desserts', 'Ice Cream & Frozen Yogurt', 'Japanese', 'Cafes', 'Seafood', 'Thai', 'Vietnamese', 'Salad', 'Chicken Wings', 'Diners', 'Delis', 'Indian', 'Caterers', 'Asian Fusion', 'Barbeque', 'Juice Bars & Smoothies', 'Steakhouses', 'Local Flavor', 'Greek', 'Canadian (New)', 'Vegeterian', 'Middle Eastern', 'Korean', 'Mediterranean', 'Hotels', 'Coffee & Tea', 'American (Traditional)'   ), 'Food&Beverage').otherwise(when(col('First_category').isin('Day Spas', 'Beauty & Spas', 'Sporting Goods', 'Gyms', 'Hair Stylists', 'Hair Salons', 'Car Dealers', 'Home Cleaning', 'Home Cleaning', 'Massage', 'Mobile Phones',  'Pet Groomers', 'Jewelry', 'Car Wash', 'Hair Removal', 'Waxing', 'Dry Cleaning & Laundry', 'Pet Stores', "Men's Clothing", 'Optometrists', 'Furniture Stores', 'Accessories', 'Hotels & Travel', 'Home Services', 'Body Shops',   'Fitness & Instruction', 'Car Rental', 'Shopping', 'Skin Care', 'Nail Salons',  'Barbers', 'Active Life', 'Shoe Stores', 'Home & Garden', 'Eyewear & Opticians', 'Massage Therapy', 'Laundry Services', "Women's Clothing", 'Eyelash Service', 'Fashion', 'Cosmetics & Beauty Supply', 'Gyms', 'Sporting Goods', 'Local Services'), 'Beauty and lifestyle'). when(col('First_category').isin('Pubs', 'Bars', 'Beer', 'Event Planning & Services', 'Arts & Entertainment', 'Nightlife', 'Venues & Event Spaces', 'Lounges'), 'Nightlife').when(col('First_category').isin('Local Services', 'Discount Store', 'Tires', 'Real Estate Services', 'Plumbing', 'Auto Repair', 'Landscaping', 'Professional Services', 'Electronics', 'Heating & Air Conditioning/HVAC', 'Convenience Stores', 'Auto Parts & Supplies', 'Contractors', 'Financial Services'), 'Services and appliances').when(col('First_category').isin('Medical Centers', 'Drugstores', 'Pet Services', 'Health & Medical', 'Pets', 'Doctors', 'Dentists', 'Veterinarians'), 'Health').otherwise('other')))
# dummy enconding categories
from pyspark.ml.feature import OneHotEncoder, StringIndexer
#String indexing
indexer = StringIndexer(inputCol="category", outputCol="catNumericIndex")
df = indexer.fit(business_sp).transform(business_sp.select("category"))
#One-hot encoding
ohe = OneHotEncoder().setInputCol("catNumericIndex").setOutputCol("categoryInd2")
df = ohe.fit(df).transform(df)
#  gather the distinct values
distinct_values = list(df.select("category")
                       .distinct()
                       .toPandas()["category"])
for distinct_value in distinct_values:
    function = udf(lambda item: 
                   1 if item == distinct_value else 0, 
                   IntegerType())
    new_column_name = "Category"+'_'+distinct_value
    business_sp = business_sp.withColumn(new_column_name, function(col("category")))

# COMMAND ----------

business_sp = business_sp.drop("categories","First_category","category")
business_sp.display(5)

# COMMAND ----------

# DBTITLE 1,Check-in
# Tip Table 
checkin = spark.read.json("/FileStore/tables/parsed_checkin.json")
checkin.printSchema()

# Creating pandas dataframe to visualize the table
checkin_df = checkin.toPandas()
checkin_df.head(2)

# COMMAND ----------

#Count the number of nulls values per column
from pyspark.sql.functions import isnan, when, count, col
checkin.select([count(when(col(c).isNull(), c)).alias(c) for c in checkin.columns]).show()

# COMMAND ----------

#Count unique values per column
print("num observations " + str(checkin.select(col("business_id")).count()))
from pyspark.sql.functions import isnan, when, count, col
checkin.select([countDistinct(col(c)).alias(c) for c in checkin.columns]).show()

# COMMAND ----------

# formatting date column to date formatt and transform dates in week days
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")
checkin = checkin.withColumn("input_timestamp",to_timestamp(col("date"))).withColumn("week_day", date_format(col("input_timestamp"), "E"))

# filter date according to timeline
checkin = checkin.where(col("date")>date_timeline)
  
checkin.show(1)

# COMMAND ----------

# aggregating data

# dummy enconding week_day
categ = checkin.select('week_day').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('week_day') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
checkin_sp = checkin.select(checkin.columns+exprs)

# group variables
checkin_sp = checkin_sp.groupBy("business_id").agg(count("business_id").alias("count_checkin"),sum("Mon").alias("checkin_Mon"),sum("Tue").alias("checkin_Tue"),sum("Wed").alias("checkin_Wed"),sum("Thu").alias("checkin_Thu"),sum("Fri").alias("checkin_Fri"),sum("Sat").alias("checkin_Sat"),sum("Sun").alias("checkin_Sun"))
checkin_sp.show(5)

# COMMAND ----------

# DBTITLE 1,Tips
# Tip Table 
tip = spark.read.json("/FileStore/tables/parsed_tip.json")
tip.printSchema()

# Creating pandas dataframe to visualize the table
tip_df = tip.toPandas()
tip_df.head(2)

# COMMAND ----------

#Count the number of nulls values per column
from pyspark.sql.functions import isnan, when, count, col
tip.select([count(when(col(c).isNull(), c)).alias(c) for c in tip.columns]).show()

# COMMAND ----------

#Count unique values per column
print("num observations " + str(tip.select(col("business_id")).count()))
from pyspark.sql.functions import isnan, when, count, col
tip.select([countDistinct(col(c)).alias(c) for c in tip.columns]).show()

# COMMAND ----------

# formatting date column to date formatt
tip = tip.withColumn("date",to_date(col("date"),"yyyy-MM-dd HH:mm:ss").alias("date"))

# filter date according to timeline
tip = tip.where(col("date")>date_timeline)

tip.show(1)

# COMMAND ----------

# processing text



# COMMAND ----------

# aggregating data
tip_sp = tip.groupBy("business_id").agg(count("business_id").alias("tip_count"),sum("compliment_count").alias("tip_compliments_total"))
tip_sp.show(5)

# COMMAND ----------

# DBTITLE 1,Review
# review Table 
review = spark.read.json("/FileStore/tables/parsed_review.json")
review.printSchema()

# Creating pandas dataframe to visualize the table
review_df = tip.toPandas()
review_df.head(2)

# COMMAND ----------

from wordcloud import WordCloud
# Word cloud
cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(review_df['text'].astype(str)))
plt.figure(figsize=(10, 7))
plt.imshow(cloud)
plt.axis('off');

# COMMAND ----------

#Count the number of nulls values per column
from pyspark.sql.functions import isnan, when, count, col
review.select([count(when(col(c).isNull(), c)).alias(c) for c in review.columns]).show()

# COMMAND ----------

#Count unique values per column
print("num observations " + str(review.select(col("business_id")).count()))
from pyspark.sql.functions import isnan, when, count, col
review.select([countDistinct(col(c)).alias(c) for c in review.columns]).show()

# COMMAND ----------

# formatting date column to date formatt
review = review.withColumn("date",to_date(col("date"),"yyyy-MM-dd HH:mm:ss").alias("date"))

# filter date according to timeline
review = review.where(col("date")>date_timeline)

review.show(1)

# COMMAND ----------

## processing text



# COMMAND ----------

# calculate the mean centered rating/star of user

# calculate the rating average per user
user_avg = review.groupBy("user_id").agg(mean("stars").alias("start_user_mean"))

# merge user_avg table with review
review = review.join(user_avg,review["user_id"] == user_avg['user_id'],"left")

# calculate the centered mean
review = review.withColumn("start_centered_mean",review['stars'] - review["start_user_mean"])

# COMMAND ----------

# aggregating data
review_sp = review.groupBy("business_id").agg(count("business_id").alias("review_count"),mean("stars").alias("star_mean"),mean("start_centered_mean").alias("start_centered_mean"),mean("useful").alias("review_useful_mean"),mean("funny").alias("review_funny_mean"),mean("cool").alias("review_cool_mean"))

review_sp = review_sp.withColumnRenamed("review_count","review_count_rev")

review_sp.show(5)

# COMMAND ----------

# DBTITLE 1,User
# user Table 
#user = spark.read.json("/FileStore/tables/user.json")
#user.printSchema()

# Creating pandas dataframe to visualize the table
#user_df = user.toPandas()
#user_df.head(2)

# COMMAND ----------

#Count the number of nulls values per column
#from pyspark.sql.functions import isnan, when, count, col
#user.select([count(when(col(c).isNull(), c)).alias(c) for c in user.columns]).show()

# COMMAND ----------

#Count unique values per column
#print("num observations " + str(user.select(col("business_id")).count()))
#from pyspark.sql.functions import isnan, when, count, col
#user.select([countDistinct(col(c)).alias(c) for c in user.columns]).show()

# COMMAND ----------

# merge user with review table
#review_user = review.join(user,review["business_id"] == user['business_id'],"left")

# Creating pandas dataframe to visualize the table
#review_user = review_user.toPandas()
#review_user.head(2)

# COMMAND ----------

# Merge review with user
#review_user = review.select(col("business_id"),col("user_id"))
#review_user = review_user.join(user,review["user_id"] == user['user_id'],"left")
#review_user_sp = review_user.groupBy("business_id").mean()

# COMMAND ----------

# DBTITLE 1,Covid
# user Table 
covid = spark.read.json("/FileStore/tables/parsed_covid.json")
covid.printSchema()

# Creating pandas dataframe to visualize the table
covid_df = covid.toPandas()
covid_df.head(2)

# COMMAND ----------

#Count the number of nulls values per column
from pyspark.sql.functions import isnan, when, count, col
covid.select([count(when(col(c).isNull(), c)).alias(c) for c in covid.columns]).show()

# COMMAND ----------

#Count unique values per column
print("num observations " + str(covid.select(col("business_id")).count()))
print("unique business_id " + str(covid.select(col("business_id")).distinct().count()))

# COMMAND ----------

# Drop duplicates business_id
covid = covid.dropDuplicates(["business_id"])

# COMMAND ----------

# covid table ready to merge
covid_sp = covid.select(col("business_id"),"delivery or takeout")
covid_sp = covid_sp.withColumnRenamed("delivery or takeout","delivery_takeout")

covid_sp = covid_sp.withColumn("delivery_takeout",expr("regexp_replace(delivery_takeout, 'FALSE', 0)")).withColumn("delivery_takeout",expr("regexp_replace(delivery_takeout, 'TRUE', 1)"))

covid_sp.show(4)

# COMMAND ----------

# DBTITLE 1,Final Basetable
# Count values in each table
print("business_sp " + str(business_sp.select(col("business_id")).count()))
print("checkin_sp " + str(business_sp.select(col("business_id")).count()))
print("tip_sp " + str(business_sp.select(col("business_id")).count()))
print("review_sp " + str(business_sp.select(col("business_id")).count()))
print("covid_sp " + str(business_sp.select(col("business_id")).count()))   

# COMMAND ----------

# Merge Tables
basetable = business_sp.join(checkin_sp,on="business_id")  .join(tip_sp,on='business_id').join(review_sp,on='business_id').join(covid_sp,on='business_id')
basetable.display(4)

# COMMAND ----------

basetable.select('delivery_takeout').count()

# COMMAND ----------

basetable.groupBy("delivery_takeout").count().show()

# COMMAND ----------

# basetable.select(countDistinct("ID","Name")).show()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Pipeline
basetable = basetable.withColumn("checkin_Mon",col("checkin_Mon").cast("int")).withColumn("checkin_Tue",col("checkin_Tue").cast("int")).withColumn("checkin_Wed",col("checkin_Wed").cast("int")).withColumn("checkin_Thu",col("checkin_Thu").cast("int")).withColumn("checkin_Fri",col("checkin_Fri").cast("int")).withColumn("checkin_Sat",col("checkin_Sat").cast("int")).withColumn("checkin_Sun",col("checkin_Sun").cast("int")).withColumn("review_count_bus",col("review_count_bus").cast("int")).withColumn("Is_open",col("Is_open").cast("int")).withColumn("count_checkin",col("count_checkin").cast("int")).withColumn("tip_count",col("tip_count").cast("int")).withColumn("tip_compliments_total",col("tip_compliments_total").cast("int")).withColumn("review_count_rev",col("review_count_rev").cast("int")).withColumn("delivery_takeout",col("delivery_takeout").cast("int"))


#col("Category_Nightlife"),col("Category_Health"),col("Category_other"),col("Category_Food&Beverage"),col("Category_Beauty and lifestyle"),col("Category_Services and appliances"))
basetable = basetable.drop("business_id")
basetable.printSchema()

# COMMAND ----------

basetable.show(2)

# COMMAND ----------

#Create categorical variables for gender and class
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

#City
cityIndxr = StringIndexer().setInputCol("city").setOutputCol("cityInd")

#state
stateIndxr = StringIndexer().setInputCol("state").setOutputCol("stateInd")

#One-hot encoding
ohee_catv = OneHotEncoder(inputCols=["cityInd","stateInd"],outputCols=["city_dum","state_dum"])
pipe_catv = Pipeline(stages=[cityIndxr, stateIndxr, ohee_catv])

basetable_final = pipe_catv.fit(basetable).transform(basetable)
basetable_final = basetable_final.drop("cityInd","stateInd")
basetable_final.show(3)

# COMMAND ----------

#Drop city and state from the basetable
basetable_final = basetable_final.drop("city", "state")
basetable_final.show(3)

# COMMAND ----------

#Create a train and test set with a 70% train, 30% test split
basetable_train, basetable_test = basetable_final.randomSplit([0.7, 0.3],seed=123)

print(basetable_train.count())
print(basetable_test.count())

# COMMAND ----------

#Transform the tables in a table of label, features format
from pyspark.ml.feature import RFormula

trainBig = RFormula(formula="delivery_takeout ~ . - business_id").fit(basetable_final).transform(basetable_final)
train = RFormula(formula="delivery_takeout ~ . - business_id").fit(basetable_train).transform(basetable_train)
test = RFormula(formula="delivery_takeout ~ . - business_id").fit(basetable_test).transform(basetable_test)
print("trainBig nobs: " + str(trainBig.count()))
print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))

# COMMAND ----------

#Train a Logistic Regression model
from pyspark.ml.classification import LogisticRegression

#Define the algorithm class
lr = LogisticRegression()

#Fit the model
lrModel = lr.fit(train)

# COMMAND ----------

# ROC Curve for Training set
trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# Precission and Recall
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# COMMAND ----------

# AUC on Test Set
predictions = lrModel.transform(test)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# predictions.show(2)
#sparkDF=spark.createDataFrame(predictions) 
#predictions.coalesce(2).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/FileStore/tables/predictionsLR.csv")

# predictions.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/FileStore/df/Sample.csv")

#predictions.coalesce(1).write.format("com.databricks.spark.json").option("header", "true").option("delimiter", "\t").option("compression", "gzip").save("dbfs:/FileStore/df/fl_insurance_sample.json")

# COMMAND ----------

predictions_df = predictions.toPandas()

# COMMAND ----------

predictions_df.groupby("prediction").mean()

# COMMAND ----------

predictions_df.groupby("prediction")["tip_count"].mean()

# COMMAND ----------

x = predictions_df.groupby("Category_Food&Beverage")["Category_Food&Beverage"].count()
plt.pie(x, labels = ["not_delivery", "delivery"], autopct='%1.0f%%')
plt.title("Category_Food&Beverage")
plt.show()

# COMMAND ----------

x = predictions_df.groupby("Category_Beauty and lifestyle")["Category_Beauty and lifestyle"].count()
plt.pie(x, labels = ["not_delivery", "delivery"], autopct='%1.0f%%')
plt.title("Category_Beauty and lifestyle")
plt.show()

# COMMAND ----------

x = predictions_df.groupby("Category_Nightlife")["Category_Nightlife"].count()
plt.pie(x, labels = ["not_delivery", "delivery"], autopct='%1.0f%%')
plt.title("Category_Nightlife")
plt.show()

# COMMAND ----------

# ROC Curve for Training set
trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# Feature Importance... 
# https://github.com/xinyongtian/py_spark_ml_utils

# COMMAND ----------

# Feature Importance for Logistic Regression in Train Set
training_pred=lrModel.transform(train)
import spark_ml_utils.LogisticRegressionModel_util as lu
lu.feature_importance(lrm_model=lrModel
                      , trainDF=training_pred, trainFeatures='features'
                      , nonzero_only=True ).head(10)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Random Forest
#Exercise: Train a RandomForest model and tune the number of trees for values [150, 300, 500]
from pyspark.ml.classification import RandomForestClassifier

#Define pipeline
rfc = RandomForestClassifier()

#Run cross-validation, and choose the best set of parameters.
rfcModel = rfc.fit(train)


# COMMAND ----------

#Get predictions on the test set
preds = rfcModel.transform(test)
#Get model accuracy
print("accuracy: " + str(evaluator.evaluate(preds)))
#Get AUC
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

#Cast a DF of predictions to an RDD to access RDD methods of MulticlassMetrics
preds_labels = rfcModel.transform(test)\
  .select("prediction", "label")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = MulticlassMetrics(preds_labels)

print("accuracy = %s" % metrics.accuracy)

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

labels = preds.rdd.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

# COMMAND ----------

# DBTITLE 1,Feature Importance
#Prettify feature importances
import pandas as pd
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
ExtractFeatureImp(rfcModel.featureImportances, test, "features").head(10)

# COMMAND ----------



# COMMAND ----------

## Gradient-Boosted Tree Classifier

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

#Performance Evaluation
xgbpred_train = gbtModel.transform(train)
xgbpred_test = gbtModel.transform(test)
evaluator = BinaryClassificationEvaluator()
aucxbg_train = evaluator.evaluate(xgbpred_train)
aucxbg_test = evaluator.evaluate(xgbpred_test)
accuracy_train = xgbpred_train.filter(xgbpred_train.label == xgbpred_train.prediction).count() / float(xgbpred_train.count())
accuracy_test = xgbpred_test.filter(xgbpred_test.label == xgbpred_test.prediction).count() / float(xgbpred_test.count())
print("Gradient Boost Train AUC: " + str(aucxbg_train))
print("Gradient Boost Train Accuracy: " + str(accuracy_train))
print("Gradient Boost Test AUC: " + str(aucxbg_test))
print("Gradient Boost Test Accuracy: " + str(accuracy_test))

# COMMAND ----------

print(gbtModel.explainParams())

# COMMAND ----------

# The cluster ran into problems (terminate in 120 min), and we couldn't finish the Gridsearch for any of the models, we tried reducing the k folds to 3, and the ranges of parameters and its the same problem. 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
# Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)

# COMMAND ----------

# https://www.kaggle.com/code/sudhirnl7/basic-exploration-of-business-review-at-yelp-com
# https://stackoverflow.com/questions/28009370/get-weekday-day-of-week-for-datetime-column-of-dataframe
# https://analyticjeremy.github.io/Databricks_Examples/Write%20to%20a%20Single%20CSV%20File
# https://www.kaggle.com/code/vksbhandary/exploring-yelp-reviews-dataset
# https://sparkbyexamples.com/pyspark/convert-pandas-to-pyspark-dataframe/
# https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
