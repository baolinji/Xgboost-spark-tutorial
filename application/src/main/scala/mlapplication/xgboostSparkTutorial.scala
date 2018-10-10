package mlapplication

/**
  * Created with AItech.
  * Description: 
  * User: baolj
  * Date: 2018-05-10
  * Time: 16:55
  */

import breeze.linalg.DenseVector
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import com.microsoft.ml.spark.LightGBMClassifier
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.sql.types.DoubleType


object xgboostSparkTutorial {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apche").setLevel(Level.INFO)
    System.setProperty("hadoop.home.dir", "F:\\hadoop-2.7.6")

    val spark = SparkSession.builder().master("local")
      .appName("xgbTutorial")
      .getOrCreate()

    /**
      * Load iris data from hdfs or local file.
      */
    val iris_data = spark.read.format("csv")
      .option("header", "true")
      .load("hdfs:/user/bigdata_userprofile/blj/iris.csv")

    val iris_data_1 = iris_data.select(
      iris_data.col("SepalLength").cast(DoubleType).as("SepalLength"),
      iris_data.col("SepalWidth").cast(DoubleType).as("SepalWidth"),
      iris_data.col("PetalLength").cast(DoubleType).as("PetalLength"),
      iris_data.col("PetalWidth").cast(DoubleType).as("PetalWidth"),
      iris_data.col("Species"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
      .setOutputCol("features")

    val train_iris_data = assembler.transform(iris_data_1)

    val splits = train_iris_data.randomSplit(Array(0.7, 0.2, 0.1), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    val valid = splits(2)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(train_iris_data)

    val labelIndexer = new StringIndexer()
      .setInputCol("Species")
      .setOutputCol("label")
      .fit(train_iris_data)

    val labelConverter = new IndexToString()
      .setInputCol("predict")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setPredictionCol("predict")
      .setProbabilityCol("probability")
      .setImpurity("gini")
      .setMinInstancesPerNode(1)
      .setNumTrees(1)
      .setSeed(2018L)
      .setMaxDepth(7)

    val lightgbm = new LightGBMClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setPredictionCol("predict")
      .setProbabilityCol("probability")

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setPredictionCol("predict")
      .setProbabilityCol("probability")
      .setSeed(2018L)

    val xgb = new XGBoostClassifier(
      Map("eta" -> 0.1f,
        "max_depth" -> 2,
        "objective" -> "multi:softprob",
        "num_class" -> 3,
        "num_round" -> 100,
        "num_workers" -> 2
      )
    )
    xgb.setFeaturesCol("indexedFeatures")
    xgb.setLabelCol("label")
    xgb.setPredictionCol("predict")
    xgb.setProbabilityCol("probability")

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, xgb, labelConverter))
      .fit(train)

    val result = pipeline.transform(test)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predict")
      .setMetricName("weightedRecall")
    val Recall = evaluator.evaluate(result)
    println("Recall = " + Recall)

    val Precision = evaluator.setMetricName("weightedPrecision").evaluate(result)
    println("Precision = " + Precision)

    val accuracy = evaluator.setMetricName("accuracy").evaluate(result)
    println("accuracy = " + accuracy)

    val f1 = evaluator.setMetricName("f1").evaluate(result)
    println("f1 = " + f1)

    result.select("Species", "probability", "predict", "predictedLabel").show(50, false)
  }
}
