
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf


COLUMNS =["stat_date", "object_id", "buyer_member_id", "expo_cnt", "click_cnt", "label",
 "expo_cnt_30days", "click_cnt_30days", "ctr_cnt_30days",
 "expo_cnt_7days", "click_cnt_7days", "ctr_cnt_7days",
 "offer_expo_cnt_7days", "offer_click_cnt_7days", "offer_ctr_cnt_7days",
 "offer_expo_cnt_30days", "offer_click_cnt_30days", "offer_ctr_cnt_30days",
 "act_cnt_7days", "act_cnt_30days",
 "category_id", "quality_score", "quality_grade", "ordercost", "avg_score",
 "interval", "day", "near_interval"
 "buyer_encoder", "object_encoder", "category_encoder"]

LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["object_id", "buyer_member_id","category_id","quality_grade"]
CONTINUOUS_COLUMNS = ["expo_cnt_30days", "click_cnt_30days", "ctr_cnt_30days",
"expo_cnt_7days", "click_cnt_7days", "ctr_cnt_7days",
"offer_expo_cnt_7days", "offer_click_cnt_7days", "offer_ctr_cnt_7days",
"offer_expo_cnt_30days", "offer_click_cnt_30days", "offer_ctr_cnt_30days",
"act_cnt_7days", "act_cnt_30days","quality_score","ordercost", "avg_score",
"interval", "day"]



def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  object_id = tf.contrib.layers.sparse_column_with_hash_bucket(
      "object_id", hash_bucket_size=100000)
  buyer_member_id = tf.contrib.layers.sparse_column_with_hash_bucket(
      "buyer_member_id", hash_bucket_size=100000)
  category_id = tf.contrib.layers.sparse_column_with_hash_bucket(
      "category_id", hash_bucket_size=1000)
  quality_grade = tf.contrib.layers.sparse_column_with_hash_bucket(
      "quality_grade", hash_bucket_size=100)


  # Continuous base columns.
  expo_cnt_30days = tf.contrib.layers.real_valued_column("expo_cnt_30days")
  click_cnt_30days = tf.contrib.layers.real_valued_column("click_cnt_30days")
  ctr_cnt_30days = tf.contrib.layers.real_valued_column("ctr_cnt_30days")
  expo_cnt_7days = tf.contrib.layers.real_valued_column("expo_cnt_7days")
  click_cnt_7days = tf.contrib.layers.real_valued_column("click_cnt_7days")
  ctr_cnt_7days = tf.contrib.layers.real_valued_column("ctr_cnt_7days")
  offer_expo_cnt_7days = tf.contrib.layers.real_valued_column("offer_expo_cnt_7days")
  offer_click_cnt_7days = tf.contrib.layers.real_valued_column("offer_click_cnt_7days")
  offer_ctr_cnt_7days = tf.contrib.layers.real_valued_column("offer_ctr_cnt_7days")
  offer_expo_cnt_30days = tf.contrib.layers.real_valued_column("offer_expo_cnt_30days")
  offer_click_cnt_30days = tf.contrib.layers.real_valued_column("offer_click_cnt_30days")
  offer_ctr_cnt_30days = tf.contrib.layers.real_valued_column("offer_ctr_cnt_30days")
  act_cnt_7days = tf.contrib.layers.real_valued_column("act_cnt_7days")
  act_cnt_30days = tf.contrib.layers.real_valued_column("act_cnt_30days")
  quality_score = tf.contrib.layers.real_valued_column("quality_score")
  ordercost = tf.contrib.layers.real_valued_column("ordercost")
  avg_score = tf.contrib.layers.real_valued_column("avg_score")
  interval = tf.contrib.layers.real_valued_column("interval")
  day = tf.contrib.layers.real_valued_column("day")

  # Wide columns and deep columns.
  wide_columns = [object_id, buyer_member_id,category_id,quality_grade,
                  tf.contrib.layers.crossed_column([object_id, buyer_member_id],
                                                   hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column(
                      [buyer_member_id, category_id],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([buyer_member_id, quality_grade],
                                                   hash_bucket_size=int(1e6)),
                  ctr_cnt_7days,
                  offer_ctr_cnt_7days
                  ]
  deep_columns = [
      tf.contrib.layers.embedding_column(object_id, dimension=8),
      tf.contrib.layers.embedding_column(buyer_member_id, dimension=8),
      tf.contrib.layers.embedding_column(category_id, dimension=8),
      tf.contrib.layers.embedding_column(quality_grade, dimension=4),
      expo_cnt_30days,
      click_cnt_30days,
      ctr_cnt_30days,
      expo_cnt_7days,
      click_cnt_7days,
      ctr_cnt_7days,
      offer_expo_cnt_7days,
      offer_click_cnt_7days,
      offer_ctr_cnt_7days,
      offer_expo_cnt_30days,
      offer_click_cnt_30days,
      offer_ctr_cnt_30days,
      act_cnt_7days,
      act_cnt_30days,
      quality_score,
      ordercost,
      avg_score,
      interval,
      day,
#      near_interval,
  ]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        dnn_dropout = 0.7,
        fix_global_step_increment_bug=True)
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def input_fn_test(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  #train_file_name, test_file_name = maybe_download(train_data, test_data)
  df_train = pd.read_csv(
      "taohuoyuan_train1.txt",
      names=COLUMNS,index_col=False)
  df_test = pd.read_csv(
      "taohuoyuan_test1.txt",
      names=COLUMNS,index_col=False)

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)
  df_train[LABEL_COLUMN] = df_train["label"].astype(int)
  df_test[LABEL_COLUMN] = df_test["label"].astype(int)
  df_train['category_id'] = df_train.category_id.astype('string')
  df_train['object_id'] = df_train.object_id.astype('string')
  df_train['buyer_member_id'] = df_train.buyer_member_id.astype('string')
  df_train['quality_grade'] = df_train.quality_grade.astype('string')
  df_test['category_id'] = df_test.category_id.astype('string')
  df_test['object_id'] = df_test.object_id.astype('string')
  df_test['buyer_member_id'] = df_test.buyer_member_id.astype('string')
  df_test['quality_grade'] = df_test.quality_grade.astype('string')
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)
  print ("unique2 = %s" %df_test.label.unique())
  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

  for key in sorted(results):
    print("%s: %s" % (key, results[key]))
  res= m.predict_proba(input_fn=lambda: input_fn_test(df_test))
  out = list(zip(list(df_test['label']),list(res)))
  cols = ['True', 'predict']
  df_out = pd.DataFrame(out, columns=cols)
  df_out.to_csv(path_or_buf='pred2.csv', index=False)
FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
