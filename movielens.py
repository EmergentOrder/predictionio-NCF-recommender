import pyspark
from pyspark.sql.functions import explode
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.storagelevel import StorageLevel
import pypio
from pyspark.sql import utils
from pypio.data import PEventStore
import atexit
from datetime import datetime, timedelta
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

ITEM_COLUMN = "itemid"
TIMESTAMP_COLUMN = "timestamp"
USER_COLUMN = "userid"
RATING_COLUMN = "rating"

SparkContext._ensure_initialized()

spark = SparkSession.builder.master("local[*]").getOrCreate()

sc = spark.sparkContext
sql = spark.sql
def pio_cleanup():
    sc.stop()
    sc._jvm.org.apache.predictionio.workflow.CleanupFunctions.run()
atexit.register(pio_cleanup)

def run_pio_workflow(model, userdict, itemdict, orig_sys_args):
    sys.argv = orig_sys_args
    template_engine = sc._jvm.org.example.vanilla.VanillaEngine
    template_engine.modelRef().set(model)
    template_engine.userdictRef().set(userdict)
    template_engine.itemdictRef().set(itemdict)
    main_args =  utils.toJArray(sc._gateway, sc._gateway.jvm.String, sys.argv)
    create_workflow = sc._jvm.org.apache.predictionio.workflow.CreateWorkflow
    sc.stop()
    create_workflow.main(main_args)

sqlContext = spark._wrapped
sqlCtx = sqlContext

app_name = 'NCF'
event_names = utils.toJArray(sc._gateway, sc._gateway.jvm.String, ['purchased-event'])

p_event_store = PEventStore(spark._jsparkSession, sqlContext)
event_df = p_event_store.find(app_name, entity_type='user', target_entity_type='item', event_names=event_names)
ratings = event_df.toPandas().rename(index=str, columns={'entityId': 'userid', 'targetEntityId': 'itemid', 'eventTime': 'timestamp'})

#For running with eval only, drop dupe user-item interactions and users with < 2 interactions

ratings = ratings.drop_duplicates(subset=["userid", "itemid"], keep="last")
ratings = ratings[ratings.duplicated(subset=['userid'], keep=False)]

ratings['rating'] = 1

ratings['userid'] = pd.to_numeric(ratings['userid'].str[5:]).astype(int)
ratings['itemid'] = pd.to_numeric(ratings['itemid'].str[6:]).astype(int)
ratings['timestamp'] = pd.to_numeric(ratings['timestamp'])

#TODO: Hashing trick here instead of dicts
user_id = ratings[['userid']].drop_duplicates().reindex()
user_id['userIdMapped'] = np.arange(len(user_id))

ratings = pd.merge(ratings, user_id, on=['userid'], how='left')
item_id = ratings[['itemid']].drop_duplicates()
item_id['itemIdMapped'] = np.arange(len(item_id))

ratings = pd.merge(ratings, item_id, on=['itemid'], how='left')
ratings = ratings[['userIdMapped', 'itemIdMapped', 'userid', 'itemid', 'rating', 'timestamp']]


user_id_dict = user_id.to_dict()['userIdMapped']
item_id_dict = item_id.to_dict()['itemIdMapped']
user_map = {int(v):int(user_id_dict[k]) for k,v in user_id.to_dict()['userid'].items()}
item_map = {int(v):int(item_id_dict[k]) for k,v in item_id.to_dict()['itemid'].items()}

DATASET_TO_NUM_USERS_AND_ITEMS = {
    "ml-1m": (len(user_map), len(item_map))
}

#training_dataset = (
#    tf.data.Dataset.from_tensor_slices(
#        (
#            tf.cast(ratings['userid'].values, tf.int64),
#            tf.cast(ratings['itemid'].values, tf.int64),
#            tf.cast(ratings['timestamp'].values, tf.int64)
#            
#        )
#    )
#)

