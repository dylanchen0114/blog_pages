---
title: " Code Notes"
comments: true
mathjax: true
share: true
toc: true
---

#### Hive

```sql
dfs -du -h 'path'  -- show file in path in pretty format
dfs -rm -r 'directory'  -- delete directory

array(col1, col2)  -- combine two columns as array column
sort_array(array(col1, col2))  -- sort elements in array 
row_number() over(partition by col1 order by col2 desc)  -- row number
if(col1 is not null, col1, col2)  -- if-else to each element in column
nvl(col, 0) 
COLLECT_SET(col1)/COLLECT_LIST(col1)  -- group by combine elements
```

```sql
-- create table

-- 1
use ${sourcedb};
create table if not exists p_u_to_interest
(
    user_id string comment '用户id可以是pin，uuid，openid',
    cid3   array<string> comment '三级品类',
    brand  array<string> comment '品牌',
    pwd    array<string> comment '产品词',
    ext    array<string> comment '扩展属性'
) PARTITIONED BY (dt STRING)
ROW FORMAT DELIMITED
STORED AS ORC
TBLPROPERTIES("orc.compress"="SNAPPY"); 

alter table p_u_to_interest drop if exists partition(dt='$day15');

insert overwrite table ${sourcedb}.p_u_to_interest partition (dt = '${dt}')
select * from table


-- 2
drop table if exists tmp_p_u_to_interest_2;
create table tmp_p_u_to_interest_2 
location '/user/recsys/recpro/tmp.db/tmp_p_u_to_interest_2'
as 
select * from table
```



#### Hadoop

```
hadoop fs -get 'path/file_name'  # get hdfs partition to current directory
```



#### Linux

```shell
sz/rz  # download/upload file


# control exit status
echo "${HQL}"
hive -e "${HQL}"

if [ $? -ne 0 ]
then
    echo "conbine failed..."
    exit 2;
else
    echo "conbine done."
fi
```



#### Spark

```
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import *


conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("basic_feature").enableHiveSupport().getOrCreate()

# read data from HDFS table
table = spark.table("tmp.tmp_user_cid3_interest_set")

# create user defined function
def function():
	pass

function_udf = F.udf(lambda x, y: function(x, y), ArrayType(StringType()))

# add a new column
df_origin.withColumn('col_name', function_udf(col1, col2)).

# write to HDFS table
table.write.option("path", '/user/recsys/rec/tmp.db/tmp_user_interest_similarity_cid3').\
    saveAsTable("tmp.tmp_user_interest_similarity_cid3")


spark-submit --num-executors 200 --driver-memory 16g --executor-memory 20g --executor-cores 6 --master yarn-client similarity_calculate.py
```

