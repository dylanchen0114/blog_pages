---
title: " Hadoop-MapReduce"
comments: true
mathjax: true
share: true
toc: true
---

### MapReduce

#### Mapper类

mapper类是一个泛型类型，它有四个形参类型，可以定义输入键，输入值，输出键，输出值的类型；其中输入键大多为长整型LongWritable（相当于java的long类型）；输入值类为Text类型（相当于java的string类型）；输出值若为IntWritable（相当于java的int类型）。

在定义mapper类的4个参数类型之后，会重载map( )方法。map方法有三个参数，key，value和context，其中context用于输出内容的写入。以气温数据集为例，map方法会从输入value中获取年份(Text)与温度(IntWritable)，并将其写到context中。

```java
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class MaxTemperatureMapper
        extends Mapper <LongWritable, Text, Text, IntWritable> {

    private static final int MISSING = 9999;
    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        String line = value.toString();
        String year = line.substring(15,19);
        int airTemperature;
        if(line.charAt(87) == '+') {
            airTemperature = Integer.parseInt(line.substring(88,92));
        } else {
            airTemperature = Integer.parseInt(line.substring(87,92));
        }
        String quality = line.substring(92,93);
        if(airTemperature != MISSING && quality.matches("[01459]")) {
            context.write(new Text(year), new IntWritable(airTemperature));
        }
    }
}
```

