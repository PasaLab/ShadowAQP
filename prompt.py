import time
from cmd import Cmd
from utils.sql_parser import parse_query
from utils.model_sampling import query_multi_sampling, query_sampling
from utils.evaluation import compare_aggregation, compare_aggregation_norm
from pyspark.sql import SparkSession
from tabulate import tabulate
import threading
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='./exp1_logs/census2.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

class AQPPrompt(Cmd):
    def __init__(self):
        super(AQPPrompt, self).__init__()
        self.prompt = 'maqp> '
        self.intro = "Welcome to MAQP: a data-driven AQP system with Mixed AQP methods! Type 'exit' to exit!"
        self.query = " "
        self.init_session()

    def init_session(self):
        # self.spark = SparkSession.builder.appName("SparkSQLSampling").master("yarn").enableHiveSupport().getOrCreate()

        # self.spark = SparkSession.builder.appName("SparkSQLSampling").master("yarn").config("spark.driver.memory", "20g").config(
        #     "spark.driver.memory", "20g").config("spark.executor.instances", 5).config("spark.executor.cores", 10).enableHiveSupport().getOrCreate()
        
        # self.spark = SparkSession.builder.appName("SparkSQLSampling").master("yarn").config("spark.driver.memory", "20g").config(
        #     "spark.driver.memory", "20g").enableHiveSupport().getOrCreate()
        
        # self.spark = SparkSession.builder.appName("SparkSQLSampling").master("yarn").config("spark.executor.instances", 5).config("spark.executor.cores", 10).enableHiveSupport().getOrCreate()

        # self.spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        
        # self.spark.sql(sql).collect()
        # self.spark.catalog.listColumns("test", "default")
        self.spark =None

    # print the exit message.
    def do_exit(self, inp):
        '''exit the application.'''
        print("MAQP closed successfully.")
        return True

    def execute_exact_query(self, query):
        logger.info("Execute exact query.")
        start_time = time.perf_counter()
        res = self.spark.sql(query)
        # res.explain()
        res=res.toPandas()
        print(tabulate(res, headers='keys', tablefmt='psql',showindex=False))
        end_time=time.perf_counter()
        print("time elapsed:{}s".format((end_time - start_time)))
        logger.info("Exact query execution time elapsed:{}s".format(( end_time- start_time)))
        return end_time-start_time

    def execute_approximate_query(self, query,ground_truth_path=None):
        logger.info("Execute approximate query processing.")
        # parse query
        start_time = time.perf_counter()
        query = parse_query(query, self.spark)
        logger.info("parse time:{}".format(time.perf_counter() - start_time))
        # print(query)
        # print('-------------------------')
        # for table, query_table in query.query_tables_dict.items():
        #     print(query_table)
        #     print('-------------------------')

        ### sampling and estimating
        start_time = time.perf_counter()
        multi_sample_times=query.multi_sampling_times
        if multi_sample_times<=1:       ### no multi sampling 
            result,training_time = query_sampling(query)
        else:        ### with multi sampling 
            results=[]
            training_times=[]
            threads = []
            for i in range(multi_sample_times):
                logger.info("multi_sampling No.{} epoch".format(i))
                thread = threading.Thread(target=query_multi_sampling,
                                        args=(query, results,training_times))
                threads.append(thread)
                thread.start()
            for t in threads:
                t.join()
            result = pd.concat(results).groupby(level=0).mean()
            training_time=np.sum(training_times)
            
        # result,training_time = query_sampling(query)
        print(tabulate(result, headers='keys', tablefmt='psql',showindex=False))
        end_time=time.perf_counter()
        print("time elapsed:{}s".format((end_time - start_time)))
        print("training time elapsed:{}s".format(training_time))

        ### evaluation query error
        query_error=None
        if ground_truth_path is not None:
            # if len(query.group_bys)>0:
            #     result = result.set_index([t.attr_name.lower() for t in query.group_bys])
            result.columns = result.columns.str.lower()
            diff = compare_aggregation(result, ground_truth_path, True)
            diff_norm = compare_aggregation_norm(result, ground_truth_path, True)
            logger.info("relative error:\n{}".format(diff))
            # print("relative error normalized:\n{}".format(diff_norm))
            logger.info("relative error average: {}".format(
                diff.values.sum() / diff.size))
            logger.info("relative error normalized average: {}".format(
                diff_norm.values.sum() / diff_norm.size))
            query_error=diff_norm.values.sum() / diff_norm.size

        logger.info("AQP time:{}".format(
            (end_time - start_time-training_time)))
        logger.info("Saving info")
        return ((end_time - start_time-training_time),query_error)

    ### process the query
    def default(self, input_str):
        if ";" not in input_str:
            self.query = self.query + input_str + " "
        else:
            self.query += input_str.split(";")[0]
            with_idx = self.query.upper().rfind('WITH')
            if with_idx == -1:
                # if self.query.lstrip()[0:5].lower() == 'exact':
                self.execute_exact_query(self.query)
            else:
                self.execute_approximate_query(self.query)
            self.query = ""

    # deal with KeyboardInterrupt caused by ctrl+c
    def cmdloop(self, intro=None):
        print(self.intro)
        while True:
            try:
                super(AQPPrompt, self).cmdloop(intro="")
                break
            except KeyboardInterrupt:
                # self.do_exit("")
                print("MAQP closed successflly.")
                return True

    do_EOF = do_exit


if __name__ == "__main__":
    p = AQPPrompt()
    p.cmdloop()
