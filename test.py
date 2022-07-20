import numpy as np
from prompt import AQPPrompt
import logging
logger = logging.getLogger(__name__)

def get_join_group_sql(dataset):
    sql=""

    if dataset=='tpch':
        sql='''
        select c_nationkey,avg(c_acctbal),avg(s_acctbal),sum(c_acctbal),sum(s_acctbal)
        from tpch_35g.customer join tpch_35g.supplier on c_nationkey=s_nationkey
        group by c_nationkey order by c_nationkey
        '''
    elif dataset=='tpch-5':
        sql='''
        select n_name,avg(c_acctbal),sum(c_acctbal)
        from tpch_35g.customer join tpch_35g.nation on c_nationkey=n_nationkey
        group by n_name order by n_name
        '''
    elif dataset=='tpcds':
        sql='''
        select ss_promo_sk,avg(ss_wholesale_cost),avg(ss_list_price),avg(ws_wholesale_cost),avg(ws_list_price),
        sum(ss_wholesale_cost),sum(ss_list_price),sum(ws_wholesale_cost),sum(ws_list_price)
        from tpcds_0_6667g.store_sales join tpcds_0_6667g.web_sales on ss_promo_sk=ws_promo_sk
        group by ss_promo_sk order by ss_promo_sk
        '''
    elif dataset=='tpcds-2':
        sql='''
        select s_store_name,avg(ss_wholesale_cost),avg(ss_list_price),avg(ss_sales_price),avg(ss_ext_sales_price), sum(ss_wholesale_cost),sum(ss_list_price), sum(ss_sales_price),sum(ss_ext_sales_price)
        from tpcds_0_6667g.store_sales join tpcds_0_6667g.store on ss_store_sk=s_store_sk
        group by s_store_name order by s_store_name
        '''
    elif dataset=='movielen':
        sql='''
        select r_movieid,avg(rating),sum(rating)
        from movielen_1m.ratings join movielen_1m.genome_scores on r_movieid=m_movieid
        group by r_movieid order by r_movieid
        '''

    elif dataset=='genome':
        sql='''
        select r_movieid,avg(rating),sum(rating)
        from movielen_25m.ratings join movielen_25m.genome_scores on r_movieid=g_movieid
        group by r_movieid order by r_movieid
        '''
        
    elif dataset=='sdr':
        sql='''
        select prot_category,avg(l4_ul_throughput),avg(l4_dw_throughput),
        avg(l4_ul_packets),avg(l4_dw_packets),
        sum(l4_ul_throughput),sum(l4_dw_throughput),
        sum(l4_ul_packets),sum(l4_dw_packets)
        from sdr_5m.sdr_flow_with_outlier
        join sdr_5m.dim_sub_prot
        on prot_category=protocol_id
        group by prot_category
        order by prot_category
        '''

    elif dataset=='flights':
        sql='''
        select a_unique_carrier, sum(a_taxi_out), avg(a_taxi_out), sum(a_air_time), avg(a_air_time), sum(a_distance), avg(a_distance)
        from flights_300k.flight_a join flights_300k.flight_b on a_unique_carrier = b_unique_carrier
        group by a_unique_carrier order by a_unique_carrier
        '''

    elif dataset=='flights-2':
        sql='''
        select a_unique_carrier, sum(a_taxi_out), avg(a_taxi_out), sum(a_air_time), avg(a_air_time), sum(a_distance), avg(a_distance) 
        from flights_300k.flight_a join flights_300k.flight_b on a_origin_state_abr = b_origin_state_abr 
        group by a_unique_carrier order by a_unique_carrier
        '''

    elif dataset=='census':
        sql='''
        select a_education_num, sum(a_age), avg(a_age), sum(a_hours_per_week), avg(a_hours_per_week), sum(a_fnlwgt), avg(a_fnlwgt)
        from census_150k.adult_a join census_150k.adult_b on a_education_num = b_education_num
        group by a_education_num order by a_education_num
        '''

    elif dataset=='census-2':
        sql='''
        select a_relationship, sum(a_age), avg(a_age), sum(a_hours_per_week), avg(a_hours_per_week), sum(a_fnlwgt), avg(a_fnlwgt) 
        from census_150k.adult_a join census_150k.adult_b on a_education_num = b_education_num 
        group by a_relationship order by a_relationship
        '''
        
    elif dataset=='imdb':
        sql='''
        select a_tconst,avg(a_average_rating),sum(a_average_rating),avg(a_num_votes),sum(a_num_votes)
        from imdb_100000k.ratings a join imdb_100000k.ratings b on a_tconst = b_tconst
        group by a_tconst order by a_tconst
        '''

    if dataset=='test':
        sql='''
        select c_nationkey,avg(c_acctbal),avg(s_acctbal),
        sum(c_acctbal),sum(s_acctbal)
        from test.customer_test join tpch_1m.supplier on c_nationkey=s_nationkey
        group by c_nationkey order by c_nationkey
        '''
    return sql


def get_group_sql(dataset):
    sql=""

    if dataset=='tpch':
        sql='''
        select c_nationkey,avg(c_acctbal),sum(c_acctbal)
        from tpch_1m.customer
        group by c_nationkey order by c_nationkey
        '''

    elif dataset=='tpcds':
        sql='''
        select ss_promo_sk,avg(ss_wholesale_cost),avg(ss_list_price),
        sum(ss_wholesale_cost),sum(ss_list_price)
        from tpcds_1m.store_sales
        group by ss_promo_sk order by ss_promo_sk
        '''

    elif dataset=='movielen' or dataset=='genome':
        sql='''
        select r_movieid,avg(rating),sum(rating)
        from movielen_1m.ratings
        group by r_movieid order by r_movieid
        '''

    elif dataset=='sdr':
        sql='''
        select prot_category,avg(l4_ul_throughput),avg(l4_dw_throughput),
        avg(l4_ul_packets),avg(l4_dw_packets),
        sum(l4_ul_throughput),sum(l4_dw_throughput),
        sum(l4_ul_packets),sum(l4_dw_packets)
        from sdr_1m.sdr_flow_with_outlier
        group by prot_category
        order by prot_category
        '''
    return sql


def get_model_join_option(dataset,k,rate):
    option=""

    if dataset=='tpch':
        option=' with k={} model-{}-./config/train/tpch_customer_torch_cvae.json model-0.1-./config/train/tpch_supplier_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    elif dataset=='tpch-5':
        option=' with k={} model-{}-./config/train/tpch_customer_torch_cvae.json model-0.1-./config/train/tpch_nation_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    elif dataset=='tpcds':
        option=' with k={} model-{}-./config/train/tpcds_ssales_torch_cvae.json model-0.5-./config/train/tpcds_wsales_torch_cvae.json'.format(k,"%.3f" % (rate*2))
    elif dataset=='tpcds-2':
        option=' with k={} model-{}-./config/train/tpcds_ssales_torch_cvae.json model-1.0-./config/train/tpcds_store_torch_cvae.json'.format(k,"%.3f" % (rate))
    elif dataset=='movielen':
        option=' with k={} model-{}-./config/train/movielen_ratings_torch_cvae.json model-1.0-./config/train/movielen_movies_torch_cvae.json'.format(k,"%.3f" % (rate))
    elif dataset=='sdr':
        option=' with k={} model-{}-./config/train/outlier_sdr_flow_torch_cvae.json model-0.1-./config/train/dim_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    elif dataset=='genome':
        option=' with k={} model-{}-./config/train/movielen_ratings_torch_cvae.json model-0.5-./config/train/movielen_genome_torch_cvae.json'.format(k,"%.3f" % (rate*2))
    elif dataset=='census':
        option=' with k={} model-{}-./config/train/census_a_torch_cvae.json model-0.1-./config/train/census_b_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    elif dataset=='census-2':
        option=' with k={} model-{}-./config/train/census2_a_torch_cvae.json model-0.1-./config/train/census2_b_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    elif dataset=='flights':
        option=' with k={} model-{}-./config/train/flights_a_torch_cvae.json model-0.1-./config/train/flights_b_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    elif dataset=='flights-2':
        option=' with k={} model-{}-./config/train/flights2_a_torch_cvae.json model-0.1-./config/train/flights2_b_torch_cvae.json'.format(k,"%.3f" % (rate*10))
    
    return option

def get_model_group_option(dataset,k,rate):
    option=""

    if dataset=='tpch':
        option=' with k={} model-{}-./config/train/tpch_customer_torch_cvae.json'.format(k,"%.3f" % (rate))
    elif dataset=='tpcds':
        option=' with k={} model-{}-./config/train/tpcds_ssales_torch_cvae.json'.format(k,"%.3f" % (rate))
    elif dataset=='movielen':
        option=' with k={} model-{}-./config/train/movielen_ratings_torch_cvae.json'.format(k,"%.3f" % (rate))
    elif dataset=='sdr':
        option=' with k={} model-{}-./config/train/outlier_sdr_flow_torch_cvae.json'.format(k,"%.3f" % (rate))
    elif dataset=='genome':
        option=' with k={} model-{}-./config/train/movielen_ratings_torch_cvae.json'.format(k,"%.3f" % (rate))

    return option

def get_ground_truth_path(dataset):
    ground_truth_path=""

    if dataset=='tpch':
        ground_truth_path = "./ground_truth/tpch-1m/cs_truth.csv"
    elif dataset=='tpcds':
        ground_truth_path = "./ground_truth/tpcds-1m/sw_truth.csv"
    elif dataset=='movielen':
        ground_truth_path = "./ground_truth/movielen-1m/rm_truth.csv"
    elif dataset=='sdr':
        ground_truth_path = "./ground_truth/sdr-1m/sdr_outlier_truth.csv"
    elif dataset=='genome':
        ground_truth_path = "./ground_truth/movielen-1m/rg_truth.csv"

    return ground_truth_path


def get_join_group_ground_truth_path(dataset):
    ground_truth_path=""

    if dataset=='tpch':
        ground_truth_path = "./ground_truth/tpch-35g/cs_truth.csv"
    elif dataset=='tpch-5':
        ground_truth_path = "./ground_truth/tpch-35g/cn_truth.csv"
    elif dataset=='tpcds':
        ground_truth_path = "./ground_truth/tpcds-0.6667g/sw_truth.csv"
    elif dataset=='tpcds-2':
        ground_truth_path = "./ground_truth/tpcds-0.6667g/ss_truth.csv"
    elif dataset=='movielen':
        ground_truth_path = "./ground_truth/movielen-1m/rm_truth.csv"
    elif dataset=='sdr':
        ground_truth_path = "./ground_truth/sdr-1m/sdr_outlier_truth.csv"
    elif dataset=='genome':
        ground_truth_path = "./ground_truth/movielen-1m/rg_truth.csv"
    elif dataset=='census':
        ground_truth_path = "./ground_truth/census/census_truth.csv"
    elif dataset=='census-2':
        ground_truth_path = "./ground_truth/census/census2_truth.csv"
    elif dataset=='flights':
        ground_truth_path = "./ground_truth/flights/flights_truth.csv"
    elif dataset=='flights-2':
        ground_truth_path = "./ground_truth/flights/flights2_truth.csv"
    if dataset=='test':
        ground_truth_path = "./ground_truth/tpch-1m/test.csv"
    return ground_truth_path

def get_group_ground_truth_path(dataset):
    ground_truth_path=""

    if dataset=='tpch':
        ground_truth_path = "./ground_truth/tpch-1m/csnj_truth.csv"
    elif dataset=='tpcds':
        ground_truth_path = "./ground_truth/tpcds-1m/swnj_truth.csv"
    elif dataset=='movielen':
        ground_truth_path = "./ground_truth/movielen-1m/rmnj_truth.csv"
    elif dataset=='sdr':
        ground_truth_path = "./ground_truth/sdr-1m/sdr_outliernj_truth.csv"
    elif dataset=='genome':
        ground_truth_path = "./ground_truth/movielen-1m/rgnj_truth.csv"
    if dataset=='test':
        ground_truth_path = "./ground_truth/tpch-1m/testnj.csv"
    return ground_truth_path


if __name__ == "__main__":
    prompt = AQPPrompt()
    # datasets=['tpch','tpcds','movielen','sdr']
    # datasets=['tpch','tpcds','genome','sdr']
    # datasets=['tpcds','tpcds-2','sdr']
    # datasets=['census','census-2','flights','flights']
    datasets=['census-2']

    average_running_times={t:[] for t in datasets}
    average_query_errors={t:[] for t in datasets}
    rounds=3
    k=1
    res=[]
    query_type='join'
    for t in datasets:
        logger.info("dataset:{}".format(t))
        for r in range(1,11):
            running_times=[]    
            query_errors=[]
            rate=r*0.001
            ### aqp option
            option=''
            if query_type=='join':
                option=get_model_join_option(t,k,rate)
                sql=get_join_group_sql(t)+option
                ground_truth_path=get_join_group_ground_truth_path(t)
            elif query_type=='group':
                option=get_model_group_option(t,k,rate)
                sql=get_group_sql(t)+option
                ground_truth_path=get_group_ground_truth_path(t)

            logger.info("query:{}".format(sql))
            logger.info("ground_truth_path:{}".format(ground_truth_path))

            for i in range(rounds):
                logger.info("round:{}".format(i+1))
                query_error=None
                runing_time,query_error=prompt.execute_approximate_query(sql,ground_truth_path=ground_truth_path)
                running_times.append(round(runing_time,3))
                if query_error is not None:
                    query_errors.append(round(query_error,6))
            average_time=round(np.mean(running_times[:]),3)
            average_running_times[t].append(average_time)
            res.append("dataset:{} rate:{} option:{} running times:{}".format(t,rate,option,running_times))
            res.append("dataset:{} rate:{} option:{} average running time:{}".format(t,rate,option,average_time))
            
            if len(query_errors)>0:
                average_error=round(np.mean(query_errors),6)
                average_query_errors[t].append(average_error)
                res.append("dataset:{} rate:{} option:{} query errors:{}".format(t,rate,option,query_errors))
                res.append("dataset:{} rate:{} option:{} average query error:{}".format(t,rate,option,average_error))
            res.append("------------------------")
        res.append("#####################################################")
    # logger.info("---------final result---------")
    # for t in res:
    #     logger.info(t)
    
    for t in datasets:
        logger.info("dataset:{}\naverage running times:{}\naverage query errors:{}".format(t,average_running_times[t],average_query_errors[t]))
    
