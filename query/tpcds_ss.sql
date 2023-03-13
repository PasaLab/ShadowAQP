select s_store_name, avg(ss_wholesale_cost), avg(ss_list_price), avg(ss_sales_price), avg(ss_ext_sales_price), sum(ss_wholesale_cost), sum(ss_list_price), sum(ss_sales_price), sum(ss_ext_sales_price)
from store_sales join store 
on ss_store_sk = s_store_sk
group by s_store_name
order by s_store_name;