select ss_promo_sk, avg(ss_wholesale_cost), avg(ss_list_price), avg(ws_wholesale_cost), avg(ws_list_price), sum(ss_wholesale_cost), sum(ss_list_price), sum(ws_wholesale_cost), sum(ws_list_price)
from store_sales join web_sales 
on ss_promo_sk = ws_promo_sk
group by ss_promo_sk 
order by ss_promo_sk;