select n_name, avg(c_acctbal), sum(c_acctbal)
from customer join nation 
on c_nationkey = n_nationkey
group by n_name 
order by n_name;