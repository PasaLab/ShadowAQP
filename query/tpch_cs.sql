select c_nationkey, avg(c_acctbal), avg(s_acctbal), sum(c_acctbal), sum(s_acctbal)
from customer join supplier 
on c_nationkey = s_nationkey
group by c_nationkey 
order by c_nationkey;