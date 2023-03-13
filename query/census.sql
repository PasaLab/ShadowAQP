select A_education_num, sum(A_age), avg(A_age), sum(A_hours_per_week), avg(A_hours_per_week), sum(A_fnlwgt), avg(A_fnlwgt)
from adult_a join adult_b 
on A_education_num = B_education_num
group by A_education_num 
order by A_education_num;