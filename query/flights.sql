select A_unique_carrier, sum(A_taxi_out), avg(A_taxi_out), sum(A_air_time), avg(A_air_time), sum(A_distance), avg(A_distance)
from flight_a join flight_b 
on A_unique_carrier = B_unique_carrier
group by A_unique_carrier 
order by A_unique_carrier;