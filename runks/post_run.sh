
cd result
for i in `ls *.txt`; do python3 ../utils/get_avg_time.py $i; done > ../timing.txt
for i in `ls *stats.log`; do python3 ../utils/get_cpu_mem.py $i; done > ../resource_usage.txt
python3 ../utils/merge.py ../timing.txt ../resource_usage.txt  > ../end_result.txt
python3 ../utils/tables.py ../end_result.txt
cd ..
