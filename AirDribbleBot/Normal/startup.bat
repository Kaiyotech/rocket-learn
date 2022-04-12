REM verify redis is running already (user1@MSI:/$ sudo redis-server /etc/redis/redis.conf)
call ..\..\venv\Scripts\activate.bat
start python learner.py
TIMEOUT 10
FOR /L %%G IN (1,1,8) DO (start python worker.py & TIMEOUT 30)
start python worker.py STREAMER