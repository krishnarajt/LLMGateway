# scron
Simpler-cron job repo, that makes deploying cron jobs easy and straight forward. Removes the hassle of maintaining dangling cron scripts with dependencies, and hardcoded environment variables. 

# Problems
1. With traditional CRON, its hard to see what happened when, there is no concept of history. 
2. its also hard to pause, resume and stop something. 
3. its harder to control with terminal access always required. no UI support. 
4. its unreliable at times. 
5. dependencies are a major issue because if you use python to run a script, it now means you need to have those dependencies installed, and a traditional cron script can only at the maximum, activate an environment you have to maintain somewhere and then trigger your python app. this means you have to maintain dangling scripts with dependencies. 
6. no version control if the scripts are dangling. 
7. adding a new cronjob is easy but what if there are sensitive things like AI keys that you need to store, in which case you cannot hard code them into the cronjob. 
8. for every such requirement, spinning up a new repository, using docker resources for a whole python script becomes redundant very soon the moment you have even 5 python scripts running on your server that require any sort of sensitive env or dependency. 
