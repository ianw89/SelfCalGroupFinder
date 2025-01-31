import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import sys

qdel_sent = []

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        abspath = os.path.abspath(event.src_path)
        
        try:
            size_MB = os.path.getsize(abspath) / 1024 / 1024
        except FileNotFoundError:
            return
        
        if abspath.rfind('mock_') != -1 and size_MB > 100: # These shouldn't be larger than 100MB...
            # Now use qstat to find my pbs jobs and the one associated with this file
            # Then qdel that job
            # Then qsub the new job
            # /some/path/mcmc_N/file
            job_num = (int)(abspath.split('mcmc_')[1][0])

            if job_num in qdel_sent:
                if size_MB > 500:
                    print(f'Job {job_num} should already been deleted, now it is {size_MB:.2f}MB.')
                return
            
            #print(f'MODIFIED. Size: {size_MB:.2f}MB.  Large Mock File: {abspath}')

            user = os.getenv('USER')

            # Find the job ID associated with this file
            qstat_output = os.popen(f'qstat -u {user}').read()
            job_id = None
            for line in qstat_output.splitlines():
                if f"ian.emcee{job_num}" in line or f"ian.optuna{job_num}" in line:
                    job_id = line.split()[0]
                    break

            if job_id:
                # Delete the job
                os.system(f'qdel {job_id}')
                print(f'LARGE FILE BUG: Sending qdel to Job {job_id} associated with (Size: {size_MB:.2f}MB) {abspath}.')
                qdel_sent.append(job_num)

            # Submit a new job
            #qsub_command = f'qsub batch{job_num}.pbs'
            #print(f'Running: {qsub_command}')
            #os.system(qsub_command)



    def on_created(self, event):
        abspath = os.path.abspath(event.src_path)
        size_MB = os.path.getsize(abspath) / 1024 / 1024
        if abspath.rfind('mock') != -1 and size_MB > 50:
            print(f'CREATED. Size: {size_MB:.2f}MB.  Large Mock File: {abspath}')

    def on_deleted(self, event):
        abspath = os.path.abspath(event.src_path)
                
        try:
            size_MB = os.path.getsize(abspath) / 1024 / 1024
        except FileNotFoundError:
            return
        
        if event.src_path.rfind('mcmc_') != -1 and size_MB > 100:
            job_num = (int)(abspath.split('mcmc_')[1][0])

            if job_num in qdel_sent:
                qdel_sent.remove((int)(event.src_path.split('mcmc_')[1][0]))
                print(f'Job {job_num} has been removed from the qdel_sent list.')
            
            if len(qdel_sent) > 0:
                print(f'qdel_sent: {qdel_sent}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python watch.py <path_to_monitor>")
        sys.exit(1)

    path_to_monitor = sys.argv[1]
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path_to_monitor, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()