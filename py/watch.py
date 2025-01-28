import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        size_MB = os.path.getsize(event.src_path) / 1024 / 1024
        if event.src_path.rfind('mock') != -1 and size_MB > 50:
            print(f'Large Mock File {event.src_path} has been modified. Size: {size_MB:.2f}MB')

    def on_created(self, event):
        size_MB = os.path.getsize(event.src_path) / 1024 / 1024
        if event.src_path.rfind('mock') != -1 and size_MB > 50:
            print(f'Large Mock File {event.src_path} has been created. Size: {size_MB:.2f}MB')

    #def on_deleted(self, event):
    #    print(f'File {event.src_path} has been deleted.')


if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()