import fasteners
import json
import os


class ResultsJsonDB(object):

    def __init__(self, file_path):
        """
        Simple way of saving results to a json file with multiprocess locks
        This is not atomic or well thought out

        There is the potential for a race condition on first run creating the databse.
        To avoid data loss initialise before running on fast threads.
        :param file_path: path to location on disk
        """
        self._file_path = os.path.abspath(file_path)
        if not os.path.exists(self._file_path):
            with open(self._file_path, "w+") as rfile:
                json.dump({}, rfile)

    @property
    def context_lock(self):
        lock_path = self._file_path + '.context_lock'
        return fasteners.InterProcessLock(lock_path)

    def save(self, value, *keys):
        with self.context_lock:
            with open(self._file_path) as fp:
                db = json.load(fp)
                for key in keys[:-1]:
                    try:
                        xv = xv[str(key)]
                    except KeyError:
                        xv[str(key)] = {}
                        xv = xv[str(key)]

                xv[str(keys[-1])] = value

            with open(self._file_path, "w+") as rfile:
                json.dump(db, rfile, indent=4)

    def load(self):
        with open(self._file_path) as fp:
            db = json.load(fp)
        return db