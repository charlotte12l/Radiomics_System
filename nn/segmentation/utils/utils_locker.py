import fcntl

class Locker(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.is_locked = False

    def lock(self):
        status = True
        if self.is_locked:
            # already locked
            return self.is_locked
        self.lock_file = open(self.file_name, 'w')
        try:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            status = False
            self.lock_file.close()
        self.is_locked = status
        return status

    def lock_block(self):
        status = True
        if self.is_locked:
            # already locked
            return self.is_locked
        self.lock_file = open(self.file_name, 'w')
        try:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX)
        except IOError:
            status = False
            self.lock_file.close()
        self.is_locked = status
        return status


    def unlock(self):
        if self.is_locked:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.is_locked = False
        return True

