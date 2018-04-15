import os
class Configuration(object):
    def __init__(self,conf):
        self.conf = conf
        pass

    def get_dir_conf_(self,name):
        return self.conf.get(name)

    def get_ell_file_available(self,dir_data):
        filenames = os.listdir(dir_data)
        return [os.path.join(dir_data, filename) for filename in filenames]

    def get_duration_(self):
        return self.conf.get("duration", 2)



