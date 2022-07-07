import os
from ProgressBar import Progress_Bar

class colors:
    GREEN = '\033[92m' #GREEN
    YELLO = '\033[93m' #YELLOW
    RED = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

class File_Control(object):

    @staticmethod
    def __getNumberOfFiles(dirname):
        num_of_dir = 0
        for _, _, _ in os.walk(dirname, followlinks=True):
            num_of_dir = num_of_dir + 1
        return num_of_dir

    @staticmethod
    def searchFilesInDirectory(dirname):
        files = []
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext=='.jpg':
                files.append(full_filename)
        return files

    @staticmethod
    def searchAllFilesInDirectory(dirname, ext, progressbar = False):
        """
        Get all files in directory and also sub-directories
        Parameters
            - dirname : name of the directory
            - ext : the file extention want to return
            - progressbar(option) : print progressbar
        """
        filenames = []
        i = 0
        num_of_dir = File_Control.__getNumberOfFiles(dirname)
        
        if progressbar:
            print(colors.GREEN + "Searching All Files in Directory.... \t\t directory name : " + dirname + colors.RESET)
            for path, _, files in os.walk(dirname, followlinks=True):
                i = i+1
                Progress_Bar.printProgressBar(i, num_of_dir)
                for name in files:
                    full_filename = os.path.join(path, name)
                    get_ext = os.path.splitext(full_filename)[-1]
                    if get_ext == ("." + ext):
                        filenames.append(os.path.join(path, name))
                #Progress_Bar.printCurrentProcess(full_filename)
        else:
            for path, _, files in os.walk(dirname, followlinks=True):
                for name in files:
                    full_filename = os.path.join(path, name)
                    get_ext = os.path.splitext(full_filename)[-1]
                    if get_ext == ("." + ext):
                        filenames.append(os.path.join(path, name))
        return filenames

    @staticmethod
    def searchAllFilesInDirectoryByDir(dirname, ext, progressbar = False):
            """
            Get all files in directory and also sub-directories
            Parameters
                - dirname : name of the directory
                - ext : the file extention want to return
                - progressbar(option) : print progressbar
            """
            files_by_path = []
            i = 0
            num_of_dir = File_Control.__getNumberOfFiles(dirname)
            temp_path = ""
            if progressbar:
                print(colors.GREEN + "Searching All Files in Directory.... \t\t directory name : " + dirname + colors.RESET)
                for path, _, files in os.walk(dirname, followlinks=True):
                    i = i+1
                    Progress_Bar.printProgressBar(i, num_of_dir)
                    if temp_path != path:
                        temp = []
                    for file in files:
                        full_filename = os.path.join(path, file)
                        get_ext = os.path.splitext(full_filename)[-1]
                        if get_ext == ("." + ext):
                            temp.append(os.path.join(path, file))
                    temp_path = path
                    if temp != []:
                        files_by_path.append(temp)

            else:
                for path, _, files in os.walk(dirname, followlinks=True):
                    if temp_path != path:
                        temp = []
                    for file in files:
                        full_filename = os.path.join(path, file)
                        get_ext = os.path.splitext(full_filename)[-1]
                        if get_ext == ("." + ext):
                            temp.append(os.path.join(path, file))
                    temp_path = path
                    if temp != []:
                        files_by_path.append(temp)

            return files_by_path

    @staticmethod
    def printFileStructure(dirname, depth = 0):
        if depth == 0:
            print(colors.RED + dirname + colors.RESET)
        filenames = os.listdir(dirname)
        for filename in filenames:
            ext = os.path.splitext(filename)[-1]
            if os.path.isdir(dirname + "/" + filename):
                print("│", end="")
                for i in range(depth):
                    print("   │", end="")
                print("┣" + filename)
                File_Control.printFileStructure(dirname + "/" + filename, depth+1)