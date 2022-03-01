# call Nauman's function to check if user is in list
# if False, then create user folder to record data
# load in classifier if existing, if not, then ask user to train classifier
# check pandas/sk-learn library

# /Users/adrianyu/opt/anaconda3/envs/maction_potential/bin/python3 "/Users/adrianyu/Desktop/Maction Potential/DataStorageAndLoading.py"


import os
from os import path
import csv
import joblib

#creating folder
#parameter: name of the folder (ie name of user)
#output: nothing - will print error message if folder already exists
def create_folder(username):
    parent_dir = "/Users/adrianyu/Desktop/Maction Potential" #change based on desired location of folder
    directory = username
    path = os.path.join(parent_dir, directory)
    
    #try-catch statement to print error message when folder already exists
    try:
        os.mkdir(path)
        print("Directory '%s' created" % directory)
    except OSError as error:
        print (error)

#writes into csv files under user's folder
#parameters: username, data type (ie Left blink, right blink, or normal blinks) and interval between blinks
#returns: nothing
def write_csv_files(username, typeOfData, interval):        
    
    version = check_version (username, typeOfData, interval)
    
    parent_dir = "/Users/adrianyu/Desktop/Maction Potential/" + username #customize
    fileName = username + "_" + typeOfData + "_" + interval + "_" + str(version) + ".csv"
    path = os.path.join(parent_dir, fileName)
    fileObj = open(path, 'w')

    # create the csv writer
    writer = csv.writer(fileObj)
    
    # write a row to the csv file
    headers = ["Time", "Date", "Event"] #customize based on file format
    writer.writerow(headers)
    
    #reads in data from muse using bluetooth?
    #not sure how the entries/ info are supposed to work, so I tried writing some random things into the file
    numOfEntries = 10
    for row in range (1, numOfEntries + 1):
        rowToWrite = ["Time" + str(row), "Date" + str(row), "Event" + str(row)]
        writer.writerow(rowToWrite)
    
    # close the file
    fileObj.close()

#a function to check for the version of csv file
#parameters: name, data type (ie left blink, right blink or normal blinks), interval of blinks
#return: the version number that the file should generate
def check_version (username, typeOfData, interval):
    
    parent_dir = "/Users/adrianyu/Desktop/Maction Potential/" + username #folder
    
    ver = 1
    while ver > 0:
        fileName = fileName = username + "_" + typeOfData + "_" + interval + "_" + str(ver) + ".csv" 
        path = os.path.join(parent_dir, fileName)
        
        if not check_path(path):
            return ver
        ver += 1

def check_path (path):
    if os.path.isdir(path):
        return True
    else:
        return False

def check_classifier (username):

    fileName = username + ".joblib"
    parent_dir = "/Users/adrianyu/Desktop/Maction Potential/"
    path = os.path.join (parent_dir, fileName)
    
    if check_path (path):
        # not sure why this doesnt work
        joblib.load(fileName)
    # else:
        #ask user for classifier

create_folder("Matthew")
write_csv_files ("Matthew", "RightBlink", "5s")