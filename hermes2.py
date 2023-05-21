import googleapiclient.discovery
import logging
import os
from google.cloud import storage
from os import makedirs
from os.path import join, isdir, isfile, basename
from firebase import Firebase
import shutil
from os import walk
import pickle
from time import sleep


bucket_name = 'https://myfirstapplication-16270.appspot.com'

def download_info(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    users = []
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        blob = str(blob)
        user = blob[45:73]
        if user not in users:
            users.append(user)


    def list_sub_directories(bucket_name, prefix):
        """Returns a list of sub-directories within the given bucket."""
        service = googleapiclient.discovery.build('storage', 'v1')

        req = service.objects().list(bucket=bucket_name, prefix=prefix, delimiter='/')
        res = req.execute()
        return res['prefixes']

    with open("olds.txt", "rb") as fp:
        old= pickle.load(fp)

    for user in users:
        projects = list_sub_directories(bucket_name='myfirstapplication-16270.appspot.com', prefix=user+'/')

        for project in projects:
            if project in old:
                sadbear = '44'
            if project not in old:

                print('Found new project, downloading')
                trades = [0,0,0,0]
                blobroofing = 'myfirstapplication-16270.appspot.com/'+project+'roofing/'
                projectName = project[-20:-1]
                if os.path.isdir('D:/models/research/object_detection/server/connect/'+projectName) == False:
                    os.mkdir('D:/models/research/object_detection/server/connect/'+projectName)
                    os.mkdir('D:/models/research/object_detection/server/connect/'+projectName+'/roofing/')
                    os.mkdir('D:/models/research/object_detection/server/connect/'+projectName+'/fencing/')
                    os.mkdir('D:/models/research/object_detection/server/connect/'+projectName+'/siding/')
                    os.mkdir('D:/models/research/object_detection/server/connect/'+projectName+'/interior/')

                # Retrieve all blobs with a prefix matching the file.
                storage_client = storage.Client()
                bucket=storage_client.get_bucket(bucket_name)
                # List blobs iterate in folder

                #blobs roofing
                roofingbucket =bucket.list_blobs(prefix=project+'roofing/')
                x = 0
                for blob in roofingbucket:

                   name = 'roof'+str(x)+'.jpg'
                   #destination_uri = name.format(folder, blob.name)
                   blob.download_to_filename('D:/models/research/object_detection/server/connect/'+projectName+'/roofing/'+name)
                   x= x+1


                fencingbucket =bucket.list_blobs(prefix=project+'fencing/')
                x = 0
                for blob in fencingbucket:
                   name = 'fence'+str(x)+'.jpg'
                   #destination_uri = name.format(folder, blob.name)
                   blob.download_to_filename('D:/models/research/object_detection/server/connect/'+projectName+'/fencing/'+name)
                   x= x+1

                #blobs interior
                interiorbucket =bucket.list_blobs(prefix=project+'interior/')
                x = 0
                for blob in interiorbucket:
                   name = 'interior'+str(x)+'.jpg'
                   #destination_uri = name.format(folder, blob.name)
                   blob.download_to_filename('D:/models/research/object_detection/server/connect/'+projectName+'/interior/'+name)
                   x= x+1

                #sidingbucket
                sidingbucket =bucket.list_blobs(prefix=project+'siding/')
                x = 0
                for blob in sidingbucket:
                   name = 'siding'+str(x)+'.jpg'
                   #destination_uri = name.format(folder, blob.name)
                   blob.download_to_filename('D:/models/research/object_detection/server/connect/'+projectName+'/siding/'+name)
                   x= x+1




                ###########extract firebase shit
                config = {
                  "apiKey": "AIzaSyDUrE9pxHzw4qoE-2-MJWI0RiSVcEE2tzs",
                  "authDomain": "myfirstapplication-16270.firebaseapp.com",
                  "databaseURL": "https://myfirstapplication-16270.firebaseio.com",
                  "storageBucket": "myfirstapplication-16270.appspot.com"
                }

                firebase = Firebase(config)

                db = firebase.database()

                admininfo = {}


                filename = project
                nameraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("name").get()
                name = str(nameraw.val())
                addressraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("address").get()
                address = str(addressraw.val())
                phoneraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("phone").get()
                phone = str(phoneraw.val())
                latraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("latitude").get()
                latitude  = str(latraw.val())
                longituderaw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("longitude").get()
                longitude = str(longituderaw.val())
                fenceReplaceraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("fencetoReplace").get()
                fenceReplace = str(fenceReplaceraw.val())
                fenceRepaintraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("fencetoRepaint").get()
                fenceRepaint = str(fenceReplaceraw.val())
                emailraw = db.child("ProjectSubmissions").child("-"+projectName).child("adminInfo").child("email").get()
                email = str(emailraw.val())
                admininfo[filename+"name"] = name
                admininfo[filename+"address"] = address
                admininfo[filename+"phone"] = phone
                admininfo[filename+"latitude"] = latitude
                admininfo[filename+"longitude"] = longitude
                admininfo[filename+"fenceReplace"] = fenceReplace
                admininfo[filename+"fenceRepaint"] = fenceRepaint

                text_file = open(projectName+".txt", "w")
                text_file.write("zzz\n"+name+"zzz\n"+email+"zzz\n"+phone+"zzz\n"+email+"zzz\n"+latitude+" "+longitude+"zzz\n"+address+"zzz\n"+fenceReplace+"zzz\n"+fenceRepaint+"zzz\n12322\n")
                text_file.close()

                print('downloaded admin info correctly for ', project)

                #paint me like your frech girls jack

                shutil.move('D:/models/research/object_detection/'+projectName+".txt", 'D:/models/research/object_detection/server/connect/'+projectName+'/'+projectName+'.txt')

                print('Made ',project)

                old.append(project)
                with open("olds.txt", "wb") as fp:   #Pickling
                    pickle.dump(old, fp)


def running():
    download_info('myfirstapplication-16270.appspot.com')
    sleep(60)
    running()
