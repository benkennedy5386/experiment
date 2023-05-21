import json
import shutil

import glob
for filename in glob.glob('*.txt'):
   with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
        contents =f.read()
        lines = contents.splitlines()
        print(lines)
        clientname = lines[1]
        clientemail = lines[2]
        clientphone = lines[3]
        recipientemail = lines[4].strip()
        latlong = lines[5]
        clientaddress = lines[6]

Login = '''{
  "Username": "benkennedy5386@gmail.com",
  "Password": "Bruce78!",
}'''

OrderRequest = {
"Client" : recipientemail,
"Address" : clientaddress,
"IsPDF" : "1",
"LatLon" : latlong,
}'''

print(login)
print(OrderRequest)
