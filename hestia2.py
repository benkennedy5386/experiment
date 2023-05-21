import argparse
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
import datetime
import tensorflow as tf
import sys
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from utils import label_map_util
#sys.path.append("D:/models/research/slim")
#sys.path.append("D:/models/research/object_detection")
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
import PyPDF2
from fractions import Fraction
import unidecode
import math
import smtplib
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
import smtplib
import yagmail
from pricelist import pricelist
import math
import time


tf.enable_eager_execution()
def persistence2():
    ############################ siding ###############################
    PATH_TO_CKPT3 = 'D:/models/research/object_detection/exported/siding/frozen_inference_graph.pb'
    PATH_TO_LABELS3 = 'D:/models/research/object_detection/exported/siding/labelmap2.pbtxt'
    NUM_CLASSES3 = 7
    boxweights = [0,0,0,0,0,0,0,0]
    areas = 0




    def get_subs(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]


    files = get_subs('D:/models/research/object_detection/server/siding/')


    files = get_subs('D:/models/research/object_detection/server/siding/')
    for file in files:
        PATH_TO_TEST_IMAGES_DIR = 'D:/models/research/object_detection/server/siding/'+file+'/'
        PATH_TO_TEST_INFO = 'D:/models/research/object_detection/server/siding/'+file+'.txt'
                    #rename the photos
        def rename():
            i = 1
            for name in os.listdir(PATH_TO_TEST_IMAGES_DIR):
                dst = PATH_TO_TEST_IMAGES_DIR + "image" + str(i) + ".jpg"
                src = PATH_TO_TEST_IMAGES_DIR + name
                os.rename(src, dst)
                i += 1

            # execute functions

        rename()
        print('renamed ok')

        coount = 4


        TEST_IMAGE_PATHS3 = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, coount) ]

        #simple counters
        board = 0
        brick = 0
        lap = 0
        log = 0
        shingled = 0
        stone = 0
        stucco = 0
        x= 1

        eligibleclass1 = 0
        eligibleclass2 = 0
        eligibleclass3 = 0
        eligibleclass4 = 0
        eligibleclass5 = 0

        area1 = 0
        area2 = 0
        area3 = 0
        area4 = 0
        area5 = 0



        ##########extract the data from PDF into a string
        pdfFileObj = open('eagleview.pdf', 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        raws = (pageObj.extractText())
        raw = str(raws)

        #define units
        sq = 'sqaures'
        sqft = 'square feet'
        lnft = 'linear foot'
        inc = 'as incurred'

        ##################Cut the good variables out, clean them

        #extract total square footage of roofing
        sqfootloc = raw.find('Total Roof Area')
        sqfootraw = raw[sqfootloc+18:sqfootloc+24]
        sqfootcleaned = sqfootraw.replace(",", "")
        sqfoot = float(sqfootcleaned)

        #derive squares from square footage
        squares = float(sqfoot/100)

        #extract number of fascets
        facetsloc = raw.find('Facets')
        facetsraw = raw[facetsloc+8:facetsloc+11]
        facets = float(facetsraw)

        #Extracting pitch
        pitchloc = raw.find('Predominant Pitch')
        pitchraw = raw[pitchloc+20:pitchloc+24]
        pitch = float(Fraction(pitchraw))

        #extract number of stories
        storiesloc = raw.find('Number of Stories')
        storiesraw = raw[storiesloc+20:storiesloc+21]
        stories = float(Fraction(storiesraw))

        #extract ridges
        ridgesloc = raw.find("Total Ridges/Hips")
        ridgesraw = raw[ridgesloc+20:ridgesloc+24]
        ridges = float(ridgesraw)

        #extrac valleys
        valleysloc = raw.find("Valleys")
        valleysraw = raw[valleysloc+10:valleysloc+13]
        valleys = float(valleysraw)

        #extract rake length
        rakesloc = raw.find("Rakes")
        rakesraw = raw[rakesloc+8:rakesloc+11]
        rakes = float(rakesraw)

        #extract eave length
        eavesloc = raw.find('Eave')
        eavesraw = raw[eavesloc+8:eavesloc+12]
        eaves = float(eavesraw)

        xRun = rakes*(math.sqrt(1/(1+(pitch**2))))
        gableend = (.5*pitch*xRun)

        perimeterRoof = eaves + xRun
        overhang = 1.5
        storyHeight = 11
        totalsiding = (((perimeterRoof-4*overhang)*stories*storyHeight)+gableend)
        totalsiding = float(totalsiding)


        #set up inference
        def run_inference_for_single_image(image, sess, tensor_dict):
            if 'detection_masks' in tensor_dict:
                print(detection_masks)
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict

        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT3, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS3)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES3, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        def load_image_into_numpy_array(image):
          (im_width, im_height) = image.size
          return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8)



        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            tf.enable_eager_execution()
            sess.run(tf.global_variables_initializer())
            tf.enable_eager_execution()
            ops = tf.get_default_graph().get_operations()
            tf.enable_eager_execution()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            img = 1
            for image_path in TEST_IMAGE_PATHS3:
              tf.enable_eager_execution()
              image = Image.open(image_path)
              image_np = load_image_into_numpy_array(image)
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')
              output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
              dictionary = output_dict['detection_scores']
              detectedscores = output_dict['detection_scores']
              detectedclasses = output_dict['detection_classes']
              detectedboxes = output_dict['detection_boxes']


              eligible1 = detectedscores[0]
              eligible2 = detectedscores[1]
              eligible3 = detectedscores[2]
              eligible4 = detectedscores[3]
              eligible5 = detectedscores[4]
              eligibleclass1 = detectedclasses[0]
              eligibleclass2 = detectedclasses[1]
              eligibleclass3 = detectedclasses[2]
              eligibleclass4 = detectedclasses[3]
              eligibleclass5 = detectedclasses[4]
              box1 = detectedboxes[0]
              box2 = detectedboxes[1]
              box3 = detectedboxes[2]
              box4 = detectedboxes[3]
              box5 = detectedboxes[4]

             #since this is detection not competition, using lower threshold

              detectionscoreslist = [eligible1, eligible2, eligible3, eligible4, eligible5]

             # print(detectedclasses)
              #print(detectedscores)

              detectedthreshold = .9

              #set up a list

              detectedlist = [9,9,9,9,9,9,9]
              if eligible1 >= detectedthreshold:
                  detectedlist[0] = eligibleclass1
                  area1 = ((box1[2]-box1[0])*(box1[3]-box1[1]))
                  print(area1)
              if eligible2 >= detectedthreshold:
                  detectedlist[1] = eligibleclass2
                  area2 = ((box2[2]-box2[0])*(box2[3]-box2[1]))
                  print(area2)
              if eligible3 >= detectedthreshold:
                  detectedlist[2] = eligibleclass3
                  area3 = ((box3[2]-box3[0])*(box3[3]-box3[1]))
              if eligible4 >= detectedthreshold:
                  detectedlist[3] = eligibleclass4
                  area4 = ((box4[2]-box4[0])*(box4[3]-box4[1]))
              if eligible5 >= detectedthreshold:
                  detectedlist[4] = eligibleclass5
                  area5 = ((box5[2]-box5[0])*(box5[3]-box5[1]))

              if area1 > 0:
                  areas = areas+area1+area2+area3+area4+area5
                  boxweights[eligibleclass1] = (boxweights[eligibleclass1]+area1)
                  boxweights[eligibleclass2] = (boxweights[eligibleclass2]+area2)
                  boxweights[eligibleclass3] = (boxweights[eligibleclass3]+area3)
                  boxweights[eligibleclass4] = (boxweights[eligibleclass4]+area4)
                  boxweights[eligibleclass5] = (boxweights[eligibleclass5]+area5)
                  print(boxweights)




              print('Reviewing image# :', x)


              if 1 in detectedlist:
                  print('board siding found in photo: ', x)
              if 2 in detectedlist:
                  print('brick siding found in photo: ', x)
              if 3 in detectedlist:
                  print('lap siding found in photo: ', x)
              if 4 in detectedlist:
                  print('log siding found in photo :', x)
              if 5 in detectedlist:
                  print('shingled siding found in phot :', x)
              if 6 in detectedlist:
                  print('stone siding found in photo :', x)
              if 7 in detectedlist:
                  print('stucco found in photo :', x)

              featuresclasses = [0,'board', 'brick', 'lap', 'log', 'shingled', 'stone', 'stucco']


              x = x+1






        heavyweights = sum(boxweights)
        if heavyweights > 0 :
            shareboard = float((boxweights[1]/heavyweights))
            sharebrick = float((boxweights[2]/heavyweights))
            sharelap = float((boxweights[3]/heavyweights))
            sharelog = float((boxweights[4]/heavyweights))
            shareshingled = float((boxweights[5]/heavyweights))
            sharestone = float((boxweights[6]/heavyweights))
            sharestucco = float((boxweights[7]/heavyweights))

            print('The total share of board siding is ',shareboard)
            print('The total share of brick siding is: ',sharebrick)
            print('The total share of lap siding is: ',sharelap)
            print('The total share of log siding is: ',sharelog)
            print('THe total share of shingled siding is: ',shareshingled)
            print('The total share of stone siding is :', sharestone)
            print('THe total area of stucco siding is :', sharestucco)


            print('siding processed ',file)

        #open workbook

        wb = openpyxl.load_workbook('C:/Users/benke/Desktop/odin/template/siding.xlsx')
        sheet = wb.get_sheet_by_name('Estimate')
        sheet2 = wb.get_sheet_by_name('PriceList')

        f=open(PATH_TO_TEST_INFO, "r")
        contentss=f.read()
        lines = contentss.splitlines()
        print(lines)
        clientname = lines[1]
        clientemail = lines[2]
        clientphone = lines[3]
        recipientemail = lines[4].strip()
        latlong = lines[5]
        clientaddress = lines[6]
        f.close()
        #insert logo
        img = openpyxl.drawing.image.Image('C:/Users/benke/Desktop/odin/template/logo.jpeg')
        img.anchor = 'A1'
        sheet.add_image(img)

        for row, (key, price) in enumerate(pricelist.items(), start=2):
            sheet2 [f"A{row}"] = key
            sheet2 [f"B{row}"] = price

        #drop down menus
        sheet2["A1"] = "Key"
        sheet2["B1"] = "Price"

        dvsiding = DataValidation(type="list", formula1='"Board and Batten,Masonry Siding,Vinyl Lap Siding,Log Siding,Shingled Siding,Stone Siding,Stucco,House Wrap"', allow_blank=True)
        sheet.add_data_validation(dvsiding)

        sheet['A12'] = 'House Wrap'
        val0 = sheet['A12']
        dvsiding.add(val0)
        sheet['B12'] = round(totalsiding)
        sheet['C12'] = 'sq ft'
        sheet['D12'] = '=VLOOKUP(A12, PriceList!A1:B100, 2, FALSE)'
        sheet['E12'] = 'per sq ft'
        sheet['G12'] = '=b12*d12'

        #counter for cells
        y = 0

        if shareboard >= .0001:

            val1 = sheet['A14']
            val1.value = "Board and Batten"
            val12 = 'A14'
            dvsiding.add(val1)
            sheet['B14'] = round(totalsiding*shareboard)
            sheet['C14'] = 'sq ft'
            sheet['D14'] = '=VLOOKUP('+val12+', PriceList!A1:B100, 2, FALSE)'
            sheet['E14'] = 'per sq ft'
            sheet['G14'] = '=b12*d12'

            y = y+2

        if sharebrick >= .0001:

            sheet['A'+str(14+y)] = 'Masonry Siding'
            val2 = sheet['A'+str(14+y)]
            val21 = 'A'+str(14+y)
            dvsiding.add(val2)
            sheet['B'+str(14+y)] = round(totalsiding*sharebrick)
            sheet['C'+str(14+y)] = 'sq ft'
            sheet['D'+str(14+y)] = '=VLOOKUP('+', PriceList!A1:B100, 2, FALSE)'
            sheet['E'+str(14+y)] = 'per sq ft'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)

            y = y+2

        if sharelap >= .0001:

            val3 = sheet['A'+str(14+y)]
            val3.value = "Vinyl Lap Siding"
            dvsiding.add(val3)
            val31 = 'A'+str(14+y)
            sheet['B'+str(14+y)] = round(totalsiding*sharelap)
            sheet['C'+str(14+y)] = 'sq ft'
            sheet['D'+str(14+y)] = '=VLOOKUP('+val31+', PriceList!A1:B100, 2, FALSE)'
            sheet['E'+str(14+y)] = 'per sq ft'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)

            y = y+2

        if sharelog >= .0001:

            val4 = sheet['A'+str(14+y)]
            val4.value = "Log Siding"
            val41 = 'A'+str(14+y)
            dvsiding.add(val4)
            sheet['B'+str(14+y)] = round(sharelog*totalsiding)
            sheet['C'+str(14+y)] = 'sq ft'
            sheet['D'+str(14+y)] = '=VLOOKUP('+val41+' PriceList!A1:B100, 2, FALSE)'
            sheet['E'+str(14+y)] = 'per sq ft'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)

            y = y+2

        if shareshingled >= .0001:

            val5 = sheet['A'+str(14+y)]
            val5.value = "Shingled Siding"
            dvsiding.add(val5)
            val51 = 'A'+str(14+y)
            sheet['B'+str(14+y)] = round(shareshingled*totalsiding)
            sheet['C'+str(14+y)] = 'sq ft'
            sheet['D'+str(14+y)] = '=VLOOKUP('+val51+', PriceList!A1:B100, 2, FALSE)'
            sheet['E'+str(14+y)] = 'per sq ft'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)

            y = y+2

        if sharestone >= .0001:

            val6 = sheet['A'+str(14+y)]
            val6.value = "Stone Siding"
            dvsiding.add(val6)
            val61 = 'A'+str(14+y)
            sheet['B'+str(14+y)] = round(sharestone*totalsiding)
            sheet['C'+str(14+y)] = 'square foot'
            sheet['D'+str(14+y)] = '=VLOOKUP('+val61+', PriceList!A1:B100, 2, FALSE)'
            sheet['E'+str(14+y)] = 'per square foot'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)

            y = y+2

        if sharestucco >= .0001:
            val7 = sheet['A'+str(14+y)]
            val7.value = "Stucco Siding"
            dvsiding.add(val7)
            val71 = 'A'+str(14+y)
            sheet['B'+str(14+y)] = round(sharestucco*totalsiding)
            sheet['C'+str(14+y)] = 'square foot'
            sheet['D'+str(14+y)] = '=VLOOKUP('+val71+', PriceList!A1:B100, 2, FALSE)'
            sheet['E'+str(14+y)] = 'per square foot'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)

            y = y+2


        #debriscost
        debrishaulunitcosts = pricelist.get("Debris Haul")
        dumpsterunitcost = pricelist.get("Dumpster")
        if totalsiding <= 500:
            debriscost = debrishaulunitcost
            dumpsterquant = 1
        if totalsiding > 500:
            dumpsterquant = round(totalsiding/1250)
            debriscost = dumpsterquant*dumpsterunitcost

            sheet['A'+str(y+14)] = 'Debris Removal'
            sheet['B'+str(y+14)] = dumpsterquant
            sheet['C'+str(y+14)] = 'each'
            sheet['D'+str(y+14)] = debriscost
            sheet['E'+str(y+14)] = 'each'
            sheet['G'+str(14+y)] = '=B'+str(14+y)+'*d'+str(14+y)


        #Subtotal
        sumint = 'G'+str(y+14)
        sheet['E'+str(y+16)] = 'Subtotal'
        sheet['F'+str(y+16)] = '  = '
        sheet['G'+str(y+16)] = '=sum(G12:'+sumint+')'


        #tax
        taxint = 'G'+str(y+16)
        sharematerial = '.4'
        sheet['E'+str(y+18)] = 'Tax'
        sheet['F'+str(y+18)] = ' = '
        sheet['G'+str(y+18)] = '='+taxint+'*.0825*'+sharematerial

        #grandtotal
        totalint1 = 'G'+str(y+16)
        totalint2 = 'G'+str(y+18)
        sheet['E'+str(y+20)] = 'Grand Total'
        sheet['F'+str(y+20)] = ' = '
        sheet['G'+str(y+20)] = '='+totalint1+'+'+totalint2

        #extract user info


        sheet['B8'] = clientname
        sheet['B9'] = clientaddress
        sheet['E8'] = clientphone
        sheet['E9'] = clientemail

        wb.save(clientname+'.xlsx')



        import yagmail

        yag = yagmail.SMTP(user='ben5386simplifyai@gmail.com', password='Bruce78!')
        attach = 'D:\\models\\research\\object_detection\\'+clientname+'.xlsx'
        body = 'Estimate for '+clientname
        subject = 'Please find attached your estimate for '+clientname
        yag.send(to = recipientemail, subject = subject, contents = [body, attach])

        import shutil

        if not os.path.exists('D:/models/research/object_detection/sent/siding/'+file+'/'):
            os.mkdir('D:/models/research/object_detection/sent/siding/'+file+'/')
        shutil.copy('D:/models/research/object_detection/'+clientname+'.xlsx', 'D:/models/research/object_detection/sent/siding/'+file+'/'+clientname+'.xlsx')
        shutil.copy('D:/models/research/object_detection/server/siding/'+file+'.txt', 'D:/models/research/object_detection/sent/siding/'+file+'/'+file+'.txt')
        if os.path.exists('D:/models/research/object_detection/'+clientname+'.xlsx'):
            os.remove('D:/models/research/object_detection/'+clientname+'.xlsx')
        if os.path.exists('D:/models/research/object_detection/server/siding/'+file+'.txt'):
            os.remove('D:/models/research/object_detection/server/siding/'+file+'.txt')

        import glob

        srcDir = PATH_TO_TEST_IMAGES_DIR
        dstDir = 'D:/models/research/object_detection/sent/siding/'+file+'/'

        if os.path.isdir(srcDir) and os.path.isdir(dstDir) :
            for filePath in glob.glob(srcDir + '\*'):
                shutil.copy(filePath, dstDir);
        else:
            print("srcDir & dstDir should be Directories")

        shutil.rmtree('D:/models/research/object_detection/'+file+'/')



        print('siding processed ',file)


persistence2()
