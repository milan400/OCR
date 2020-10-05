#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:36:07 2020

@author: milan
"""
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os, os.path

import shutil


from threading import Timer

from flask import make_response

import random
import string



import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


app = Flask(__name__,static_folder=os.path.abspath('static'))

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'static/'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Reading the data and processing

def photo_to_xlsx(file):
    img = cv2.imread(file,0)
    img.shape

    plotting = plt.imshow(img, cmap='gray')
    #plt.show()



    #thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    #inverting the image
    img_bin = 255 - img_bin
    #cv2.imwrite('cv_inverting.png',img_bin)

    #Plotting the image to see the output
    plotting = plt.imshow(img_bin, cmap='gray')
    #plt.show()



    #length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100

    #Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel_len))

    #Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    #A kernel of 2*2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))


    #Use vertical kernel to detect and save the vertical lines in image
    image_1 = cv2.erode(img_bin, ver_kernel, iterations =3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations =3)
    #cv2.imwrite("vertical.jpg", vertical_lines)

    #plot the generated image
    plotting = plt.imshow(image_1, cmap='gray')
    #plt.show()

    #Use horizontal kernel to detect and save the vertical lines in image
    image_2 = cv2.erode(img_bin, hor_kernel, iterations =3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations =3)
    #cv2.imwrite("horizontal.jpg", horizontal_lines)

    #plot the generated image
    plotting = plt.imshow(image_2, cmap='gray')
    #plt.show()



    #Combine horizontal and vertical lines in a new third image, with both having same weight
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines,0.5,0.0)

    #Eroding and threholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite("img_vh.jpg", img_vh)

    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    #Plotting the generated image
    plotting = plt.imshow(img_vh, cmap='gray')
    #plt.show()


    #Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    def sort_contours(cnts, method="left-to-right"):
        
        #initialize the reverse flag and sort_index
        reverse = False
        i = 0
        
        #handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
            
        #handle if we are sorting against the y-coordinate rather than 
        #the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i=1
        
        #construct the list of bounding boxes and sort them from top to
        #bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key = lambda b:b[1][i], reverse = reverse))
        
        #return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    #Sort all the contours by top to bottom
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    #Get mean of heights
    mean = np.mean(heights)


    #Create list box to store all boxes in
    box = []

    #Get position (X,Y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        if(w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0),2)
            box.append([x,y,w,h])

    plotting = plt.imshow(image, cmap='gray')
    #plt.show()


    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

    #Sorting the boxes to their respective row and column
    for i in range(len(box)):
        
        if(i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if(box[i][1] <= previous[1]+mean/2):
                column.append(box[i])
                previous = box[i]
                
                if(i == len(box) - 1):
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])
                

    #calculating maximum number of cells

    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if(countcol > countcol):
            countcol = countcol

    # Retrieving the center of each column

    center = [int(row[i][j][0] + row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

    center = np.array(center)
    center.sort()




    finalboxes = []

    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if(len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0], finalboxes[i][j][k][1],finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,1))
                    
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2,cv2.BORDER_CONSTANT, value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
                    
                    dilation = cv2.dilate(resizing, kernel, iterations = 1)
                    erosion = cv2.erode(dilation, kernel, iterations = 1)
                    
                    out = pytesseract.image_to_string(erosion)
                    if(len(out) == 0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                outer.append(inner)
                

    #Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    data = dataframe.style.set_properties(align="left")

    return data


def delete_files():
  dirpath = os.path.join(app.config['UPLOAD_FOLDER'])
  for filename in os.listdir(dirpath):
      if filename.endswith('.gitkeep') !=True:
        filepath = os.path.join(dirpath, filename)
        try:
          shutil.rmtree(filepath)
        except OSError:
          os.remove(filepath)


@app.route('/')
def upload_file():
   return render_template('upload.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        image_dir = (os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        
        generated_data = photo_to_xlsx(image_dir)

        #Converting it in a CSV-File
        file_name_xlsx = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
    
        generated_data.to_excel(os.path.join(app.config['UPLOAD_FOLDER']+file_name_xlsx+".xlsx"))


        images_names = os.listdir('static')
        
        for names in images_names:
            #response = make_response(render_template('result.html', image_names = images_names))
            t = Timer(500,delete_files)
            t.start()
            
            return send_from_directory(directory='static', filename=file_name_xlsx +'.xlsx')


if __name__ == '__main__':
    app.run(debug = True)