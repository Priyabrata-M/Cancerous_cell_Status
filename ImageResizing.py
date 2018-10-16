#Image resizing
import cv2
import os

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    
#    print image.shape[:2]
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
#    print "image resized"
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


path = 'VGGNet_GPU/Test_PNG'
i=0
for casefoldername in os.listdir(path): 
    for filename in os.listdir(path+"/"+casefoldername): 
        if filename=="STR.DCM.png":
            os.remove(path+"/"+casefoldername+"/"+"STR.DCM.png")
        elif filename=="str.dcm.png":
            os.remove(path+"/"+casefoldername+"/"+"str.dcm.png")
        elif filename=="Str.dcm.png":
            os.remove(path+"/"+casefoldername+"/"+"Str.dcm.png")
        else:
            i=i+1
            print i
            filePath = path+"/"+casefoldername+"/"+filename
            print filePath
            img = cv2.imread(filePath)
    #        cv2.imshow('image',img)
    #        cv2.waitKey(0)
    #        print filename
            image = image_resize(img, width=224, height = 224)
            cv2.imwrite(filePath,image)
    #
#
#path = 'CT1.2.840.113619.2.55.1.1762532279.1950.1140095497.127.dcm.png'
#img = cv2.imread(path)
#image = image_resize(img, width=64, height = 64)
#cv2.imwrite(path,image)
#cv2.imshow('image',img)
#cv2.waitKey(0)



