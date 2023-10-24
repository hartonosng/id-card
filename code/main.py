
import cv2
from matplotlib import pyplot as plt
import numpy as np
import utlis
import torch
from find_nearest_box import NearestBox
from pytorch_unet.unet_predict import UnetModel
from pytorch_unet.unet_predict import Res34BackBone
from extract_words import OcrFactory
import extract_words
import os
import time
import argparse
import detect_face
import imgproc

def getCenterRatios(img, centers):
    """
    Calculates the position of the centers of all boxes 
    in the ID card image and Unet Mask relative to the width and height of the image 
    and returns these ratios as a numpy array.
    """
    if(len(img.shape) == 2):
        img_h, img_w = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios
    else :
        img_h, img_w,_ = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios


def matchCenters(ratios1, ratios2):
    """
    It takes the ratio of the centers of the regions 
    included in the mask and CRAFT result on the image 
    and maps them according to the absolute distance. 
    Returns the index of the centers with the lowest absolute difference accordingly
    """

    bbb0 = np.zeros_like(ratios2)
    bbb1 = np.zeros_like(ratios2)
    bbb2 = np.zeros_like(ratios2)
    bbb3 = np.zeros_like(ratios2)

    for i , r2 in enumerate(ratios2):
        bbb0[i] = abs(ratios1[0] - r2)
        bbb1[i] = abs(ratios1[1] - r2)
        bbb2[i] = abs(ratios1[2] - r2)
        bbb3[i] = abs(ratios1[3] - r2)

    sum_b0 = np.sum(bbb0, axis = 1)
    sum_b0 = np.reshape(sum_b0, (-1, 1))
    arg_min_b0 = np.argmin(sum_b0, axis=0)

    sum_b1 = np.sum(bbb1, axis = 1)
    sum_b1 = np.reshape(sum_b1, (-1, 1))
    arg_min_b1 = np.argmin(sum_b1, axis=0)

    sum_b2 = np.sum(bbb2, axis = 1)
    sum_b2 = np.reshape(sum_b2, (-1, 1))
    arg_min_b2 = np.argmin(sum_b2, axis=0)

    sum_b3 = np.sum(bbb3, axis = 1)
    sum_b3 = np.reshape(sum_b3, (-1, 1))
    arg_min_b3 = np.argmin(sum_b3, axis=0)

    return np.squeeze(arg_min_b0), np.squeeze(arg_min_b1), np.squeeze(arg_min_b2),np.squeeze(arg_min_b3)         



def getCenterOfMasks(thresh):
    """
    Find centers of 4 boxes in mask from top to bottom with unet model output and return them
    """
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by size from smallest to largest
    contours = sorted(contours, key = cv2.contourArea, reverse=False)
    
    # contours = contours[-4:] # get the 4 largest contours

    #print("size of cnt", [cv2.contourArea(cnt) for cnt in contours])
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort the 4 largest regions from top to bottom so that we filter the relevant regions
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][1], reverse=False))
    
    detected_centers = []
 
    for contour in cnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        #cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        detected_centers.append((cX, cY))
        #cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)

    return np.array(detected_centers)


def getBoxRegions(regions):
    """
    The coordinates of the texts on the id card are converted 
    to x, w, y, h type and the centers and coordinates of these boxes are returned.
    """
    boxes = []
    centers = []
    x01,y01, x02, y02, x03, y03, x04, y04 = np.int0(regions[0].reshape(-1))
    x11,y11, x12, y12, x13, y13, x14, y14 = np.int0(regions[1].reshape(-1))
    wmax = max(x01,x02,x03,x04,x11,x12,x13,x14)
    x21,y21, x22, y22, x23, y23, x24, y24 = np.int0(regions[2].reshape(-1))
    hmax = abs(min(y21,y22) - max(y23, y24))
    # print(regions)
    
    for box_region in regions:
        x1,y1, x2, y2, x3, y3, x4, y4 = np.int0(box_region.reshape(-1))
        if x1>=100 and x1<=wmax:
            x = min(x1, x3)
            y = min(y1, y2)
            w = abs(min(x1,x3) - max(x2, x4))
            h = abs(min(y1,y2) - max(y3, y4))
            if h<=hmax*1.5:
                bbox = (int(x), w, int(y), h)
                boxes.append(bbox)
                xmin=min(x1,x2,x3,x4)
                xmax=max(x1,x2,x3,x4)
                ymin=min(y1,y2,y3,y4)
                ymax=max(y1,y2,y3,y4)
                print([xmin,xmax,ymin, ymax])   
    print("number of detected boxes", len(boxes))
    
    return np.array(boxes)



if '__main__' == __name__:
    
    parser = argparse.ArgumentParser(description = 'Identity Card Information Extractiion')
    parser.add_argument('--folder_name', default="images", type=str, help='folder that contain tc id images')
    # parser.add_argument('--neighbor_box_distance', default = 50, type = float, help='Nearest box distance threshold')
    parser.add_argument('--face_recognition',  default = "ssd", type = str,   help='face detection algorithm')
    parser.add_argument('--ocr_method',  default = "EasyOcr", type = str,   help='Type of ocr method for converting images to text')
    parser.add_argument('--rotation_interval', default = 30,   type = int, help='Face search interval for rotation matrix')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    args = parser.parse_args()
    
    Folder = args.folder_name # identity card images folder
    ORI_THRESH = 3 # Orientation angle threshold for skew correction
    
    use_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UnetModel(Res34BackBone(), use_cuda)
    # nearestBox = NearestBox(distance_thresh = args.neighbor_box_distance, draw_line=False)
    face_detector = detect_face.face_factory(face_model = args.face_recognition)
    findFaceID = face_detector.get_face_detector()
    #Image2Text = extract_words.ocr_factory(ocr_method = args.ocr_method, border_thresh=3, denoise = False)
    Image2Text =  OcrFactory().select_ocr_method(ocr_method = args.ocr_method, border_thresh=2, denoise = False)
    
    start = time.time()
    end = 0
    for filename in sorted(os.listdir(Folder)):
        
        img = cv2.imread(os.path.join(Folder,filename))
        # img, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)

        # img = imgproc.denormalizeMeanVariance(img)
        img = cv2.resize(img  , (640 , 448))
        img1 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
  
        final_img = findFaceID.changeOrientationUntilFaceFound(img1, args.rotation_interval)
        
        if(final_img is None):
            print(f"No face detected in KTP {filename}")
            break

        final_img = utlis.correctPerspective(final_img)
    
        txt_heat_map, regions = utlis.createHeatMapAndBoxCoordinates(final_img)
        
        # txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2GRAY)
        
        # predicted_mask = model.predict(txt_heat_map)

        # orientation_angle = utlis.findOrientationofLines(predicted_mask.copy())
        # print("Orientation of KTP is {} ".format(orientation_angle))
        
        # if ( abs(orientation_angle) > ORI_THRESH ):
            
        #     print("Absulute orientation_angle is greater than {}".format(ORI_THRESH)  )

        #     final_img = utlis.rotateImage(orientation_angle, final_img)

        #     txt_heat_map, regions = utlis.createHeatMapAndBoxCoordinates(final_img)
            # txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2GRAY)
            # predicted_mask = model.predict(txt_heat_map)

    
        bbox_coordinates = getBoxRegions(regions)
        
        # mask_centers = getCenterOfMasks(predicted_mask)

        # # centers ratio for 4 boxes
        # centers_ratio_mask = getCenterRatios(predicted_mask, mask_centers) 

        # # centers ratio for all boxes
        # centers_ratio_all = getCenterRatios(final_img, box_centers) 
    
        # matched_box_indexes = matchCenters(centers_ratio_mask , centers_ratio_all)
        
        # new_bboxes = nearestBox.searchNearestBoundingBoxes(bbox_coordinates, matched_box_indexes, final_img)
       
        PersonInfo = Image2Text.ocrOutput(filename, final_img, bbox_coordinates)
        # print(bbox_coordinates)
        print(PersonInfo.items())
        for id, val in PersonInfo.items():
            print(id,':' ,val)
        print(" ")
        end = time.time()
        # utlis.displayMachedBoxes(final_img, new_bboxes)
        
        utlis.displayAllBoxes(final_img, bbox_coordinates)
        
    
        # plt.title("final_img")
        # plt.imshow(final_img)
        # plt.imsave("/result/final_imgp.jpg",final_img)
        # final_img = cv2.cvtColor(final_img , cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./result/{filename}.jpg", final_img)
        # plt.show()
    
        
        # plt.title("Predicted Mask")
        # plt.imshow(predicted_mask)
        # # plt.imsave("/result/predicted_mask.jpg", predicted_mask)
        # # cv2.imwrite("./result/predicted_mask.jpg", predicted_mask)
        # plt.show()
    
   
    print("Execution Time:", (end -start))
   
        

