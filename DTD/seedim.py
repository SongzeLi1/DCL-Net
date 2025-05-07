import cv2
from PIL import Image

root = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/test_result_type2/pred512x512/psc_honor30_oc_110_rma_0_0.png"
image = cv2.imread(root)
print(image.shape)
