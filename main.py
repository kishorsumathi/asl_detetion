import cv2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import time

model=torch.load("efficientnet_b0_trained_v2.pt")
device="cuda" if torch.cuda.is_available() else "cpu"
classes=['A','B','C','D','E','Empty','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
model.to(device)

def on_mouse(event, x, y, flags, userdata):
        global state, p1, p2
        if event == cv2.EVENT_LBUTTONUP:
            if state == 0:
                p1 = (x,y)
                state += 1
            elif state == 1:
                p2 = (x,y)
                state += 1


p1, p2 = None, None
x_1,y_1=0,0
w_1,z_1=0,0
state = 0
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Frame', on_mouse)
cap = cv2.VideoCapture(0)
while True:
    start=time.time()
    ret, frame = cap.read()
    if p1!=None and p2!=None:
        if ret:
            x_1,y_1=p1
            w_1,z_1=p2
            frame_crop = frame[y_1:z_1,x_1:w_1]
    if p1==None or p2==None:
        frame_crop=frame
    
    if state > 1:
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 5)
    if p1!=None and p2!=None:
        image_transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        model.eval()
        cv2.imwrite("frame.jpg",frame_crop)
        frame_crop=Image.open("frame.jpg")
        with torch.inference_mode():
                transformed_image = image_transform(frame_crop).unsqueeze(dim=0)
                target_image_pred = model(transformed_image.to(device))
                target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
                target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        prob=target_image_pred_probs.cpu().numpy()[0][target_image_pred_label.cpu()[0].item()]
        a=f'{prob:.3f}'
        if float(a) > 0.6:
            cv2.putText(frame, classes[target_image_pred_label], (x_1,y_1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (179, 0, 149), 2)

    end=time.time()
    print("[INFO] Total time taken for inferencing :",end-start)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()