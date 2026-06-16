import cv2                                     #PyTorch is the foundation, facenet-pytorch is a specialized toolkit 
import torch                                   #built on top of it, and MTCNN is a specific tool within that toolkit used to find faces.
from facenet_pytorch import MTCNN              #MTCNN Stands for Multi-task Cascaded Convolutional Networks.
                                               #It is a specific AI model included within the facenet-pytorch library. Its sole job is Face Detection and Alignment.


#Telling the AI to use NVIDIA GPU
#To shift heavy math from CPU to RTX GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#initializing MTCNN for face detection
mtcnn= MTCNN(keep_all=True, device=device)
#keep_all= True makes sure all faces and not just largest one is detected
cap=cv2.VideoCapture(0)
#connection to default camera of the system
while True:
    ret, frame = cap.read()
    if not ret: break
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks = True)
    #boxes variable contains the coordinates of detected faces
    #probs variable contains the confidence score of detected faces
    
    #drawing boxes:
    if boxes is not None:
        for box, prob, pts in zip(boxes, probs, landmarks):
            #checking confidence threshold
            if prob > 0.90:
                cv2.rectangle(frame, 
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), 
                          (0,255,0),2)
                #drawing landmarks (small circles for eyes, nose, mouth)
                for point in pts:
                    cv2.circle(frame, 
                               (int(point[0]), int(point[1])),
                               2, (0,0,255), -1)    
                cv2.putText(frame, f'{prob:.2f}', (int(box[0]), int(box[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow('Face Detection - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
#end of program
