
import git


git clone https://github.com/vishalrk1/car-numplate-detection.git # type: ignore



from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from pathlib import Path
import os
import uuid
# import easyocr

app = Flask(__name__)

# Load YOLOv5 model
model_path = Path("automatic-number-plate-recognition/best.pt")
cpu_or_cuda = "cpu"
device = torch.device(cpu_or_cuda)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(device)

# Initialize EasyOCR
# ocr = easyocr.Reader(['en'], gpu=False)

@app.route('/detect_number_plate', methods=['POST'])
def detect_number_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400

    # Read the image from the request
    image_file = request.files['image']
    # image_np = np.fromstring(image_file.read(), np.uint8)
    # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), -1)

    # Run inference with YOLOv5
    output = model(image)

    # Results
    result = np.array(output.pandas().xyxy[0])
    plates = []

    for i in result:
        p1 = (int(i[0]),int(i[1]))
        p2 = (int(i[2]),int(i[3]))
        text_origin = (int(i[0]),int(i[1])-5)
        text_font = cv2.FONT_HERSHEY_PLAIN
        color= (0,0,255)
        text_font_scale = 1.25
        #print(p1,p2)

        # Extract region of interest (ROI) for OCR
        roi = image[int(i[1]):int(i[3]), int(i[0]):int(i[2])]

        image = cv2.rectangle(image,p1,p2,color=color,thickness=2) #drawing bounding boxes
        image = cv2.putText(image,text=f"{i[-1]}",org=text_origin,
                            fontFace=text_font,fontScale=text_font_scale,
                            color=color,thickness=2)
    
    # Save the annotated image
    plates_folder = 'webapp/public/assets'
    if not os.path.exists(plates_folder):
        os.makedirs(plates_folder)

    unique_code = uuid.uuid4()
    annotated_image_path = os.path.join(plates_folder, f'{unique_code}.jpg')
    cv2.imwrite(annotated_image_path, image)
        
    # Prepare response
    response = [{'image_code': unique_code}]
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
