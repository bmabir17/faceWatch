import numpy as np, cv2
from flask import Flask, request, Response, jsonify
from face_recognizer_512d.embedder import FaceRecognizer512D_Embedder
model_path = "face_recognizer_512d/recognizer_models/model"
face_encoder_512d = FaceRecognizer512D_Embedder(model_path=model_path, epoch_num='0000', image_size=(112, 112))


def return_encoding(request):
    photos_obj = request.files.getlist('files')
    ## Loop through all the files sent, each file contains unique face of different persons
    for im in photos_obj:
        ## Convert each file(network file type) into np array and decode(recover images from network transmission data) them to jpg using opencv 
        npimg = np.fromstring(im.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        im_str = cv2.imencode('.jpg', img)[1].tostring()
        print(im_str)
    # nparr = np.fromstring(request.data, np.uint8)
    # print(nparr, img)
    # cv2.imshow('test', img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    device_face_encoding = face_encoder_512d.embed_image(img)
    # device_face_encoding = None
    # print(device_face_encoding)
    if device_face_encoding is not None:
    	return jsonify({"Encoding_512D": device_face_encoding.tolist()}), 200
    else:
    	print('Face Not Found')
    	return jsonify({"Encoding_512D": None}), 200
