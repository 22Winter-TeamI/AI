import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import shutil
import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse

app = FastAPI()

from network.Transformer import Transformer


def picture(image):

    model=Transformer()
    model.load_state_dict(torch.load('pretrained_model/Paprika_net_G_float.pth'))
    model.eval()
    print('Model loaded!')
    img_size=450
    img_path=image
    img=cv2.imread(img_path)

    T=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size,2),
        transforms.ToTensor()
    ])

    img_input=T(img).unsqueeze(0)
    img_input=-1+2*img_input

    plt.figure(figsize=(16,10))
    plt.imshow(img[:,:,::-1])

    img_output=model(img_input)
    img_output=(img_output.squeeze().detach().numpy()+1.)/2.
    img_output=img_output.transpose([1,2,0])
    plt.figure(figsize=(16,10))
    plt.axis('off')
    plt.imshow(img_output[:,:,::-1])
    plt.savefig('savefig_default.png')   

    return img_output

@app.post("/changeImage")
async def changeStyle(file: UploadFile=File(...)):

    if not os.path.exists('./temp'):
        os.mkdir('./temp')

    print(f"{file.filename}")

    file_path = "temp/"

    with open(f"{file_path}.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(file_path)
    picture(f"{file_path}.png")
    os.remove(f"{file_path}.png")

    return FileResponse("./savefig_default.png")

# @app.post("/changeImage")
# async def changeStyle(file: UploadFile=File(...)):
#     data = await file.read()
#     print(data)
#     # for a,b in range(file):
#         # print(a,b)
#     # image=picture(x)
#     # print(image)
#     return FileResponse("./savefig_default.png")

# @app.post("/changeImage")
# async def baz(request: Request, file: UploadFile = File(...)): # and here to use it
#     # Do something with the file object
#     return {"file": file.filename}

# @app.post("/changeImage")
# async def receive_image(file: UploadFile = File(...)):
#     # do something with the image data
#     data = await file.read()
#     filename = file.filename
#     file_type = file.content_type
#     # ...
#     # save the image to disk or process it
#     with open(filename, 'wb') as f:
#         f.write(data)
#     return {"filename": filename, "file_type": file_type}

# @app.post("/image-convert/")
# async def convert_image(file: UploadFile = File(...)):

#     if not os.path.exists('./temp'):
#         os.mkdir('./temp')
#     print(f"{file.filename}")
#     file_path = "temp/"

#     with open(f"{file_path}.png", "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # read the image data
#     data = await file.read()
#     print(data)
#     # convert image using AI model
#     converted_data = picture(f"{file_path}.png")
#     # converted_data = picture(data)

#     # save the converted image
#     with open("converted_"+file.filename, 'wb') as f:
#         f.write(converted_data)
#     return {"message": "Image successfully converted"}