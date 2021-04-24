from fastapi import APIRouter, UploadFile, File, Response
from app.transform import transform_face
from starlette.responses import StreamingResponse
import io
import cv2
import imageio
import base64


router = APIRouter()

@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    img = imageio.imread(content, pilmode='RGB')

    img_result = transform_face(img)
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    _, im_png = cv2.imencode(".png", img_result * 255)
    base64_img = base64.b64encode(im_png)
    return base64_img


@router.get("/test")
async def test():
    img = imageio.imread("./data/example/Joker.jpeg",  pilmode='RGB')

    img_result = transform_face(img)
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    _, im_png = cv2.imencode(".png", img_result * 255)
    
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")