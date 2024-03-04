from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import ultralytics.engine.results
import os, rasterio
from pyproj import Transformer

Image.MAX_IMAGE_PIXELS = None

def cut_images(input_file:str, output_url:str, size:int, stride:int)->tuple:
    file_name = input_file.split('/')[-1]
    with Image.open(input_file) as im:
        width, height = im.size
        for i in range(0, width-size, stride):
            for j in range(0, height-size, stride):
                box = (i, j, i+size, j+size)
                cropped = im.crop(box)
                cropped.save(output_url+file_name.split('.')[0] + f'_{i}_{j}_{stride}.' + file_name.split('.')[-1])
        return im.size

def compute_iou(box1, box2):
    """计算两个框的交并比"""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def remove_overlaps(boxes, iou_threshold):
    """移除重合度高的框"""
    # 根据面积大小排序
    # boxes = sorted(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    # 按照置信度排序
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    keep = []
    while boxes:
        box = boxes.pop(0)
        keep.append(box)
        boxes = [b for b in boxes if compute_iou(box, b) < iou_threshold]
    return keep

def storm_detect(input_img: str, model_path: str, size: int, stride: int,confidence:float=0.5):
    model = YOLO(model_path)
    if not os.path.exists("./temp/input"):
        os.makedirs("./temp/input")
    img_width,img_height = cut_images(input_img, output_url="./temp/input/", size=size, stride=stride)
    all_files = os.listdir("./temp/input")
    detect_result = []
    for file in all_files:
        img = Image.open(f"./temp/input/{file}")
        origin_keypoint = (int(file.split('_')[-3]),int(file.split('_')[-2]))
        results:ultralytics.engine.results.Results = model(img,save=True)
        conf = results[0].boxes.conf.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        obj_types = results[0].boxes.cls.cpu().numpy()
        for i in range(len(conf)):
            if conf[i] > confidence:
                x1,y1,x2,y2 = xyxy[i]
                x1 = int(x1 + origin_keypoint[0])
                y1 = int(y1 + origin_keypoint[1])
                x2 = int(x2 + origin_keypoint[0])
                y2 = int(y2 + origin_keypoint[1])
                detect_result.append((x1,y1,x2,y2,conf[i],obj_types[i]))
    return detect_result
 
def display_result(input_img:str, output_img:str, detect_result:list):
    with Image.open(input_img) as im:
        draw = ImageDraw.Draw(im)
        for item in detect_result:
            x1, y1, x2, y2, obj_conf,obj_type = item
            obj_type = "cyclone" if int(obj_type)==0  else "anticyclone"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=20)
            text_position = (x1, y1 - 10)
            label = f"{obj_type}: {obj_conf:.2f}"
            draw.text(text_position, label, fill="red", font=ImageFont.truetype("font.ttf", 200))
        im.save(output_img)

def transform_CRS(detect_result:list,input_img:str):
    with rasterio.open(input_img) as src:
        transform = src.transform
    transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)

    result = []
    for item in detect_result:
        x1, y1, x2, y2, obj_conf,obj_type = item
        size = ((x2-x1),(y2-y1))
        x = (x1+x2)/2
        y = (y1+y2)/2
        x, y = transform * (x, y)
        lon, lat = transformer.transform(x, y)
        result.append((lon, lat, size, obj_conf,obj_type))
    return result

def save_result(detect_result:list, output_file:str="output.txt"):
    with open(output_file,'w') as f:
        for item in detect_result:
            f.write(str(item)+'\n')
    
def clean_temp():
    all_files = os.listdir("./temp/input")
    for file in all_files:
        os.remove(f"./temp/input/{file}")
    os.rmdir("./temp/input")

def main(input_img: str, output_img: str, model_path: str, size_stride: list[tuple],confidence:float=0.5,exclude:float=0.5):
    all_result = []
    for size,stride in size_stride:
        all_result += storm_detect(input_img, model_path, size, stride,confidence)
        clean_temp()
    fixed_detect_result = remove_overlaps(all_result,exclude)
    display_result(input_img, output_img, fixed_detect_result)
    fixed_detect_result = transform_CRS(fixed_detect_result,input_img)
    save_result(fixed_detect_result, output_img.split('.')[0]+'.txt')

if __name__ == "__main__":
    main(
        input_img="test.tif",
        output_img="output.tif",
        model_path=r'./best.pt',
        size_stride=[(4000,2000)],
        confidence=0.5,
        exclude=0.5)