from main2 import main
import os

Path = "/Users/jyy/code/pytorch/wind/detect/input/"

all_img = os.listdir(Path)
if not os.path.exists("./output"):
    os.makedirs("./output")

for img in all_img:
    main(
        input_img=Path+img,
        output_img=f"./output/{img}",
        model_path=r'./best.pt',
        size_stride=[(4000,2000)],
        confidence=0.5,
        exclude=0.5)
    print(f"Done {img}")