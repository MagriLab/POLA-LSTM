import os 


def make_img_filepath(model_path):
    img_filepath = model_path + "images/"
    if not os.path.exists(img_filepath):
        os.makedirs(img_filepath)
    return img_filepath