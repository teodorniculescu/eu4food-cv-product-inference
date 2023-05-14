import json
import requests
from PIL import Image
from google.cloud import storage
import numpy as np
import cv2
import torch
import urllib.request
import os
import tempfile
from torchvision import transforms


def get_model(model_name, num_classes):
    model = torch.hub.load('pytorch/vision:v0.11.3', model_name, pretrained=True)

    if model_name.startswith('vgg') or model_name.startswith('efficientnet') or model_name.startswith('alexnet') :
        linear_layer_position = len(model.classifier) - 1
        in_features = model.classifier[linear_layer_position].in_features
        module_list = [
                torch.nn.Linear(in_features, num_classes),
                torch.nn.Softmax(dim=1)
                ]
        model.classifier[linear_layer_position] = torch.nn.Sequential(*module_list)

    elif model_name.startswith('resnet') or model_name.startswith('regnet'):
        in_features = model.fc.in_features
        module_list = [
                torch.nn.Linear(in_features, num_classes),
                torch.nn.Softmax(dim=1)
                ]
        model.fc = torch.nn.Sequential(*module_list)

    else:
        raise Exception(f"ERROR: Unknown model {model_name}")

    return model



def get_image_from_request(request):
    request_json = request.get_json()

    #This is the front product image
    #print(request_json["data"]["imagePath"])

    #The bucket where the images will be loaded from
    #bucket_name = "eu4food-public"

    #This is useful if we need to take the images directly from the firebase storage
    #storage_client = storage.Client()
    #source_bucket = storage_client.bucket(bucket_name)
    #blob = source_bucket.get_blob(blob_name)

    #Load an image
    url = request_json["data"]["imagePath"]
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def predict_old(image):
    class_names = ['C01_Alpro_Soya', 'C02_Barilla_Bolognese', 'C03_BelVita_Breakfast', 'C04_BenJerrys_ChocolateFudgeBrownie', 'C05_Bonduelle_Corn', 'C06_Bonduelle_PeasCarrot', 'C07_CocaCola_CocaCola', 'C08_CocaCola_Fanta', 'C09_CocaCola_FuzeTea', 'C10_Corona_CoronaExtra', 'C11_Danone_Activia', 'C12_Desperados_Tequila', 'C13_DrOetker_PizzaMozarella', 'C14_DrOetker_PuddingPowder', 'C15_Ferrero_Nutella', 'C16_Gosser_MarzenBeer', 'C17_HaagenDazs_Vanilla', 'C18_Haribo_Goldbears', 'C19_Heineken_HeinekenLagerBeer', 'C20_Heinz_MayonnaiseSeroiuslyGood']
    class_name_to_idx = {cn: idx for idx, cn in enumerate(class_names)}
    idx_to_class_name = {idx: cn for idx, cn in enumerate(class_names)}
    num_classes = len(class_names)
    image_size = 224

    tmpdir = tempfile.gettempdir()
    filename = os.path.join(tmpdir, 'checkpoint')
    if not os.path.isfile(filename):
        url = "https://firebasestorage.googleapis.com/v0/b/eu4food-public/o/resnet18_checkpoint?alt=media&token=eb7bad43-ae76-42ba-828c-61aaa0a32cda"
        urllib.request.urlretrieve(url, filename)

    model_name = 'resnet18'
    model, _ = get_model(model_name, num_classes)
    device = 'cpu'
    best_acc_checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(best_acc_checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tr_frame = transform(image).to(device)
    output = model(tr_frame[None, ...])
    _, predicted_labels = torch.max(output.data, 1)
    idx = predicted_labels.cpu().numpy()[0]
    class_name = idx_to_class_name[idx]

    return class_name

def predict(image, device='cpu'):
    def preprocess_image(image, image_size, mean, std, use_mean_std):
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

        if use_mean_std:
            transforms_list += [ transforms.Normalize(mean=mean, std=std) ]

        else:
            transforms_list += [ transforms.Lambda(lambda x: x * 255) ]
        
        transform = transforms.Compose(transforms_list)

        image = transform(image)

        return image

    # create temporary directory
    tmpdir = tempfile.gettempdir()

    # get args json
    args_url = 'https://storage.googleapis.com/eu4food-public/best_model/args.json'
    args_filepath = os.path.join(tmpdir, 'args.json')
    urllib.request.urlretrieve(args_url, args_filepath)
    with open(args_filepath) as f:
        args_data = json.load(f)

    # get obj names
    obj_names_url = 'https://storage.googleapis.com/eu4food-public/best_model/obj.names'
    obj_names_filepath = os.path.join(tmpdir, 'obj.names')
    urllib.request.urlretrieve(obj_names_url, obj_names_filepath)
    with open(obj_names_filepath) as f:
        obj_names_data = f.readlines()
    class_names = [cn.rstrip() for cn in obj_names_data]
    class_name_to_idx = {cn: idx for idx, cn in enumerate(class_names)}
    idx_to_class_name = {idx: cn for idx, cn in enumerate(class_names)}
    num_classes = len(class_names)

    # get model
    best_model_url = 'https://storage.googleapis.com/eu4food-public/best_model/best_model.pth'
    best_model_filepath = os.path.join(tmpdir, 'best_model.pth')
    urllib.request.urlretrieve(best_model_url, best_model_filepath)
    model = get_model(args_data['model'], num_classes)
    model.load_state_dict(torch.load(best_model_filepath))
    model.eval()

    # inference
    image = preprocess_image(image, args_data['image_size'], args_data['norm_mean'], args_data['norm_std'], args_data['use_mean_std'])
    image = image.to(device)
    output = model(image[None, ...])
    _, predicted_labels = torch.max(output.data, 1)
    idx = predicted_labels.cpu().numpy()[0]
    class_name = idx_to_class_name[idx]

    return class_name


def identifyProduct(request):
    image = get_image_from_request(request)
    class_name = predict_old(image)
    return {"data": {"productId": class_name}}


filepath = 'dataset_gcloud/20_products/C01_Alpro_Soya/RO/IMG_4012.JPEG'
image = cv2.imread(filepath)
class_name = predict(image)
print(class_name)


