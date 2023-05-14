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

    optimization_param = None
    if model_name[:3] == 'vgg':
        for param in model.parameters():
            param.requires_grad = False
        linear_layer_position = len(model.classifier) - 1
        in_features = model.classifier[linear_layer_position].in_features
        model.classifier[linear_layer_position] = torch.nn.Linear(in_features, num_classes)
        optimization_param = model.classifier[linear_layer_position].parameters()

    if model_name[:6] == 'resnet':
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        optimization_param = model.fc.parameters()

    return model, optimization_param

def identifyProduct(request):

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
    
    class_names = ['C01_Alpro_Soya', 'C02_Barilla_Bolognese', 'C03_BelVita_Breakfast', 'C04_BenJerrys_ChocolateFudgeBrownie', 'C05_Bonduelle_Corn', 'C06_Bonduelle_PeasCarrot', 'C07_CocaCola_CocaCola', 'C08_CocaCola_Fanta', 'C09_CocaCola_FuzeTea', 'C10_Corona_CoronaExtra', 'C11_Danone_Activia', 'C12_Desperados_Tequila', 'C13_DrOetker_PizzaMozarella', 'C14_DrOetker_PuddingPowder', 'C15_Ferrero_Nutella', 'C16_Gosser_MarzenBeer', 'C17_HaagenDazs_Vanilla', 'C18_Haribo_Goldbears', 'C19_Heineken_HeinekenLagerBeer', 'C20_Heinz_MayonnaiseSeroiuslyGood']
    class_name_to_idx = {'C01_Alpro_Soya': 0, 'C02_Barilla_Bolognese': 1, 'C03_BelVita_Breakfast': 2, 'C04_BenJerrys_ChocolateFudgeBrownie': 3, 'C05_Bonduelle_Corn': 4, 'C06_Bonduelle_PeasCarrot': 5, 'C07_CocaCola_CocaCola': 6, 'C08_CocaCola_Fanta': 7, 'C09_CocaCola_FuzeTea': 8, 'C10_Corona_CoronaExtra': 9, 'C11_Danone_Activia': 10, 'C12_Desperados_Tequila': 11, 'C13_DrOetker_PizzaMozarella': 12, 'C14_DrOetker_PuddingPowder': 13, 'C15_Ferrero_Nutella': 14, 'C16_Gosser_MarzenBeer': 15, 'C17_HaagenDazs_Vanilla': 16, 'C18_Haribo_Goldbears': 17, 'C19_Heineken_HeinekenLagerBeer': 18, 'C20_Heinz_MayonnaiseSeroiuslyGood': 19}
    idx_to_class_name = {0: 'C01_Alpro_Soya', 1: 'C02_Barilla_Bolognese', 2: 'C03_BelVita_Breakfast', 3: 'C04_BenJerrys_ChocolateFudgeBrownie', 4: 'C05_Bonduelle_Corn', 5: 'C06_Bonduelle_PeasCarrot', 6: 'C07_CocaCola_CocaCola', 7: 'C08_CocaCola_Fanta', 8: 'C09_CocaCola_FuzeTea', 9: 'C10_Corona_CoronaExtra', 10: 'C11_Danone_Activia', 11: 'C12_Desperados_Tequila', 12: 'C13_DrOetker_PizzaMozarella', 13: 'C14_DrOetker_PuddingPowder', 14: 'C15_Ferrero_Nutella', 15: 'C16_Gosser_MarzenBeer', 16: 'C17_HaagenDazs_Vanilla', 17: 'C18_Haribo_Goldbears', 18: 'C19_Heineken_HeinekenLagerBeer', 19: 'C20_Heinz_MayonnaiseSeroiuslyGood'}
    num_classes = 20
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
    return {"data": {"productId": class_name}}

