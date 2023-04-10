from tqdm import tqdm
import requests
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

firebase_certificate = 'firebase_certificate.json'

# Use a service account
cred = credentials.Certificate(firebase_certificate)
firebase_admin.initialize_app(cred)

# Get a reference to the Firestore database
db = firestore.client()

rpid_to_pn = {}
none_rpid = []
for product_name in tqdm(db.collection('product').get()):
    product_dict = product_name.to_dict()
    if 'rootProposalID' in product_dict:
        root_proposal_id = product_dict['rootProposalID']
        rpid_to_pn[root_proposal_id] = product_name
    else:
        none_rpid.append(product_name.id)
    # print(json.dumps(product_dict, indent=4, sort_keys=True, default=str))

if len(none_rpid) != 0:
    print('WARNING: The following products do not have a rootProposalID', none_rpid)

for capture in tqdm(db.collection('capture').get()):
    capture_dict = capture.to_dict()
    root_proposal_id = capture_dict['ancestorsData']['rootProposalID']

    # URL path of the image
    front_package_image_url = capture_dict['images']['frontPackagePath']
    # print(capture.id, root_proposal_id, front_package_image_url)

    # Filename to save the image as
    filename = 'my_image.jpg'

    response = requests.get(front_package_image_url)

    with open(filename, 'wb') as f:
        f.write(response.content)

    break

