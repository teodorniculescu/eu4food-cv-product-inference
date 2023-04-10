import argparse
import os
from tqdm import tqdm
import requests
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def main(args):
    # Use a service account
    cred = credentials.Certificate(args.firebase_certificate)
    firebase_admin.initialize_app(cred)

    # Get a reference to the Firestore database
    db = firestore.client()

    keymap_rpid_to_pi = {} # root proposal id -> product id, product name
    none_rpid = []
    for product_doc in tqdm(db.collection('product').get()):
        product_dict = product_doc.to_dict()

        if 'rootProposalID' in product_dict:
            root_proposal_id = product_dict['rootProposalID']
            product_id = product_dict['id']
            product_name = product_dict['info']['name']['en']
            keymap_rpid_to_pi[root_proposal_id] = (product_id, product_name)

            # create directory where the images will be saved
            product_path = os.path.join(args.save_path, product_id)
            os.makedirs(product_path, exist_ok=True)

        else:
            none_rpid.append(product_doc.id)

    if len(none_rpid) != 0:
        print('WARNING: The following products do not have a rootProposalID', none_rpid)


    none_capture = []
    num_captures = -1
    for capture_idx, capture_doc in tqdm(enumerate(db.collection('capture').get())):
        num_captures += 1
        capture_dict = capture_doc.to_dict()

        root_proposal_id = capture_dict['ancestorsData']['rootProposalID']

        if root_proposal_id not in keymap_rpid_to_pi:
            none_capture.append((capture_doc.id, root_proposal_id))
            continue

        product_id, _ = keymap_rpid_to_pi[root_proposal_id]

        front_package_image_url = capture_dict['images']['frontPackagePath']

        response = requests.get(front_package_image_url)

        idx_str = str(capture_idx).zfill(10)
        filename = f'{idx_str}.jpg'

        """
        product_path = os.path.join(args.save_path, product_id, filename)
        with open(product_path, 'wb') as f:
            f.write(response.content)
        """

    if len(none_capture) != 0:
        print(f'WARNING: {len(none_capture)} / {num_captures} not saved')
        print('The following captures have an unknown rootProposalID')
        print(none_capture)

def get_args():
    parser = argparse.ArgumentParser(description='Downloads images from firebase')
    parser.add_argument('--firebase_certificate', type=str, default='firebase_certificate.json')
    parser.add_argument('--save_path', type=str, default='dataset')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
