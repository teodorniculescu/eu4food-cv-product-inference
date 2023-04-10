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

    keymap_pi_to_pn = {}
    for product_doc in tqdm(db.collection('product').get()):
        product_dict = product_doc.to_dict()

        product_id = product_dict['id']
        product_name = product_dict['info']['name']['en']
        keymap_pi_to_pn[product_id] = product_name

        # create directory where the images will be saved
        product_path = os.path.join(args.save_path, product_id)
        os.makedirs(product_path, exist_ok=True)




    #none_capture = []

    nonexistent_mission_capture = []
    proposal_capture = []
    nocaptureid_mission_capture = []
    num_captures = -1
    capture_idx = -1
    for capture_doc in tqdm(db.collection('capture').get()):
        num_captures += 1
        capture_dict = capture_doc.to_dict()

        capture_id = capture_doc.id

        #print(json.dumps(capture_dict, indent=4, sort_keys=True, default=str))

        if capture_dict['isProposal']:
            proposal_capture.append(capture_doc.id)
            continue

        mission_id = capture_dict['missionID']

        mission_doc = db.collection('mission').document(mission_id).get()

        if not mission_doc.exists:
            nonexistent_mission_capture.append(capture_doc.id)
            continue
        
        mission_dict = mission_doc.to_dict()

        product_id = None
        image_url = None
        for item in mission_dict['items']:
            if item['captureID'] == capture_id:
                product_id = item['productID']
                image_url = item['itemImagePath']
                break

        if product_id is None:
            nocaptureid_mission_capture.append(capture_doc.id)
            continue

        if product_id not in keymap_pi_to_pn:
            print(product_id, 'not in keymap')


        response = requests.get(image_url)

        capture_idx += 1
        idx_str = str(capture_idx).zfill(10)
        filename = f'{idx_str}.jpg'

        product_path = os.path.join(args.save_path, product_id, filename)
        with open(product_path, 'wb') as f:
            f.write(response.content)

        """

        root_proposal_id = capture_dict['ancestorsData']['rootProposalID']

        if root_proposal_id not in keymap_rpid_to_pi:
            none_capture.append(capture_doc.id)
            continue

        product_id, _ = keymap_rpid_to_pi[root_proposal_id]

        front_package_image_url = capture_dict['images']['frontPackagePath']

        response = requests.get(front_package_image_url)

        idx_str = str(capture_idx).zfill(10)
        filename = f'{idx_str}.jpg'
        """

        """
        product_path = os.path.join(args.save_path, product_id, filename)
        with open(product_path, 'wb') as f:
            f.write(response.content)
        """

    if len(nonexistent_mission_capture) + len(proposal_capture) + len(nocaptureid_mission_capture) != 0:
        print(f'WARNING: {len(nonexistent_mission_capture) + len(proposal_capture) + len(nocaptureid_mission_capture)} / {num_captures} not saved')
        print(f'non existent missions are {len(nonexistent_mission_capture)}')
        print(f'proposals are {len(proposal_capture)}')
        print(f'mission without captureID {len(nocaptureid_mission_capture)}')

def get_args():
    parser = argparse.ArgumentParser(description='Downloads images from firebase')
    parser.add_argument('--firebase_certificate', type=str, default='firebase_certificate.json')
    parser.add_argument('--save_path', type=str, default='dataset')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
