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

    product_doc_dict = {x.id: x for x in db.collection('product').get()}
    capture_doc_dict = {x.id: x for x in db.collection('capture').get()}
    mission_doc_dict = {x.id: x for x in db.collection('mission').get()}

    keymap_pi_to_pn = {}
    keymap_rpid_to_pi = {} 
    for product_doc in tqdm(product_doc_dict.values()):
        product_dict = product_doc.to_dict()

        product_id = product_dict['id']
        product_name = product_dict['info']['name']['en']
        keymap_pi_to_pn[product_id] = product_name

        # create directory where the images will be saved
        product_path = os.path.join(args.save_path, product_id)
        os.makedirs(product_path, exist_ok=True)

        if 'rootProposalID' in product_dict:
            root_proposal_id = product_dict['rootProposalID']
            product_id = product_dict['id']
            keymap_rpid_to_pi[root_proposal_id] = product_id

    url_check = True
    url_check_dict = {}
    images_to_download = []

    nonexistent_mission_capture = []
    proposal_capture = []
    nocaptureid_mission_capture = []
    num_captures = -1


    for capture_doc in tqdm(capture_doc_dict.values()):
        num_captures += 1
        capture_dict = capture_doc.to_dict()

        capture_id = capture_doc.id

        if capture_dict['isProposal']:
            proposal_capture.append(capture_doc.id)
            continue

        mission_id = capture_dict['missionID']

        if not mission_id in mission_doc_dict:
            nonexistent_mission_capture.append(capture_doc.id)
            continue

        mission_doc = mission_doc_dict[mission_id]
        
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

        if url_check:
            if image_url in url_check_dict:
                print(image_url, 'already exists')
                continue
            url_check_dict[image_url] = None
            images_to_download.append((image_url, product_id))


    for capture_doc in tqdm(capture_doc_dict.values()):
        capture_dict = capture_doc.to_dict()

        root_proposal_id = capture_dict['ancestorsData']['rootProposalID']

        if root_proposal_id not in keymap_rpid_to_pi:
            continue

        product_id = keymap_rpid_to_pi[root_proposal_id]

        image_url = capture_dict['images']['frontPackagePath']

        if url_check:
            if image_url in url_check_dict:
                print(image_url, 'already exists')
                continue
            url_check_dict[image_url] = None
            images_to_download.append((image_url, product_id))


    print('save images')
    capture_idx = -1
    #for image_url, product_id in tqdm(images_to_download.items()):
    for image_url, product_id in tqdm(images_to_download):
        capture_idx += 1
        idx_str = str(capture_idx).zfill(10)
        filename = f'{idx_str}.jpg'
        product_path = os.path.join(args.save_path, product_id, filename)

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(product_path, 'wb') as f:
                f.write(response.content)
        else:
            print('request to address', image_url, 'for class', product_id, 'was unsuccessful')


    if len(nonexistent_mission_capture) + len(proposal_capture) + len(nocaptureid_mission_capture) != 0:
        print(f'WARNING: {len(nonexistent_mission_capture) + len(proposal_capture) + len(nocaptureid_mission_capture)} / {num_captures} not saved')
        print(f'non existent missions are {len(nonexistent_mission_capture)}')
        print(f'proposals are {len(proposal_capture)}')
        print(f'mission without captureID {len(nocaptureid_mission_capture)}')



def get_args():
    parser = argparse.ArgumentParser(description='Downloads images from firebase')
    parser.add_argument('save_path', type=str)
    parser.add_argument('--firebase_certificate', type=str, default='firebase_certificate.json')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
