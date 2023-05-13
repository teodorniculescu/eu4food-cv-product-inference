import argparse
import os
from tqdm import tqdm
import requests
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def get_paths_extra(db):
    images_to_download = []
    keymap_pi_to_pn = {}
    url_check_dict = {}

    product_doc_dict = {x.id: x for x in db.collection('product').get()}

    for product_doc in tqdm(product_doc_dict.values()):
        product_dict = product_doc.to_dict()

        product_id = product_dict['id']
        product_name = product_dict['info']['name']['en']
        keymap_pi_to_pn[product_id] = product_name

        for image_url in product_dict['images']['extra']:
            if image_url in url_check_dict:
                print(image_url, 'already exists')
                continue
            url_check_dict[image_url] = None
            item = {'url': image_url, 'product_id': product_id}
            images_to_download.append(item)
        

    return images_to_download, keymap_pi_to_pn


def get_paths_pcm_reference(db):
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


        if 'rootProposalID' in product_dict:
            root_proposal_id = product_dict['rootProposalID']
            product_id = product_dict['id']
            keymap_rpid_to_pi[root_proposal_id] = product_id

    use_extra_captures = False
    url_check_dict = {}
    images_to_download = []

    nonexistent_mission_capture = []
    proposal_capture = []
    nocaptureid_mission_capture = []
    num_captures = -1

    for capture_doc in tqdm(capture_doc_dict.values()):
        capture_id = capture_doc.id
        num_captures += 1
        capture_dict = capture_doc.to_dict()

        image_url_list = [capture_dict['images']['frontPackagePath']]

        if use_extra_captures:
            for image_url in capture_dict['images']['extra']:
                image_url_list.append(image_url)


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
        #image_url = None
        for item in mission_dict['items']:
            if item['captureID'] == capture_id:
                product_id = item['productID']
                #image_url = item['itemImagePath']
                break

        if product_id is None:
            nocaptureid_mission_capture.append(capture_doc.id)
            continue

        if product_id not in keymap_pi_to_pn:
            print(product_id, 'not in keymap')

        for image_url in image_url_list:
            if image_url in url_check_dict:
                print(image_url, 'already exists')
                continue
            url_check_dict[image_url] = None
            item = {'url': image_url, 'product_id': product_id, 'capture_id': capture_id}
            images_to_download.append(item)


    for capture_doc in tqdm(capture_doc_dict.values()):
        capture_id = capture_doc.id
        capture_dict = capture_doc.to_dict()

        root_proposal_id = capture_dict['ancestorsData']['rootProposalID']

        if root_proposal_id not in keymap_rpid_to_pi:
            continue

        product_id = keymap_rpid_to_pi[root_proposal_id]

        image_url_list = [capture_dict['images']['frontPackagePath']]

        if use_extra_captures:
            for image_url in capture_dict['images']['extra']:
                image_url_list.append(image_url)

        for image_url in image_url_list:
            if image_url in url_check_dict:
                print(image_url, 'already exists')
                continue
            url_check_dict[image_url] = None
            item = {'url': image_url, 'product_id': product_id, 'capture_id': capture_id}
            images_to_download.append(item)

    if len(nonexistent_mission_capture) + len(proposal_capture) + len(nocaptureid_mission_capture) != 0:
        print(f'WARNING: {len(nonexistent_mission_capture) + len(proposal_capture) + len(nocaptureid_mission_capture)} / {num_captures} not saved')
        print(f'non existent missions are {len(nonexistent_mission_capture)}')
        print(f'proposals are {len(proposal_capture)}')
        print(f'mission without captureID {len(nocaptureid_mission_capture)}')

    return images_to_download, keymap_pi_to_pn


def main(args):
    # Use a service account
    cred = credentials.Certificate(args.firebase_certificate)
    firebase_admin.initialize_app(cred)

    # Get a reference to the Firestore database
    db = firestore.client()
    
    print('Getting image url paths')
    if args.method == 'extra':
        images_to_download, product_id_to_name = get_paths_extra(db)
    
    elif args.method == 'product-capture-mission':
        images_to_download, product_id_to_name = get_paths_pcm_reference(db)

    else:
        raise Exception("ERROR: Unknown method of selecting captures")

    print('Creating class directories')
    for product_id in product_id_to_name.keys():
        class_name = product_id
        product_path = os.path.join(args.save_path, class_name)
        os.makedirs(product_path, exist_ok=True)

    print('Downloading images')
    capture_idx = -1
    for item in tqdm(images_to_download):
        capture_idx += 1
        idx_str = str(capture_idx).zfill(10)

        image_url = item['url']
        product_id = item['product_id']
        class_name = product_id

        filename = f'{idx_str}.jpg'


        product_path = os.path.join(args.save_path, class_name, filename)

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(product_path, 'wb') as f:
                f.write(response.content)

        else:
            print('request to address', image_url, 'for class', class_name, 'was unsuccessful')


def get_args():
    parser = argparse.ArgumentParser(description='Downloads images from firebase')
    parser.add_argument('save_path', type=str, help='path where the dataset images will be saved')
    parser.add_argument('--firebase_certificate', type=str, default='firebase_certificate.json', help='path to a json file containing the firebase certificate')
    parser.add_argument('--method', type=str, default='extra', choices=('extra', 'product-capture-mission'), help="selects the method of downloading the captures. extra method download the captures from the extra field of each product while the product-capture-mission matches the capture using the product and mission collections")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
