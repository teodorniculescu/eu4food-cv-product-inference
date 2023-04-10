import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

firebase_certificate = 'firebase_certificate.json'

# Use a service account
cred = credentials.Certificate(firebase_certificate)
firebase_admin.initialize_app(cred)

# Get a reference to the Firestore database
db = firestore.client()

# Get all documents from a collection
docs = db.collection('product').get()

# Iterate over the documents and print their data
for doc in docs:
    print(f'{doc.id} => {doc.to_dict()}')
