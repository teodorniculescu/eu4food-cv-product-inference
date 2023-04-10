Setup:

# Product Inference for EU4Food Project
## Author: Teodor-Vicentiu Niculescu
### 1.1. Setup: Set Google Cloud Credentials:

```
gcloud config set project eu4food
gcloud init
```

### 1.2. Setup: Clone Repository:

```
git clone https://github.com/teodorniculescu/eu4food-cv-product-inference.git
```

### 1.3. Setup: Add Firebase Certificate:

Upload json file which constains the firebase certificate to the newly cloned project and rename the file as firebase_certificate.json

### 2. Run setup.sh script

```
cd eu4food-cv-product-inference
./setup.sh
```

Note: Accept prompts regarding package installation if necessary.

### 3. Train Model and Upload to Bucket

```
./train.sh
```
