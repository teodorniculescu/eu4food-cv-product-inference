Setup:

# Product Inference for EU4Food Project
## Author: Teodor-Vicentiu Niculescu
### 1. Set Google Cloud Credentials:

```
gcloud config set project eu4food
gcloud init
```

### 2. Clone Repository:

```
git clone https://github.com/teodorniculescu/eu4food-cv-product-inference.git
```

### 3. Run setup.sh script

```
cd eu4food-cv-product-inference
./setup.sh
```

Note: Accept prompts regarding package installation if necessary.

### 4. Train Model and Upload to Bucket

```
./train.sh
```
