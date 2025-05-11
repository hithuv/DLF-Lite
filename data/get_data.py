import gdown

# url = "https://drive.google.com/file/d/1BuJ21w2BKS5P6Y3sdrUM4OC5AsI8e-xj/view?usp=share_link"
url = "https://drive.google.com/file/d/1_jhuYbVAgVXeTdn1eW8vYtjTyND0LnSO/view?usp=drive_link"
output = "aligned_mosei_dataset.pkl"  # You can change the output filename
gdown.download(url, output, fuzzy=True)

print(f"File downloaded to {output}")