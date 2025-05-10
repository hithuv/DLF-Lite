import gdown

url = "https://drive.google.com/file/d/1BuJ21w2BKS5P6Y3sdrUM4OC5AsI8e-xj/view?usp=share_link"
output = "aligned_mosei_dataset.npy"  # You can change the output filename
gdown.download(url, output, fuzzy=True)

print(f"File downloaded to {output}")