import os
import subprocess

# 캐글 인증
def authenticate_kaggle(api_path="kaggle.json"):
    kaggle_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_path, exist_ok=True)
    kaggle_json_path = os.path.expanduser(api_path)

    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"Kaggle 인증 파일이 없습니다. 경로: {kaggle_json_path}.")

    destination = os.path.join(kaggle_path, "kaggle.json")
    if not os.path.exists(destination):
        subprocess.run(["cp", kaggle_json_path, destination])
        os.chmod(destination, 0o600) 
        print("Kaggle API key configured.")

# kaggle json file으로 데이터세트 받기.
def download_dataset(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    command = [
        "kaggle", "datasets", "download", "-d", dataset, "-p", save_dir, "--unzip"
    ]
    subprocess.run(command, check=True)
    print(f"Dataset {dataset} downloaded and extracted to {save_dir}.")

if __name__ == "__main__":
    DATASET = "skihikingkevin/pubg-match-deaths"
    SAVE_DIR = "./data/raw/"
    API_KEY_PATH = "./kaggle.json"

    print("Kaggle API 인증 중...")
    authenticate_kaggle(api_path=API_KEY_PATH)

    print("인증 완료 후 dataset 다운 중...")
    download_dataset(DATASET, SAVE_DIR)
    print("Download fin.")
