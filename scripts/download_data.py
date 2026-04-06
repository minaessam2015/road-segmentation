from road_segmentation.data.download import download_dataset


if __name__ == "__main__":
    destination = download_dataset()
    print(f"Dataset saved to: {destination}")
