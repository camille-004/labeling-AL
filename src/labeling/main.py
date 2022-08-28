from src.labeling.prepare_data import load_raw_data

if __name__ == "__main__":
    train, test = load_raw_data("train_data.txt", "test_data.txt")
