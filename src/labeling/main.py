from src.labeling.prepare_data import load_raw_data, unlabel_data
from src.labeling.visualization import target_countplot

if __name__ == "__main__":
    train, test = load_raw_data("train_data.txt", "test_data.txt")
    target_countplot(train)
    train = unlabel_data(train)
