from config.config import logger
from src.labeling.modeling import clf_eval
from src.labeling.prepare_data import (
    clean_text,
    load_raw_data,
    split_transform_data,
    unlabel_data,
    view_examples,
)
from src.labeling.visualization import target_countplot

if __name__ == "__main__":
    train, test = load_raw_data("train_data.txt", "test_data.txt")
    target_countplot(train)
    labeled, unlabeled = unlabel_data(train)
    labeled = labeled.assign(synopsis=labeled["synopsis"].apply(clean_text))
    unlabeled = unlabeled.assign(
        synopsis=unlabeled["synopsis"].apply(clean_text)
    )
    logger.info("Synoptic texts cleaned.")
    view_examples(labeled)
    X_train, X_val, y_train, y_val, X_pool = split_transform_data(
        labeled, unlabeled
    )
    print(clf_eval(X_train, X_val, y_train, y_val))
