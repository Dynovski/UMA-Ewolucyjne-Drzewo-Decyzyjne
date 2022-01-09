import data_processing.data_loader as loader
import algorithm.utils as dt
import visualizer.visualizer as vs


def test():
    data_set_type = loader.Dataset.IRIS
    data = loader.load_data(data_set_type)
    tree = dt.get_basic_trained_tree_classifier(data)
    matrix = dt.get_confusion_matrix(tree, data)
    vs.save_confusion_matrix(matrix, data_set_type, "test")


if __name__ == "__main__":
    test()
