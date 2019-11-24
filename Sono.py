# This file is used for the audio to emotion mapping part
# Implement KNN algorithm to use the data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def train_model():
    data = pd.read_csv('Dataset/Audio_features_train.csv')
    # Get all the features starting from tempo
    features = data.loc[:, 'tempo':]
    # Get all the feature names from tempo
    feature_names = list(features)

    # for name in feature_names:
    #     features[name] = (features[name] - features[name].min()) / (features[name].max() - features[name].min())

    plt.style.use('ggplot')

    array = np.array(data)

    features = features.values
    labels = data.loc[:, 'class'].dropna()
    test_size = 0.333
    random_seed = 5

    train_data, test_data, train_label, test_label = train_test_split(features, labels,
                                                                      test_size=test_size, random_state=random_seed)
    n_range = range(1, 80)
    x_label = [i for i in n_range]
    result = find_neighbour_values(n_range, train_data, train_label, test_data, test_label)
    num_neighbours = result.index(max(result))

    # Use the predict function to figure out the emotion of every 5s chunk
    print("Train Data: ", train_data)
    print("Test Data: ", test_data)
    print("Train Label: ", train_label)
    print("Test Label: ", test_label)
    print("Accuracy Results: ", result)
    print("Number of neighbours: ", num_neighbours)

    plt.figure(figsize=(10, 10))
    plt.xlabel('kNN Neighbors')
    plt.ylabel('Accuracy Score')
    plt.title('kNN Classifier Results')
    plt.ylim(0, 100)
    plt.xlim(0, x_label[len(x_label) - 1] + 1)
    plt.plot(x_label, result)
    plt.savefig('1-fold 2NN Result.png')
    plt.show()


def find_neighbour_values(n_range, train_data, train_label, test_data, test_label):
    # Finding out the optimal number of neighbors to fit the model iteratively
    result = []
    for neighbors in n_range:
        knn_model = KNeighborsClassifier(n_neighbors=neighbors)
        knn_model.fit(train_data, train_label)
        prediction = knn_model.predict(test_data)
        print("PREDICTION: ", prediction)
        result.append(accuracy_score(prediction, test_label) * 100)

    return result


def predict_emotion():
    emotion_list = []
    data = pd.read_csv('Dataset/Audio_features_train.csv')
    test_data = pd.read_csv('Dataset/Audio_features.csv')
    # Get all the features starting from tempo
    features = data.loc[:, 'tempo':]
    test_features = test_data.loc[:, 'tempo':]
    # Get all the feature names from tempo
    feature_names = list(features)

    for name in feature_names:
        features[name] = (features[name] - features[name].min()) / (features[name].max() - features[name].min())

    features = features.values
    print("Total Values: ", len(features))
    print("Features: ", features)
    labels = data.loc[:, 'class'].dropna()
    test_size = 0.333
    random_seed = 5

    train_data, test_data, train_label, test_label = train_test_split(features, labels,
                                                                      test_size=test_size, random_state=random_seed)

    knn = KNeighborsClassifier(n_neighbors=8)

    # Train the model using the training sets
    knn.fit(train_data, train_label)

    # Predict the response for test dataset
    prediction = knn.predict(test_features)
    emotion_list.append(prediction)
    # for i in range(len(features)):
    #     prediction = knn.predict(test_features)
    #     emotion_list.append(prediction)
        # print(accuracy_score(prediction, labels) * 100)

    print("Emotions found in the song are: ", emotion_list)
    return emotion_list
