import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cityblock
import statistics
#####
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def KNN_jay(train_features, train_labels, test_features, k, dist_fun, weighted = False):

    predictions = []

    ###########################
    ## YOUR CODE BEGINS HERE
    ###########################

    def most_frequent(List):
        return max(set(List), key = List.count)

    distances_test = []
    for testi in test_features:
        dist = []
        for traini in train_features:
            dist.append(dist_fun(testi, traini))

        # Indices of k smallest elements in list
        res = sorted(range(len(dist)), key = lambda sub: dist[sub], reverse = True)[-k:]
        nearest_neighbours = list(map(train_labels.__getitem__, res))
        predictions.append(most_frequent(nearest_neighbours))



    ###########################
    ## YOUR CODE BEGINS HERE
    ###########################

    return predictions
def custom_preprocessing(original_data):
    numerical_data = [[] for i in original_data]
    #   Numerizing the data so that all the data is in numeric form.
    for index in range(len(original_data[0])):
        str_feat_value= [values[index] for values in original_data]
        num_feat_values = standardize_function(str_feat_value)
        for idx, instance in enumerate(original_data):
            #           print(idx , '\n', instance)
            numerical_data[idx].append(num_feat_values[idx])
    return numerical_data

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))
def KNN_checker(train_features, train_labels, test_features, k, dist_fun, weighted=False):
    print(test_features)
    predictions = []
    labels_colors = [values[1] for values in train_labels]
    labels_colors = list(set(labels_colors))
    print('the labels_colors are :', labels_colors)
    # creating the distance matrix for all the test instances against all the features
    count =1
    for index, test_instance in enumerate(test_features):
        distance_vector = [[] for value in train_features]
        dummy_distance = []
        for index2, train_instance in enumerate(train_features):
            distance = dist_fun(test_instance,train_instance)
            distance_vector[index2].append(distance)
            dummy_distance.append(distance)
            distance_vector[index2].extend(train_labels[index2])
        print('the unsorted distance list is : ', distance_vector)
        distance_vector.sort(key = lambda x: x[0])
        predictions = distance_vector
        break
    return predictions
def KNN(train_features, train_labels, test_features, k, dist_fun, weighted=False):

    predictions = []
    labels_colors = [values[1] for values in train_labels]
    labels_colors = list(set(labels_colors))
    # creating the distance matrix for all the test instances against all the features
    for index, test_instance in enumerate(test_features):
        distance_vector = [[] for value in train_features]
        dummy_distance = []
        for index2, train_instance in enumerate(train_features):
            distance = dist_fun(test_instance,train_instance)
            distance_vector[index2].append(distance)
            dummy_distance.append(distance)
            distance_vector[index2].extend(train_labels[index2])
        distance_vector.sort(key = lambda x: x[0])
        dummy_distance.sort()
        closest_neighbors = []
        mainhue = []
        mainhue2 = []
        predicted_color = ''
        k_distance_vector= distance_vector[:k]
        for values in k_distance_vector:
            closest_neighbors.append(values[1])
            mainhue.append(labels_colors.index(values[2]))
            mainhue2.append(values[2])
        if weighted is False:
            predicted_color = max(set(mainhue2), key = mainhue2.count)
            predicted_color = stats.mode(mainhue2)
            print('predicted color is ', predicted_color)
            exit()
        else:
            set_mainhue = list(set(mainhue2))
            weighted_distance_vector = [[] for i in range(len(set_mainhue))]
            result_weighted =  []
            for values in k_distance_vector:
                weighted_distance_vector[set_mainhue.index(values[2])].append(values[0])
            epsilon = 0.00001
            for index, values in enumerate(weighted_distance_vector):
                result_weighted.append(math.fsum([(1/(i+epsilon)) for i in values]))
            predicted_color = set_mainhue[result_weighted.index(max(result_weighted))]
        predictions.append(predicted_color)
    return predictions

def manhattan_distance(fw1, fw2):
    assert len(fw1) == len(fw2), "Feature vectors are of different sizes"
    distance = math.fsum([math.fabs(fw1[i]-fw2[i]) for i in range(len(fw1))])
    return distance
def cosine_distance(fw1, fw2):
    assert len(fw1) == len(fw2), "Feature vectors are of different sizes"
    distance = 1-((math.fsum([fw1[i]*fw2[i]for i in range(len(fw1))]))/((math.sqrt(math.fsum([math.pow(i,2) for i in fw1])))*(math.sqrt(math.fsum([math.pow(i,2) for i in fw2])))))
    return distance
def eucledian_distance(a,b):
    assert len(a)==len(b), "Arrays are of different sizes"
    return np.sqrt(sum([(a[i]-b[i])*(a[i]-b[i]) for i in range(len(a))]))
def string_features_to_numeric_function(str_values):

    print('the integer value for the str_values are ', str_values)
    exit()
    str_values_set = list(set(str_values)) #set would remove the duplicates and put the things in the order form I guess.

    str_values_set.sort()
    numeric_values = []
    for str_value in str_values:
        num_value = str_values_set.index(str_value) # basically index will bring out the index of the list item matching the value of the vairable that has been passed to the index function.
        numeric_values.append(num_value)
    return numeric_values

def normalize_function(str_values):
    str_values = [int(x) for x in str_values]
    minimum_value=min(str_values)
    maximum_value=max(str_values)
    numeric_values = [(x-minimum_value)/(maximum_value-minimum_value) for x in str_values]
    return numeric_values

def standardize_function(str_values):
    str_values = [int(x) for x in str_values]
    mean_value=np.mean(str_values)
    std_value=np.std(str_values)
    numeric_values = [(x-mean_value)/(std_value) for x in str_values]
    return numeric_values

def integer_typecasting(features):

    for index, feature in enumerate(features):
        features[index]=[int(i) for i in feature]
    return(features)
def main():
    #loading the flag features data in the data variable.
    data = open("flags.features", 'r').readlines()
    #loading the labels data in the labels variable.
    labels = open("flags.labels", 'r').readlines()

    #initializing the training and test lists.
    train_features = []
    train_labels   = []
    test_features = []
    test_labels   = []

    # removing the first list (row) because it is the description of each feature of the data and putting it in the data attributes vairable
    data_attributes = data.pop(0)
    #formatting the data so that each attribute is represented in the form of a list-item.
    data_attributes=data_attributes.replace("\n","")
    data_attributes=data_attributes.split(',')

    # doing the same thing for the labels attributes as well.
    labels_attributes = labels.pop(0)
    labels_attributes = labels_attributes.replace("\n","")
    labels_attributes = labels_attributes.split(',')

    #for the rest of the data, formatting the data so that each instance data against the attributes becomes the list item in a list and the training list becomes the list consisting of each instance's lists.
    for val in data[:150]:
        val=val.replace("\n","")
        val=val.split(',')
        train_features.append(val)
    #segregating and formatting the rest of the data as the test data.
    for val in data[150:]:
        val=val.replace("\n","")
        val=val.split(',')
        test_features.append(val)
    # the same procedure goes for the labels data corresponding to each instance. Both training labels and the testing labels lists are created below.
    for instance in labels[:150]:
        instance=instance.replace("\n","")
        instance=instance.split(',')
        train_labels.append(instance)
    for instance in labels[150:]:
        instance=instance.replace("\n","")
        instance=instance.split(',')
        test_labels.append(instance)

    test_features_structure = []
    train_features_structure = []
    for val in train_features:
        val= val[5:]
        train_features_structure.append(val)

    for val in test_features:
        val= val[5:]
        test_features_structure.append(val)

    train_numeric_structure_features = custom_preprocessing(train_features_structure)
    test_numeric_structure_features = custom_preprocessing(test_features_structure)

    train_features_1=train_features
    test_features_1=test_features
    for index in range(len(train_features)):
        train_features_1[index].pop(0)
    for index in range(len(test_features)):
        test_features_1[index].pop(0)

    train_numeric_features = custom_preprocessing(train_features_1)
    test_numeric_features = custom_preprocessing(test_features_1)

    print('this is the manhattan distance between all the train_features instance from the test_features instance using the default function.')
    predictions_1 = [cityblock(test_numeric_features[0],train_numeric_features[index]) for index in range(len(train_numeric_features))]
    print(predictions_1)

    predictions = KNN(train_numeric_features, train_labels, test_numeric_features , 5, manhattan_distance, False)
    true_test_labels = [values[1] for values in test_labels]

    print('predictions are in the main module',predictions)
    print('actual test labels are', true_test_labels)
    print('accuracy score for the manhattan distance is ',accuracy_score(predictions,true_test_labels))
    predictions = KNN(train_numeric_features, train_labels, test_numeric_features , 5, eucledian_distance, True)
    print('accuracy score for the eucledian distance is ',accuracy_score(predictions,true_test_labels))
    ######### this is the using the already built knn classifier #######
    le = preprocessing.LabelEncoder()
    true_train_labels = [values[1] for values in train_labels]
    encoded_labels = le.fit_transform(true_train_labels)
    print('actual train labels are ', true_train_labels)
    print('encoded labels are ',encoded_labels)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_numeric_features,encoded_labels)
    predicted = model.predict(test_numeric_features)
    print('predicted in the numeric form', predicted)
    print('predicted is ', le.inverse_transform(predicted))
    print(accuracy_score(le.inverse_transform(predicted),true_test_labels))

    list_labels_colors = [values[1] for values in train_labels]
    list_labels_colors = list(set(list_labels_colors))
    bar_width=0.35
    plt.bar(list_labels_colors,[predictions.count(value) for value in list_labels_colors],bar_width,align = 'center',label='Predicted')
    plt.bar(list_labels_colors,[true_test_labels.count(value) for value in list_labels_colors],bar_width,align = 'edge',label='Actual Colors')
    plt.legend()
    plt.xlabel('Colors')
    plt.ylabel('Frequency of colors')
    plt.title('Predictions by KNN classifier vs Actual Colors')
    plt.show()

if __name__ == "__main__":
    main()