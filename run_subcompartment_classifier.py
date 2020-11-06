import keras
import numpy as np
from sklearn.metrics import accuracy_score
import argparse


def load_keras_model(model_json,model_h5):
    #load model structure
    json_file = open (model_json, 'r')
    loaded_model_json = json_file.read ()
    json_file.close ()
    model = keras.models.model_from_json (loaded_model_json)

    # load weights into new model
    model.load_weights (model_h5)

    return model


def load_data(test_file,test_labels):
    X_test = np.loadtxt(test_file,delimiter="\t")

    if test_labels != None:
        y_test = np.loadtxt(test_labels,delimiter="\t")
        return X_test,y_test
    return X_test,[]


def classify(model,X_test,predictions_file,y_test):

    y_pred = model.predict_classes (X_test)
    np.savetxt(predictions_file,y_pred)

    #print(y_test)
    if y_test != []:
        acc = accuracy_score (y_test, y_pred)
        print ("Model accuracy is: %f" % acc)



def run_subcompartment_classifier():
    parser = argparse.ArgumentParser(description=("Sub-compartment classifier based on epigenomic and sequence data"))
    parser.add_argument("-x",
                        "--test_data",
                        action="store",
                        dest="x_test",
                        help="File has testing data in form feature",
                        type=str,
                        required=True)

    parser.add_argument ("-y",
                         "--test_labels",
                         action="store",
                         dest="y_test",
                         help="File has testing data true predictions, for benchmarking purpose",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument ("-s",
                         "--model_structure",
                         action="store",
                         dest="model_json",
                         help="JSON file describing neural network structure",
                         type=str,
                         required=True)
    parser.add_argument ("-w",
                         "--model_weights",
                         action="store",
                         dest="model_weights",
                         help="h5 file contains neural network parameters",
                         type=str,
                         required=True)
    parser.add_argument ("-o",
                         "--output_file",
                         action="store",
                         dest="output",
                         help="File name where predictions will be stored",
                         type=str,
                         required=True)

    oArgs = parser.parse_args()
    model = load_keras_model(oArgs.model_json,oArgs.model_weights)
    X_test,y_test = load_data(oArgs.x_test,oArgs.y_test)

    classify(model,X_test,oArgs.output,y_test)
    random_prediction(y_test)

def random_prediction(y_test,n=100):
    possible_labels = [0,1,2,3,4]
    p = 1.0 / len (possible_labels)
    prob = [p for i in possible_labels]

    samples = y_test.shape[0]
    total_acu = 0
    for i in range (n):
        y_pred = np.random.choice (possible_labels, samples, p=prob)
        acc = accuracy_score (y_test, y_pred)
        total_acu += acc
    print ("Uniform random accuracy is: %f" % (total_acu / n))
if __name__ == '__main__':
    run_subcompartment_classifier()