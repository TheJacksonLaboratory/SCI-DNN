**Sub-compartment classifier**
--------------

**Authors**: Haitham Ashoor, Sheng Li  **Contact**: haitham.ashoor@jax.org, sheng.li@jax.org


## Description
This is model for classifier described in our paper "Graph embedding resolves genomic sub-compartments
 and powers deep neural network predictive model"
 
 
 # Dependencies
 * Python >= 3.5
 * Keras 2.2.4
 * scikit-learn == 0.19.0
 * Numpy >= 1.15
 
 
## Example:
```sh
$ python run_subcompartment_classifier.py -x testing_data.txt -y testing_labels.txt -s sub_compartments_model.json -w 
sub_compartments_model.h5 -o predicted_labes.txt
python run_subcompartment_classifier.py -x testing_data.txt -y testing_labels.txt -s Rao_model.json -w
Rao_model.h5 -o predicted_labes.txt
```

 
