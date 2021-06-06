import pandas as pd

def fuel_data():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

    # Train test split
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset  = dataset.drop(train_dataset.index)

    # Split off labels
    trainX = train_dataset.copy()
    trainY = trainX.pop('MPG')
    testX  = test_dataset.copy()
    testY  = testX.pop('MPG')
    
    return(trainX, trainY, testX, testY)