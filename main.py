import pandas as pd


def read_data(filepath='test.csv'):
    data = pd.read_csv(filepath)
    print(data.shape)




if __name__ == '__main__':
    read_data()





