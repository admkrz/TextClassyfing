from data import load_data

if __name__ == '__main__':
    data = load_data()
    print(data.data[0])
    print(data.target_3[0])
    print(data.target_4[0])
