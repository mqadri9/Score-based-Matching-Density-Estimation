from numpy import load



data = load('run/logs/0/samples/cifa10_test.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
