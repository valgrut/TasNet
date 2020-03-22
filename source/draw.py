import matplotlib.pyplot as plt
import csv

# directories = ["2020-03-14_03:31_X7_R3", "2020-03-22_02:57_X1_R1"]

directories = ["2020-03-22_02:57_X1_R1"]
for dir in directories:
    # training loss
    x = []
    y = []
    losspath = "/root/TEST/" + dir + "/training_loss.log"
    with open(losspath,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1])/70)   #pozor, delim to zde velikosti minibatche

    plt.plot(x,y, label='Training loss of '+dir)
    
    # Validation loss
    x = []
    y = []
    losspath = "/root/TEST/" + dir + "/validation_loss.log"
    with open(losspath,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))

    plt.plot(x,y, label='Training loss of '+dir)


#plt.plot(x,y, label='Training loss')
plt.xlabel('number of processed segments')
plt.ylabel('loss')
plt.title('Graph of training loss')
plt.legend()
plt.show()
