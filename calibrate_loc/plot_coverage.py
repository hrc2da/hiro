import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('move_locs.csv', 'r') as infile:
# with open('cam_locs.csv', 'r') as infile:
    reader = csv.reader(infile)
    next(reader)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.scatter(x,y)
plt.xlim(-400,400)
plt.ylim(-10,400)
plt.savefig("move_coverage.png")
# plt.savefig("cam_coverage.png")